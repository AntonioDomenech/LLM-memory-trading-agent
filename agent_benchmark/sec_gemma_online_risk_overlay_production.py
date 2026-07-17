"""Source-bound production authorities for the v2.2 overlay.

This module owns the concrete external HTTP session used by the reviewed SEC
transport.  It deliberately does not expose a generic transport factory:
production objects can be created only from an opaque live source-tree
verification for the clean pushed implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import ctypes
from ctypes import wintypes
import hashlib
import importlib.metadata
import json
import math
try:
    import msvcrt
except ImportError:  # pragma: no cover - production publication is Windows-only.
    msvcrt = None  # type: ignore[assignment]
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    NUMERICAL_TIME_IMPORT_ROOTS,
    SOURCE_VERIFICATION_SCHEMA_VERSION,
    VerifiedSourceTree,
    is_verified_source_tree,
    load_allowed_requests,
    source_verification_material,
)

requests = load_allowed_requests()
from requests.adapters import HTTPAdapter

from agent_benchmark.sec_audit_transport import SecAuditTransport
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    FINAL_ATTEMPT_ID,
    MAX_DETERMINISTIC_SECONDS,
    MAX_MODEL_SECONDS,
    MAX_PUBLICATION_RECOVERY_SECONDS,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    MODEL_NAME,
    NEW_SOURCE_FILES,
    PUBLICATION_NORMAL_OPERATION_SHA256,
    PUBLICATION_WORKER_OWNERSHIP_FIELDS,
    canonical_json_bytes,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_baseline import (
    build_baseline_input_row,
    build_frozen_baseline_signal_batch,
)
from agent_benchmark.sec_gemma_online_risk_overlay_features import (
    REQUIRED_PREFIX_ROWS,
    _mint_feature_row_from_source_bound_components,
)
from agent_benchmark.sec_point_in_time import (
    MAX_AUDIT_BYTES,
    MAX_AUDIT_REQUESTS,
    MAX_AUDIT_SECONDS,
    BudgetCounter,
    validate_sec_user_agent,
)


PRODUCTION_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-production-authority-v2"
)
PRODUCTION_AUTHORITY_VERIFIER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-production-authority-verifier-v1"
)
PRODUCTION_SEC_SESSION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-sec-session-v1"
)
PRODUCTION_AUTHORITIES_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-production-authorities-v1"
)
PRODUCTION_ACQUISITION_ADAPTER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-production-acquisition-adapter-v1"
)
PRODUCTION_PHASE_EXECUTOR_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-production-phase-executor-v1"
)
PRODUCTION_PUBLICATION_RUNTIME_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-publication-worker-runtime-v1"
)
SEMANTIC_EXTRACTION_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-semantic-extraction-row-v1"
)
SEMANTIC_EVENT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-semantic-event-receipt-v1"
)
SEMANTIC_BATCH_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-semantic-batch-receipt-v1"
)
LATENCY_PREFLIGHT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-latency-preflight-v1"
)
PRODUCTION_PUBLICATION_RECOVERY_REQUEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-"
    "production-publication-recovery-request-v1"
)
SUPERVISED_PUBLICATION_RECOVERY_WORKER_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-"
    "supervised-publication-recovery-worker-authority-v1"
)

_AUTHORITY_SENTINEL = object()
_AUTHORITIES_SENTINEL = object()
_ADAPTER_SENTINEL = object()
_EXECUTOR_SENTINEL = object()
_PUBLICATION_RUNTIME_SENTINEL = object()
_PUBLICATION_HANDLE_SENTINEL = object()
_RECOVERY_WORKER_AUTHORITY_SENTINEL = object()
_MARKET_SYMBOLS: Final[tuple[str, ...]] = (
    "AAPL",
    "SPY",
    "QQQ",
    "IWM",
    "VIX",
    "TNX",
)
_FEATURE_MARKET_SYMBOLS: Final[tuple[str, ...]] = (
    "AAPL",
    "QQQ",
    "SPY",
    "IWM",
    "VIX",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_BY_COMMAND: Final[dict[str, str]] = {
    "development_acquisition": "development",
    "development": "development",
    "confirmation": "confirmation",
    "final": "final",
}
_SCORING_COMMANDS: Final[frozenset[str]] = frozenset(
    {"development", "confirmation", "final"}
)
_RECOVERY_COMMAND_BY_ATTEMPT_ID: Final[dict[str, str]] = {
    DEVELOPMENT_ACQUISITION_ID: "development_acquisition",
    DEVELOPMENT_ATTEMPT_ID: "development",
    CONFIRMATION_ATTEMPT_ID: "confirmation",
    FINAL_ATTEMPT_ID: "final",
}
_ZERO_COUNTERS: Final[dict[str, int]] = {
    "sec_request_count": 0,
    "market_request_count": 0,
    "model_call_count": 0,
    "retry_count": 0,
    "fallback_count": 0,
    "model_pull_count": 0,
    "paid_api_call_count": 0,
}
_OLLAMA_CHAT_ENDPOINT: Final[str] = (
    "http://127.0.0.1:11434/api/chat"
)


class SecGemmaOnlineRiskOverlayProductionError(RuntimeError):
    """A requested production authority or transport is not exact."""


def _build_owned_sec_session() -> requests.Session:
    """Build the exact raw session required by production acquisition."""

    session = requests.Session()
    if type(session) is not requests.Session:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC session construction returned a foreign type"
        )
    session.trust_env = False
    session.auth = None
    session.cookies.clear()
    session.headers.clear()
    session.proxies.clear()
    adapter = HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if (
        session.trust_env is not False
        or session.auth is not None
        or len(session.cookies) != 0
        or len(session.headers) != 0
        or session.proxies != {}
        or session.get_adapter("https://") is not adapter
        or session.get_adapter("http://") is not adapter
        or adapter.max_retries.total != 0
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC session security state changed"
        )
    return session


class VerifiedProductionAuthority:
    """Opaque clean-source authority for production object construction."""

    __slots__ = ("_payload", "_repo_root", "_sentinel")

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        repo_root: Path,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _AUTHORITY_SENTINEL:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production authority can only be issued by its verifier"
            )
        self._payload = dict(payload)
        self._repo_root = repo_root
        self._sentinel = _sentinel

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def head_commit(self) -> str:
        return self._payload["head_commit"]

    @property
    def authority_sha256(self) -> str:
        return self._payload["authority_sha256"]

    def __repr__(self) -> str:
        return "VerifiedProductionAuthority(<source-bound>)"


def is_verified_production_authority(value: Any) -> bool:
    return (
        type(value) is VerifiedProductionAuthority
        and getattr(value, "_sentinel", None) is _AUTHORITY_SENTINEL
    )


def issue_verified_production_authority(
    verified_source_tree: VerifiedSourceTree,
) -> VerifiedProductionAuthority:
    """Issue authority only for the exact clean pushed v2.2 implementation."""

    if not is_verified_source_tree(verified_source_tree):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production authority requires live source verification"
        )
    material = source_verification_material(verified_source_tree)
    if (
        material["schema_version"] != SOURCE_VERIFICATION_SCHEMA_VERSION
        or material["contract_sha256"] != CONTRACT_SHA256
        or material["branch"] != (
            "codex/aapl-sec-gemma-online-risk-overlay-v2-2"
        )
        or material["head_commit"] != material["upstream_commit"]
        or {
            item["role"]: item["path"]
            for item in material["new_sources"]
        }
        != dict(NEW_SOURCE_FILES)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Live source verification is not the exact v2.2 implementation"
        )
    body = {
        "schema_version": PRODUCTION_AUTHORITY_SCHEMA_VERSION,
        "verifier_id": PRODUCTION_AUTHORITY_VERIFIER_ID,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "source_verification_sha256": verified_source_tree.verification_sha256,
        "dependency_closure_sha256": material[
            "dependency_closure_sha256"
        ],
        "head_commit": material["head_commit"],
        "upstream_commit": material["upstream_commit"],
        "external_distributions_sha256": canonical_sha256(
            material["external_distributions"]
        ),
        "numerical_time_distributions_sha256": canonical_sha256(
            material["numerical_time_distributions"]
        ),
    }
    payload = {**body, "authority_sha256": canonical_sha256(body)}
    return VerifiedProductionAuthority(
        payload=payload,
        repo_root=verified_source_tree.repo_root,
        _sentinel=_AUTHORITY_SENTINEL,
    )


def create_production_sec_transport(
    authority: VerifiedProductionAuthority,
    *,
    user_agent: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    deadline_monotonic: float,
) -> SecAuditTransport:
    """Construct the exact reviewed SEC transport with tighter inherited caps."""

    if not is_verified_production_authority(authority):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC transport requires production authority"
        )
    if clock is not time.monotonic or sleep is not time.sleep:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC transport requires exact production clock authorities"
        )
    try:
        now = float(clock())
        deadline = float(deadline_monotonic)
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC deadline clock failed"
        ) from None
    remaining = deadline - now
    if (
        not math.isfinite(now)
        or not math.isfinite(deadline)
        or not 0.0 < remaining <= MAX_SEC_SECONDS
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC transport deadline is outside the frozen budget"
        )
    budget = BudgetCounter(
        clock=clock,
        max_requests=min(MAX_SEC_REQUESTS, MAX_AUDIT_REQUESTS),
        max_bytes=min(MAX_SEC_BYTES, MAX_AUDIT_BYTES),
        max_seconds=min(remaining, MAX_AUDIT_SECONDS),
    )
    budget_started = float(getattr(budget, "_started_at", math.nan))
    exact_budget_seconds = deadline - budget_started
    if (
        not math.isfinite(budget_started)
        or not math.isfinite(exact_budget_seconds)
        or exact_budget_seconds <= 0.0
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC transport construction exhausted its parent deadline"
        )
    budget.max_seconds = min(
        float(budget.max_seconds),
        exact_budget_seconds,
    )
    budget.check()
    session = _build_owned_sec_session()
    cache_dir = (
        authority.repo_root
        / "data"
        / "sec_gemma_online_risk_overlay"
        / CONTRACT_SHA256
        / "disabled_sec_cache"
    )
    transport = SecAuditTransport(
        session=session,
        cache_dir=cache_dir,
        user_agent=user_agent,
        budget=budget,
        clock=clock,
        sleep=sleep,
        timeout_seconds=min(30.0, remaining),
        max_retries=0,
        max_redirects=0,
        allow_cache_reads=False,
        allow_cache_writes=False,
    )
    if type(transport) is not SecAuditTransport:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "SEC transport construction returned a foreign type"
        )
    transport._parent_deadline_monotonic = deadline
    return transport


def _positive_float_hex(value: Any, location: str) -> float:
    if type(value) is not str:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be canonical float.hex text"
        )
    try:
        number = float.fromhex(value)
    except ValueError:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be canonical float.hex text"
        ) from None
    if (
        not math.isfinite(number)
        or number <= 0.0
        or number.hex() != value
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be a positive canonical float"
        )
    return number


def _stage_slice_market_material(
    stage_slice: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the cutoff-safe vault slice into replay and baseline inputs."""

    if type(stage_slice) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Stage slice must be one exact detached mapping"
        )
    rows_by_symbol = stage_slice.get("market_rows")
    if (
        type(rows_by_symbol) is not dict
        or set(rows_by_symbol) != set(_MARKET_SYMBOLS)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Stage slice market symbols changed"
        )
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    ordered_sessions: list[str] = []
    for symbol in _MARKET_SYMBOLS:
        raw_rows = rows_by_symbol[symbol]
        if type(raw_rows) is not list or not raw_rows:
            raise SecGemmaOnlineRiskOverlayProductionError(
                f"{symbol} stage rows are empty or invalid"
            )
        prior: str | None = None
        symbol_rows: dict[str, dict[str, Any]] = {}
        for ordinal, raw in enumerate(raw_rows, start=1):
            if (
                type(raw) is not dict
                or set(raw) != {"session", "available", "values"}
                or type(raw["session"]) is not str
                or type(raw["available"]) is not bool
                or raw["session"] in symbol_rows
                or (prior is not None and raw["session"] <= prior)
            ):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    f"{symbol} stage row {ordinal} is noncanonical"
                )
            if raw["available"]:
                values = raw["values"]
                if (
                    type(values) is not dict
                    or set(values)
                    != {
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "adjusted_close",
                    }
                ):
                    raise SecGemmaOnlineRiskOverlayProductionError(
                        f"{symbol} stage values changed"
                    )
                for field in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                ):
                    _positive_float_hex(
                        values[field], f"{symbol}.{raw['session']}.{field}"
                    )
                volume = values["volume"]
                if type(volume) is not str:
                    raise SecGemmaOnlineRiskOverlayProductionError(
                        f"{symbol} volume is noncanonical"
                    )
                try:
                    volume_number = float.fromhex(volume)
                except ValueError:
                    raise SecGemmaOnlineRiskOverlayProductionError(
                        f"{symbol} volume is noncanonical"
                    ) from None
                if (
                    not math.isfinite(volume_number)
                    or volume_number < 0.0
                    or volume_number.hex() != volume
                ):
                    raise SecGemmaOnlineRiskOverlayProductionError(
                        f"{symbol} volume is noncanonical"
                    )
            elif raw["values"] is not None:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    f"{symbol} unavailable row exposes values"
                )
            symbol_rows[raw["session"]] = copy.deepcopy(raw)
            prior = raw["session"]
        indexed[symbol] = symbol_rows
        if symbol == "AAPL":
            ordered_sessions = list(symbol_rows)
            if any(
                not row["available"] for row in symbol_rows.values()
            ):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "AAPL stage prefix contains an unavailable row"
                )
    if (
        not ordered_sessions
        or ordered_sessions[0] != "2000-01-03"
        or ordered_sessions[-1] != stage_slice.get("last_value_session")
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "AAPL stage prefix does not match the frozen cutoff"
        )

    market_rows: list[dict[str, str]] = []
    baseline_inputs: list[dict[str, Any]] = []
    feature_rows_by_session: dict[str, dict[str, Any]] = {}
    for session in ordered_sessions:
        aapl = indexed["AAPL"][session]
        assert aapl["available"]
        aapl_values = aapl["values"]
        raw_open = _positive_float_hex(
            aapl_values["open"], f"AAPL.{session}.open"
        )
        raw_close = _positive_float_hex(
            aapl_values["close"], f"AAPL.{session}.close"
        )
        adjusted_close = _positive_float_hex(
            aapl_values["adjusted_close"],
            f"AAPL.{session}.adjusted_close",
        )
        adjusted_open = raw_open * adjusted_close / raw_close
        if not math.isfinite(adjusted_open) or adjusted_open <= 0.0:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Derived AAPL adjusted open is invalid"
            )
        market_rows.append(
            {
                "session": session,
                "adjusted_open_hex": adjusted_open.hex(),
                "adjusted_close_hex": adjusted_close.hex(),
            }
        )
        context_values: dict[str, float] = {}
        observations: dict[str, dict[str, Any]] = {}
        for symbol in _FEATURE_MARKET_SYMBOLS:
            source = indexed[symbol].get(session)
            if source is None or not source["available"]:
                observations[symbol] = {
                    "available": False,
                    "adjusted_close_hex": None,
                }
                continue
            value = _positive_float_hex(
                source["values"]["adjusted_close"],
                f"{symbol}.{session}.adjusted_close",
            )
            observations[symbol] = {
                "available": True,
                "adjusted_close_hex": value.hex(),
            }
            context_values[symbol] = value
        feature_rows_by_session[session] = {
            "session": session,
            "observations": observations,
        }
        if "SPY" not in context_values or "QQQ" not in context_values:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "SPY or QQQ is unavailable for the frozen baseline"
            )
        evidence_body = {
            "stage_slice_sha256": stage_slice.get("slice_sha256"),
            "session": session,
            "aapl_row_sha256": canonical_sha256(aapl),
            "spy_row_sha256": canonical_sha256(indexed["SPY"][session]),
            "qqq_row_sha256": canonical_sha256(indexed["QQQ"][session]),
        }
        baseline_inputs.append(
            build_baseline_input_row(
                session=session,
                aapl_raw_open=raw_open,
                aapl_raw_close=raw_close,
                aapl_adjusted_close=adjusted_close,
                spy_adjusted_close=context_values["SPY"],
                qqq_adjusted_close=context_values["QQQ"],
                market_evidence_row_sha256=canonical_sha256(evidence_body),
            )
        )

    baseline_batch = build_frozen_baseline_signal_batch(
        baseline_inputs
    )
    source_commitments = {
        "stage_slice_sha256": stage_slice.get("slice_sha256"),
        "market_symbol_prefix_sha256s": {
            symbol: canonical_sha256(rows_by_symbol[symbol])
            for symbol in _MARKET_SYMBOLS
        },
        "market_rows_sha256": canonical_sha256(market_rows),
        "baseline_input_rows_sha256": canonical_sha256(baseline_inputs),
        "baseline_batch_sha256": baseline_batch[
            "baseline_batch_sha256"
        ],
    }
    return {
        "market_rows": market_rows,
        "baseline_input_rows": baseline_inputs,
        "baseline_signals": baseline_batch["signal_rows"],
        "feature_market_rows_by_session": feature_rows_by_session,
        "ordered_sessions": ordered_sessions,
        "source_commitments": {
            **source_commitments,
            "source_commitments_sha256": canonical_sha256(
                source_commitments
            ),
        },
    }


def _decision_market_lookback_rows(
    market_material: Mapping[str, Any],
    decision_session: str,
) -> list[dict[str, Any]]:
    sessions = market_material.get("ordered_sessions")
    rows = market_material.get("feature_market_rows_by_session")
    if (
        type(sessions) is not list
        or type(rows) is not dict
        or decision_session not in rows
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Filing decision session is outside the stage market prefix"
        )
    position = sessions.index(decision_session)
    first = max(0, position - REQUIRED_PREFIX_ROWS + 1)
    selected = sessions[first : position + 1]
    return [copy.deepcopy(rows[session]) for session in selected]


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _strict_json_mapping_bytes(
    value: bytes,
    *,
    location: str,
) -> dict[str, Any]:
    if type(value) is not bytes or not value:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be nonempty exact bytes"
        )

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    f"{location} contains a duplicate JSON key"
                )
            result[key] = child
        return result

    try:
        parsed = json.loads(
            value.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SecGemmaOnlineRiskOverlayProductionError(
                    f"{location} contains nonfinite JSON token {token}"
                )
            ),
        )
    except SecGemmaOnlineRiskOverlayProductionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} is not strict UTF-8 JSON"
        ) from exc
    if type(parsed) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must contain one exact JSON object"
        )
    return parsed


def _runtime_probe_payload_sha256(
    runtime_payload: Mapping[str, Any],
) -> str:
    if type(runtime_payload) is not dict or set(runtime_payload) != {
        "version_response_hex",
        "show_response_hex",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Runtime identity payload keys changed"
        )
    for key in ("version_response_hex", "show_response_hex"):
        value = runtime_payload[key]
        if type(value) is not str or not value or len(value) % 2:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Runtime identity payload is not exact hexadecimal bytes"
            )
        try:
            decoded = bytes.fromhex(value)
        except ValueError:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Runtime identity payload is not exact hexadecimal bytes"
            ) from None
        if not decoded or decoded.hex() != value:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Runtime identity payload is noncanonical"
            )
    return canonical_sha256(runtime_payload)


def _runtime_identity_payload(
    *,
    transport: Any | None = None,
) -> dict[str, str]:
    """Probe only version and show; this path never invokes model generation."""

    from agent_benchmark.sec_filing_gemma_ollama import (
        OLLAMA_SHOW_ENDPOINT,
        OLLAMA_VERSION_ENDPOINT,
        _PROBE_GET_HEADERS,
        _REQUEST_HEADERS,
        _perform_runtime_probe_request,
        _runtime_probe_show_request_bytes,
        build_hardened_loopback_session,
    )

    owned = transport is None
    active = build_hardened_loopback_session() if owned else transport
    if active is None or not callable(getattr(active, "request", None)):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Runtime probe transport is unavailable"
        )
    if owned and type(active) is not requests.Session:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Runtime probe returned a foreign production transport"
        )
    try:
        version_bytes, _version_http = _perform_runtime_probe_request(
            active,
            method="GET",
            endpoint=OLLAMA_VERSION_ENDPOINT,
            headers=_PROBE_GET_HEADERS,
            request_bytes=None,
        )
        show_bytes, _show_http = _perform_runtime_probe_request(
            active,
            method="POST",
            endpoint=OLLAMA_SHOW_ENDPOINT,
            headers=_REQUEST_HEADERS,
            request_bytes=_runtime_probe_show_request_bytes(),
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Pinned local runtime identity probe failed closed"
        ) from None
    finally:
        if owned:
            try:
                active.close()
            except Exception:
                pass
    payload = {
        "version_response_hex": version_bytes.hex(),
        "show_response_hex": show_bytes.hex(),
    }
    _runtime_probe_payload_sha256(payload)
    return payload


def _validate_blinded_model_request(
    request: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> tuple[bytes, tuple[str, ...], dict[str, Any]]:
    from agent_benchmark.sec_filing_gemma_contract import (
        build_extractor_model_payload,
    )

    if type(request) is not dict or set(request) != {
        "schema_version",
        "accession_number",
        "form",
        "availability_session",
        "preprocessed_event_sha256",
        "supplied_sentence_ids",
        "request_sha256",
        "request_bytes",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request fields changed"
        )
    request_bytes = request["request_bytes"]
    if (
        type(request_bytes) is not bytes
        or hashlib.sha256(request_bytes).hexdigest()
        != _sha256(request["request_sha256"], "model request")
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request bytes changed"
        )
    payload = _strict_json_mapping_bytes(
        request_bytes,
        location="blinded model request",
    )
    try:
        messages = payload["messages"]
        user_content = messages[1]["content"]
    except (KeyError, IndexError, TypeError):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request shape changed"
        ) from None
    if type(user_content) is not str:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request user content changed"
        )
    user_payload = _strict_json_mapping_bytes(
        user_content.encode("utf-8"),
        location="blinded model request user content",
    )
    if set(user_payload) != {"sentences"} or type(
        user_payload["sentences"]
    ) is not list:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request sentence payload changed"
        )
    try:
        rebuilt = build_extractor_model_payload(
            user_payload["sentences"]
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request cannot be rebuilt"
        ) from None
    from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
        canonical_json_bytes,
    )

    if rebuilt != payload or canonical_json_bytes(rebuilt) != request_bytes:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request is not the frozen canonical payload"
        )
    sentence_ids = tuple(
        item.get("id") if type(item) is dict else None
        for item in user_payload["sentences"]
    )
    supplied = request["supplied_sentence_ids"]
    if (
        type(supplied) is not list
        or tuple(supplied) != sentence_ids
        or not sentence_ids
        or len(sentence_ids) != len(set(sentence_ids))
        or any(type(item) is not str for item in sentence_ids)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request sentence IDs changed"
        )
    current = proof.get("current_record")
    if (
        type(current) is not dict
        or current.get("accession_number") != request["accession_number"]
        or current.get("form") != request["form"]
        or current.get("availability_session")
        != request["availability_session"]
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Blinded model request crossed its universe event"
        )
    return request_bytes, sentence_ids, payload


def _validate_universe_proof(
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_features import (
        validate_universe_event_proof,
    )

    if type(proof) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Universe event proof must be one exact mapping"
        )
    expected = _sha256(
        proof.get("universe_event_proof_sha256"),
        "universe event proof",
    )
    try:
        return validate_universe_event_proof(
            proof,
            expected_universe_event_proof_sha256=expected,
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Universe event proof failed exact replay"
        ) from None


def _call_blinded_ollama(
    request_bytes: bytes,
    supplied_sentence_ids: tuple[str, ...],
    *,
    transport: Any,
) -> dict[str, Any]:
    from agent_benchmark.sec_filing_gemma_ollama import (
        CONNECT_TIMEOUT_SECONDS,
        OLLAMA_ENDPOINT,
        READ_TIMEOUT_SECONDS,
        _REQUEST_HEADERS,
        _extract_ollama_attempt_output_bytes,
        _read_bounded_response,
        _validate_extractor_output_bytes,
        _validate_http_envelope,
    )

    response: Any | None = None
    started = time.monotonic()
    try:
        response = transport.request(
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
        output_bytes, normal_completion = (
            _extract_ollama_attempt_output_bytes(response_bytes)
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Exact no-retry Ollama request failed closed"
        ) from None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Ollama call diagnostic clock changed"
        )
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    validated_output: dict[str, Any] | None = None
    canonical_output_hash: str | None = None
    status = "invalid"
    if normal_completion:
        try:
            validated_hash, canonical_output_hash = (
                _validate_extractor_output_bytes(
                    output_bytes,
                    supplied_sentence_ids=supplied_sentence_ids,
                )
            )
            candidate = _strict_json_mapping_bytes(
                output_bytes,
                location="validated extractor output",
            )
        except Exception:
            pass
        else:
            if validated_hash != output_hash:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Validated extractor output hash changed"
                )
            validated_output = candidate
            status = "valid"
    return {
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "extractor_output_sha256": output_hash,
        "extractor_output_canonical_sha256": canonical_output_hash,
        "normal_completion": normal_completion,
        "extraction_status": status,
        "validated_output": validated_output,
        "http_status": http["http_status"],
        "response_url": http["response_url"],
        "response_content_type": http["response_content_type"],
        "response_history_count": http["response_history_count"],
        "elapsed_seconds_hex": elapsed.hex(),
    }


def _validated_model_slice(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
        _model_slice_index,
    )

    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released model slice must be one exact mapping"
        )
    try:
        indexed = _model_slice_index(value)
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released model slice failed exact replay"
        ) from None
    if value.get("model_slice_sha256") != canonical_sha256(indexed):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released model slice hash changed"
        )
    return copy.deepcopy(value)


def _validated_stage_slice(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
        _stage_slice_index,
    )

    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released stage slice must be one exact mapping"
        )
    try:
        indexed = _stage_slice_index(value)
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released stage slice failed exact replay"
        ) from None
    if value.get("slice_sha256") != canonical_sha256(indexed):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released stage slice hash changed"
        )
    return copy.deepcopy(value)


def _semantic_batch_receipt(
    *,
    stage: str,
    model_slice_sha256: str,
    runtime_probe_payload_sha256: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row_hashes: list[str] = []
    receipt_hashes: list[str] = []
    latency_hashes: set[str] = set()
    for ordinal, row in enumerate(rows, start=1):
        if type(row) is not dict:
            raise SecGemmaOnlineRiskOverlayProductionError(
                f"Semantic extraction row {ordinal} is not exact"
            )
        row_hashes.append(
            _sha256(
                row.get("semantic_extraction_row_sha256"),
                f"semantic extraction row {ordinal}",
            )
        )
        receipt_hashes.append(
            _sha256(
                row.get("semantic_event_receipt_sha256"),
                f"semantic event receipt {ordinal}",
            )
        )
        latency_hashes.add(
            _sha256(
                row.get("latency_preflight_receipt_sha256"),
                f"latency preflight receipt {ordinal}",
            )
        )
    if len(latency_hashes) != 1:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Semantic rows disagree on latency preflight evidence"
        )
    body = {
        "schema_version": SEMANTIC_BATCH_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "model_slice_sha256": _sha256(
            model_slice_sha256, "model slice"
        ),
        "runtime_probe_payload_sha256": _sha256(
            runtime_probe_payload_sha256, "runtime probe payload"
        ),
        "semantic_extraction_row_sha256s": row_hashes,
        "semantic_event_receipt_sha256s": receipt_hashes,
        "latency_preflight_receipt_sha256": next(
            iter(latency_hashes)
        ),
        "model_call_count": len(row_hashes),
        "retry_count": 0,
        "fallback_count": 0,
        "model_pull_count": 0,
        "paid_api_call_count": 0,
    }
    return {
        **body,
        "semantic_batch_receipt_sha256": canonical_sha256(body),
    }


def _nonnegative_elapsed_hex(value: Any, location: str) -> float:
    if type(value) is not str:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be canonical float.hex text"
        )
    try:
        elapsed = float.fromhex(value)
    except ValueError:
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be canonical float.hex text"
        ) from None
    if (
        not math.isfinite(elapsed)
        or elapsed < 0.0
        or elapsed.hex() != value
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            f"{location} must be one nonnegative canonical float"
        )
    return elapsed


def _gemma_phase_payload(
    model_slice: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
    *,
    request_call: Callable[[bytes, tuple[str, ...]], Mapping[str, Any]]
    | None = None,
    universe_validator: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _validate_universe_proof,
    slice_validator: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _validated_model_slice,
) -> dict[str, Any]:
    fixed = dict(slice_validator(model_slice))
    stage = fixed.get("stage")
    if stage not in {"development", "confirmation", "final"}:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released model slice stage changed"
        )
    requests_batch = fixed.get("model_requests")
    proofs = fixed.get("universe_event_proofs")
    if (
        type(requests_batch) is not list
        or type(proofs) is not list
        or not requests_batch
        or len(requests_batch) != len(proofs)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released model slice is empty or incomplete"
        )
    runtime_hash = _runtime_probe_payload_sha256(runtime_payload)
    prepared: list[dict[str, Any]] = []
    seen_accessions: set[str] = set()
    for canonical_ordinal, (request, raw_proof) in enumerate(
        zip(requests_batch, proofs, strict=True),
        start=1,
    ):
        proof = dict(universe_validator(raw_proof))
        request_bytes, sentence_ids, _payload = (
            _validate_blinded_model_request(request, proof)
        )
        accession = request["accession_number"]
        request_hash = request["request_sha256"]
        current = proof["current_record"]
        acceptance = current.get("acceptance_datetime")
        if (
            type(accession) is not str
            or type(acceptance) is not str
            or accession in seen_accessions
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Gemma batch contains a duplicate or unorderable event"
            )
        seen_accessions.add(accession)
        prepared.append(
            {
                "canonical_ordinal": canonical_ordinal,
                "request": request,
                "proof": proof,
                "request_bytes": request_bytes,
                "sentence_ids": sentence_ids,
                "request_byte_count": len(request_bytes),
                "acceptance_datetime": acceptance,
            }
        )
    chronological = sorted(
        prepared,
        key=lambda item: (
            item["request"]["availability_session"],
            item["acceptance_datetime"],
            item["request"]["accession_number"],
        ),
    )
    preflight: list[dict[str, Any]] = []
    if stage == "development":
        if len(prepared) < 5:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Development latency preflight requires five unique calls"
            )
        preflight = sorted(
            prepared,
            key=lambda item: (
                -item["request_byte_count"],
                item["request"]["accession_number"],
            ),
        )[:5]
        selected = {
            item["request"]["accession_number"] for item in preflight
        }
        execution_order = preflight + [
            item
            for item in chronological
            if item["request"]["accession_number"] not in selected
        ]
    else:
        execution_order = chronological

    owned_transport: Any | None = None
    if request_call is None:
        from agent_benchmark.sec_filing_gemma_ollama import (
            build_hardened_loopback_session,
        )

        owned_transport = build_hardened_loopback_session()
        if type(owned_transport) is not requests.Session:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Gemma batch returned a foreign production transport"
            )

        def exact_call(
            request_bytes: bytes,
            sentence_ids: tuple[str, ...],
        ) -> Mapping[str, Any]:
            return _call_blinded_ollama(
                request_bytes,
                sentence_ids,
                transport=owned_transport,
            )

        active_call = exact_call
    else:
        active_call = request_call
    calls_by_accession: dict[str, dict[str, Any]] = {}
    execution_evidence: list[dict[str, Any]] = []
    projection: float | None = None
    try:
        for execution_ordinal, item in enumerate(
            execution_order,
            start=1,
        ):
            request = item["request"]
            call = dict(
                active_call(
                    item["request_bytes"],
                    item["sentence_ids"],
                )
            )
            if set(call) != {
                "response_sha256",
                "extractor_output_sha256",
                "extractor_output_canonical_sha256",
                "normal_completion",
                "extraction_status",
                "validated_output",
                "http_status",
                "response_url",
                "response_content_type",
                "response_history_count",
                "elapsed_seconds_hex",
            }:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Ollama event receipt fields changed"
                )
            elapsed = _nonnegative_elapsed_hex(
                call["elapsed_seconds_hex"],
                f"Ollama event {execution_ordinal} elapsed",
            )
            if (
                call["extraction_status"] not in {"valid", "invalid"}
                or type(call["normal_completion"]) is not bool
                or call["http_status"] != 200
                or call["response_history_count"] != 0
                or call["response_url"] != _OLLAMA_CHAT_ENDPOINT
                or call["response_content_type"] != "application/json"
                or (
                    call["extraction_status"] == "valid"
                    and call["normal_completion"] is not True
                )
            ):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Ollama event result is not exact"
                )
            for field in (
                "response_sha256",
                "extractor_output_sha256",
            ):
                _sha256(
                    call[field],
                    f"Ollama event {execution_ordinal} {field}",
                )
            canonical_output = call[
                "extractor_output_canonical_sha256"
            ]
            validated_output = call["validated_output"]
            if call["extraction_status"] == "valid":
                _sha256(
                    canonical_output,
                    f"Ollama event {execution_ordinal} canonical output",
                )
                if type(validated_output) is not dict:
                    raise SecGemmaOnlineRiskOverlayProductionError(
                        "Valid Ollama event lacks its exact output"
                    )
            elif canonical_output is not None or validated_output is not None:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Invalid Ollama event exposes semantic output"
                )
            accession = request["accession_number"]
            calls_by_accession[accession] = {
                **call,
                "execution_ordinal": execution_ordinal,
            }
            execution_evidence.append(
                {
                    "execution_ordinal": execution_ordinal,
                    "accession_number": accession,
                    "request_sha256": request["request_sha256"],
                    "request_byte_count": item["request_byte_count"],
                    "elapsed_seconds_hex": elapsed.hex(),
                }
            )
            if stage == "development" and execution_ordinal == 5:
                first_five = [
                    _nonnegative_elapsed_hex(
                        evidence["elapsed_seconds_hex"],
                        "development latency preflight elapsed",
                    )
                    for evidence in execution_evidence
                ]
                remaining_count = len(execution_order) - 5
                projection = sum(first_five) + (
                    remaining_count * max(first_five)
                )
                if (
                    not math.isfinite(projection)
                    or projection > float(MAX_MODEL_SECONDS)
                ):
                    raise SecGemmaOnlineRiskOverlayProductionError(
                        "Development latency projection exceeds the frozen Gemma budget"
                    )
    finally:
        if owned_transport is not None:
            try:
                owned_transport.close()
            except Exception:
                pass
    if len(calls_by_accession) != len(prepared):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Gemma batch ended without one unique call per event"
        )
    latency_body = {
        "schema_version": LATENCY_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "selection": (
            "five_largest_request_bytes_desc_accession_asc"
            if stage == "development"
            else "not_applicable"
        ),
        "remaining_order": (
            "availability_acceptance_accession_ascending"
        ),
        "preflight_accession_numbers": [
            item["request"]["accession_number"] for item in preflight
        ],
        "execution_evidence": execution_evidence,
        "execution_order_sha256": canonical_sha256(
            execution_evidence
        ),
        "remaining_call_count": (
            len(prepared) - 5 if stage == "development" else 0
        ),
        "projected_total_seconds_hex": (
            None if projection is None else projection.hex()
        ),
        "gemma_seconds_cap": MAX_MODEL_SECONDS,
        "projection_passed": (
            True if stage == "development" else None
        ),
        "calls_are_unique_and_not_duplicated": True,
        "model_call_count": len(prepared),
    }
    latency_receipt = {
        **latency_body,
        "latency_preflight_receipt_sha256": canonical_sha256(
            latency_body
        ),
    }
    rows: list[dict[str, Any]] = []
    for item in prepared:
        ordinal = item["canonical_ordinal"]
        request = item["request"]
        proof = item["proof"]
        call = calls_by_accession[request["accession_number"]]
        canonical_output = call[
            "extractor_output_canonical_sha256"
        ]
        validated_output = call["validated_output"]
        proof_hash = _sha256(
            proof["universe_event_proof_sha256"],
            "universe event proof",
        )
        receipt_body = {
            "schema_version": SEMANTIC_EVENT_RECEIPT_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "stage": stage,
            "canonical_ordinal": ordinal,
            "execution_ordinal": call["execution_ordinal"],
            "accession_number": request["accession_number"],
            "request_sha256": request["request_sha256"],
            "request_byte_count": item["request_byte_count"],
            "universe_event_proof_sha256": proof_hash,
            "runtime_probe_payload_sha256": runtime_hash,
            "response_sha256": call["response_sha256"],
            "extractor_output_sha256": call[
                "extractor_output_sha256"
            ],
            "extractor_output_canonical_sha256": canonical_output,
            "normal_completion": call["normal_completion"],
            "extraction_status": call["extraction_status"],
            "http_status": call["http_status"],
            "response_url": call["response_url"],
            "response_content_type": call[
                "response_content_type"
            ],
            "response_history_count": call[
                "response_history_count"
            ],
            "elapsed_seconds_hex": call["elapsed_seconds_hex"],
            "retry_count": 0,
            "fallback_count": 0,
        }
        receipt = {
            **receipt_body,
            "semantic_event_receipt_sha256": canonical_sha256(
                receipt_body
            ),
        }
        row_body = {
            "schema_version": SEMANTIC_EXTRACTION_ROW_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "stage": stage,
            "ordinal": ordinal,
            "accession_number": request["accession_number"],
            "form": request["form"],
            "decision_session": request["availability_session"],
            "preprocessed_event_sha256": request[
                "preprocessed_event_sha256"
            ],
            "request_sha256": request["request_sha256"],
            "model_slice_sha256": fixed["model_slice_sha256"],
            "universe_event_proof_sha256": proof_hash,
            "extraction_status": call["extraction_status"],
            "extraction_authenticated": True,
            "document_quality": (
                None
                if validated_output is None
                else validated_output["document_quality"]
            ),
            "validated_output": copy.deepcopy(validated_output),
            "semantic_event_receipt": receipt,
            "semantic_event_receipt_sha256": receipt[
                "semantic_event_receipt_sha256"
            ],
            "latency_preflight_receipt": latency_receipt,
            "latency_preflight_receipt_sha256": latency_receipt[
                "latency_preflight_receipt_sha256"
            ],
        }
        rows.append(
            {
                **row_body,
                "semantic_extraction_row_sha256": canonical_sha256(
                    row_body
                ),
            }
        )
    batch = _semantic_batch_receipt(
        stage=stage,
        model_slice_sha256=fixed["model_slice_sha256"],
        runtime_probe_payload_sha256=runtime_hash,
        rows=rows,
    )
    return {
        "semantic_extraction_rows": rows,
        "semantic_extraction_row_sha256s": [
            row["semantic_extraction_row_sha256"] for row in rows
        ],
        "semantic_batch_receipt_sha256": batch[
            "semantic_batch_receipt_sha256"
        ],
    }


def _deterministic_phase_payload(
    stage_slice: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
    gemma_payload: Mapping[str, Any],
    *,
    universe_validator: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _validate_universe_proof,
    slice_validator: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _validated_stage_slice,
    feature_minter: Callable[..., Mapping[str, Any]] = (
        _mint_feature_row_from_source_bound_components
    ),
) -> dict[str, Any]:
    fixed = dict(slice_validator(stage_slice))
    stage = fixed.get("stage")
    if stage not in {"development", "confirmation", "final"}:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released stage slice stage changed"
        )
    if fixed.get("last_value_session") != {
        "development": "2018-12-31",
        "confirmation": "2023-12-29",
        "final": "2026-07-09",
    }[stage]:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Released stage slice cutoff changed"
        )
    if type(gemma_payload) is not dict or set(gemma_payload) != {
        "semantic_extraction_rows",
        "semantic_extraction_row_sha256s",
        "semantic_batch_receipt_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Gemma phase payload fields changed"
        )
    rows = copy.deepcopy(gemma_payload["semantic_extraction_rows"])
    row_hashes = gemma_payload["semantic_extraction_row_sha256s"]
    requests_batch = fixed.get("model_requests")
    proofs = fixed.get("universe_event_proofs")
    if (
        type(rows) is not list
        or type(row_hashes) is not list
        or type(requests_batch) is not list
        or type(proofs) is not list
        or not rows
        or not (
            len(rows)
            == len(row_hashes)
            == len(requests_batch)
            == len(proofs)
        )
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Gemma and deterministic stage batches differ"
        )
    runtime_hash = _runtime_probe_payload_sha256(runtime_payload)
    model_slice_hash = rows[0].get("model_slice_sha256")
    _sha256(model_slice_hash, "semantic model slice")
    rebuilt_batch = _semantic_batch_receipt(
        stage=stage,
        model_slice_sha256=model_slice_hash,
        runtime_probe_payload_sha256=runtime_hash,
        rows=rows,
    )
    if (
        rebuilt_batch["semantic_batch_receipt_sha256"]
        != _sha256(
            gemma_payload["semantic_batch_receipt_sha256"],
            "semantic batch receipt",
        )
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Semantic batch receipt changed before deterministic features"
        )
    market_material = _stage_slice_market_material(fixed)
    feature_rows: list[dict[str, Any]] = []
    proof_hashes: list[str] = []
    request_hashes: list[str] = []
    for ordinal, (request, raw_proof, row, row_hash) in enumerate(
        zip(
            requests_batch,
            proofs,
            rows,
            row_hashes,
            strict=True,
        ),
        start=1,
    ):
        proof = dict(universe_validator(raw_proof))
        _validate_blinded_model_request(request, proof)
        if (
            type(row) is not dict
            or row.get("semantic_extraction_row_sha256")
            != _sha256(row_hash, f"semantic row {ordinal}")
            or canonical_sha256(
                {
                    key: value
                    for key, value in row.items()
                    if key != "semantic_extraction_row_sha256"
                }
            )
            != row_hash
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Semantic extraction row changed before feature minting"
            )
        receipt = row.get("semantic_event_receipt")
        receipt_hash = row.get("semantic_event_receipt_sha256")
        latency = row.get("latency_preflight_receipt")
        latency_hash = row.get(
            "latency_preflight_receipt_sha256"
        )
        if (
            type(receipt) is not dict
            or type(latency) is not dict
            or receipt.get("semantic_event_receipt_sha256")
            != receipt_hash
            or canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "semantic_event_receipt_sha256"
                }
            )
            != receipt_hash
            or latency.get("latency_preflight_receipt_sha256")
            != latency_hash
            or canonical_sha256(
                {
                    key: value
                    for key, value in latency.items()
                    if key != "latency_preflight_receipt_sha256"
                }
            )
            != latency_hash
            or row.get("stage") != stage
            or row.get("ordinal") != ordinal
            or receipt.get("canonical_ordinal") != ordinal
            or type(receipt.get("execution_ordinal")) is not int
            or receipt["execution_ordinal"] < 1
            or row.get("model_slice_sha256") != model_slice_hash
            or row.get("request_sha256")
            != request.get("request_sha256")
            or row.get("universe_event_proof_sha256")
            != proof.get("universe_event_proof_sha256")
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Semantic event receipt lost its source binding"
            )
        current = proof["current_record"]
        lookback = _decision_market_lookback_rows(
            market_material,
            current["availability_session"],
        )
        bindings = {
            "stage_slice_sha256": fixed["slice_sha256"],
            "model_slice_sha256": model_slice_hash,
            "request_sha256": request["request_sha256"],
            "preprocessed_event_sha256": request[
                "preprocessed_event_sha256"
            ],
            "universe_event_proof_sha256": proof[
                "universe_event_proof_sha256"
            ],
            "current_record_sha256": proof["current_record_sha256"],
            "current_filing_sha256": proof["current_filing_sha256"],
            "prior_same_form_filing_sha256": proof[
                "prior_same_form_filing_sha256"
            ],
            "semantic_extraction_row_sha256": row_hash,
            "semantic_event_receipt_sha256": receipt_hash,
            "latency_preflight_receipt_sha256": latency_hash,
            "semantic_batch_receipt_sha256": rebuilt_batch[
                "semantic_batch_receipt_sha256"
            ],
            "runtime_probe_payload_sha256": runtime_hash,
            "market_source_commitments_sha256": market_material[
                "source_commitments"
            ]["source_commitments_sha256"],
        }
        minted = dict(
            feature_minter(
                accession_number=current["accession_number"],
                form=current["form"],
                decision_session=current["availability_session"],
                acceptance_datetime=current["acceptance_datetime"],
                artifact_stage=current["artifact_stage"],
                market_lookback_rows=lookback,
                extraction_status=row["extraction_status"],
                extraction_authenticated=row[
                    "extraction_authenticated"
                ],
                document_quality=row["document_quality"],
                validated_output=row["validated_output"],
                has_prior_same_form=(
                    proof["prior_same_form_record"] is not None
                ),
                upstream_bindings=bindings,
                market_unavailable_authenticated=True,
            )
        )
        feature_rows.append(minted)
        proof_hashes.append(proof["universe_event_proof_sha256"])
        request_hashes.append(request["request_sha256"])
    commitments_body = {
        **market_material["source_commitments"],
        "stage": stage,
        "stage_slice_sha256": fixed["slice_sha256"],
        "model_slice_sha256": model_slice_hash,
        "runtime_probe_payload_sha256": runtime_hash,
        "semantic_batch_receipt_sha256": rebuilt_batch[
            "semantic_batch_receipt_sha256"
        ],
        "model_request_sha256s": request_hashes,
        "universe_event_proof_sha256s": proof_hashes,
        "feature_row_sha256s": [
            row["feature_row_sha256"] for row in feature_rows
        ],
    }
    source_commitments = {
        **commitments_body,
        "production_source_commitments_sha256": canonical_sha256(
            commitments_body
        ),
    }
    market_rows = market_material["market_rows"]
    baseline_signals = market_material["baseline_signals"]
    return {
        "market_rows": market_rows,
        "market_rows_sha256": canonical_sha256(market_rows),
        "baseline_signals": baseline_signals,
        "baseline_signals_sha256": canonical_sha256(
            baseline_signals
        ),
        "feature_rows": feature_rows,
        "feature_row_sha256s": [
            row["feature_row_sha256"] for row in feature_rows
        ],
        "semantic_batch_receipt_sha256": rebuilt_batch[
            "semantic_batch_receipt_sha256"
        ],
        "source_commitments": source_commitments,
    }


def _phase_worker_dispatch(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if type(request) is not dict or set(request) != {
        "command",
        "phase",
        "prior_phase_outputs",
        "released_slice",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker request fields changed"
        )
    command = request["command"]
    phase = request["phase"]
    prior = request["prior_phase_outputs"]
    if type(prior) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker prior outputs changed"
        )
    counters = dict(_ZERO_COUNTERS)
    if phase == "runtime_identity":
        if request["released_slice"] is not None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Runtime identity worker received forbidden stage data"
            )
        payload: Mapping[str, Any] = _runtime_identity_payload()
    elif phase == "gemma" and command in _SCORING_COMMANDS:
        try:
            runtime_payload = prior["runtime_identity"]["payload"]
        except (KeyError, TypeError):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Gemma worker lacks runtime identity"
            ) from None
        payload = _gemma_phase_payload(
            request["released_slice"],
            runtime_payload,
        )
        counters["model_call_count"] = len(
            payload["semantic_extraction_rows"]
        )
    elif phase == "deterministic" and command in _SCORING_COMMANDS:
        try:
            runtime_payload = prior["runtime_identity"]["payload"]
            gemma_payload = prior["gemma"]["payload"]
        except (KeyError, TypeError):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Deterministic worker lacks runtime or Gemma evidence"
            ) from None
        payload = _deterministic_phase_payload(
            request["released_slice"],
            runtime_payload,
            gemma_payload,
        )
    else:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker phase is not preregistered"
        )
    return {"counters": counters, "payload": dict(payload)}


def _deterministic_evaluation_worker_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if type(request) is not dict or set(request) != {
        "stage",
        "input_bundle",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Deterministic evaluation worker request fields changed"
        )
    stage = request["stage"]
    input_bundle = request["input_bundle"]
    if stage not in {"development", "confirmation", "final"}:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Deterministic evaluation worker stage changed"
        )
    from agent_benchmark.sec_gemma_online_risk_overlay_runner import (
        DefaultDeterministicStageEvaluator,
        _validate_evaluation_envelope,
    )

    evaluator = DefaultDeterministicStageEvaluator()
    raw = evaluator.evaluate(
        stage=stage,
        input_bundle=input_bundle,
    )
    validated = evaluator.validate(
        raw,
        stage=stage,
        input_bundle=input_bundle,
    )
    return _validate_evaluation_envelope(
        validated,
        stage=stage,
        input_bundle=input_bundle,
    )

def _worker_handle_material(handle: Any) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        VaultHandle,
    )

    if type(handle) is not VaultHandle:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker received a foreign vault handle"
        )
    return {
        "vault_id": handle._vault_id,
        "entry_id": handle._entry_id,
        "stage": handle.stage,
        "attempt_id": handle.attempt_id,
        "generation": handle._generation,
        "bundle_sha256": handle.bundle_sha256,
        "manifest_sha256": handle.manifest_sha256,
        "private_index_sha256": handle.private_index_sha256,
        "payload_sha256": handle._payload_sha256,
        "production_authority": handle.production_authority,
        "seal_sha256": handle.seal_sha256,
    }


def _worker_execution_material(execution: Any) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
        AcquisitionExecutionResult,
    )

    if (
        type(execution) is not AcquisitionExecutionResult
        or execution.production_authority is not True
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker received a foreign opaque execution"
        )
    summary = execution.public_summary()
    accounting = execution.request_accounting()
    return {
        "handle": _worker_handle_material(execution.vault_handle),
        "verified_report": execution.terminal_sealing_material(),
        "public_summary": summary,
        "request_accounting": accounting,
        "model_slice_sha256": summary["model_slice_sha256"],
        "stage_slice_sha256": summary["stage_slice_sha256"],
    }


def _worker_vault_material(vault: Any) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        ProductionAcquisitionVault,
    )

    if (
        type(vault) is not ProductionAcquisitionVault
        or vault.production_authority is not True
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker requires the exact production vault"
        )
    return {
        "database_path": str(vault._database_path),
        "bound_store_instance_id": vault._bound_store_instance_id,
        "vault_id": vault.vault_id,
    }


def _worker_authority_material(
    authority: VerifiedProductionAuthority,
) -> dict[str, Any]:
    if not is_verified_production_authority(authority):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker requires exact production authority"
        )
    return {
        "payload": copy.deepcopy(authority._payload),
        "repo_root": str(authority.repo_root),
    }


def _worker_capability_material(
    capability: Any,
    *,
    store: Any,
) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        EffectCapability,
        SecGemmaOnlineRiskOverlayStore,
    )

    if (
        type(capability) is not EffectCapability
        or type(store) is not SecGemmaOnlineRiskOverlayStore
        or capability._store_instance_id
        != getattr(store, "_store_instance_id", None)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker requires one exact effect capability"
        )
    receipt = capability.consumption_receipt
    return {
        "attempt_id": capability.attempt_id,
        "allowed_effects": list(capability.allowed_effects),
        "receipt": {
            "table": receipt.table,
            "identity": receipt.identity,
            "attempt_id": receipt.attempt_id,
            "payload_sha256": receipt.payload_sha256,
            "journal_sequence": receipt.journal_sequence,
            "journal_entry_sha256": receipt.journal_entry_sha256,
        },
        "store_instance_id": capability._store_instance_id,
        "store_nonce": capability._store_nonce,
        "transition_sha256": capability._transition_sha256,
        "store_database_path": str(store._path),
        "implementation_manifest": copy.deepcopy(
            store._implementation_manifest
        ),
    }


def _open_worker_vault(material: Mapping[str, Any]) -> Any:
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        ProductionAcquisitionVault,
        _VAULT_CONSTRUCTOR_SENTINEL,
    )

    if type(material) is not dict or set(material) != {
        "database_path",
        "bound_store_instance_id",
        "vault_id",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker vault material is invalid"
        )
    vault = ProductionAcquisitionVault(
        database_path=Path(material["database_path"]),
        production_authority=True,
        bound_store_instance_id=material["bound_store_instance_id"],
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )
    if vault.vault_id != material["vault_id"]:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker vault identity changed"
        )
    return vault


def _worker_capability_and_store(
    material: Mapping[str, Any],
) -> tuple[Any, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        EffectCapability,
        SecGemmaOnlineRiskOverlayStore,
        StoreRecordReceipt,
        _CAPABILITY_SENTINEL,
    )

    if type(material) is not dict or set(material) != {
        "attempt_id",
        "allowed_effects",
        "receipt",
        "store_instance_id",
        "store_nonce",
        "transition_sha256",
        "store_database_path",
        "implementation_manifest",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker capability material is invalid"
        )
    receipt_material = material["receipt"]
    if type(receipt_material) is not dict or set(receipt_material) != {
        "table",
        "identity",
        "attempt_id",
        "payload_sha256",
        "journal_sequence",
        "journal_entry_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker receipt material is invalid"
        )
    allowed = material["allowed_effects"]
    if (
        type(material["attempt_id"]) is not str
        or type(allowed) is not list
        or not allowed
        or not all(type(effect) is str for effect in allowed)
        or len(set(allowed)) != len(allowed)
        or not {
            "official_sec_network",
            "market_network",
        }.issubset(allowed)
        or receipt_material["attempt_id"] != material["attempt_id"]
        or type(receipt_material["table"]) is not str
        or type(receipt_material["identity"]) is not str
        or type(receipt_material["journal_sequence"]) is not int
        or receipt_material["journal_sequence"] < 1
        or type(receipt_material["payload_sha256"]) is not str
        or _SHA256_RE.fullmatch(
            receipt_material["payload_sha256"]
        )
        is None
        or type(receipt_material["journal_entry_sha256"]) is not str
        or _SHA256_RE.fullmatch(
            receipt_material["journal_entry_sha256"]
        )
        is None
        or type(material["store_instance_id"]) is not str
        or _SHA256_RE.fullmatch(material["store_instance_id"]) is None
        or type(material["store_nonce"]) is not str
        or _SHA256_RE.fullmatch(material["store_nonce"]) is None
        or type(material["transition_sha256"]) is not str
        or _SHA256_RE.fullmatch(material["transition_sha256"]) is None
        or type(material["store_database_path"]) is not str
        or not material["store_database_path"]
        or type(material["implementation_manifest"]) is not dict
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker effects are invalid"
        )
    capability = EffectCapability(
        attempt_id=material["attempt_id"],
        allowed_effects=tuple(allowed),
        receipt=StoreRecordReceipt(**receipt_material),
        store_instance_id=material["store_instance_id"],
        store_nonce=material["store_nonce"],
        transition_sha256=material["transition_sha256"],
        _sentinel=_CAPABILITY_SENTINEL,
    )
    store = object.__new__(SecGemmaOnlineRiskOverlayStore)
    store._store_instance_id = material["store_instance_id"]
    store._store_nonce = material["store_nonce"]
    store._implementation_manifest = copy.deepcopy(
        material["implementation_manifest"]
    )
    store._active_capabilities = {
        capability.attempt_id: capability,
    }
    store._poisoned = False
    database_path = Path(material["store_database_path"]).resolve()
    if not database_path.is_file():
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker consumed store commitment is absent"
        )
    try:
        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro",
            uri=True,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
    except sqlite3.Error:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker could not open the consumed store commitment"
        ) from None
    store._connection = connection
    try:
        for effect in ("official_sec_network", "market_network"):
            SecGemmaOnlineRiskOverlayStore.authorize_effect(
                store,
                capability,
                effect,
            )
    except Exception:
        connection.close()
        store._connection = None
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker capability is not the current consumed store commitment"
        ) from None
    return capability, store


def _worker_execution_from_material(
    material: Mapping[str, Any],
    *,
    vault: Any,
) -> Any:
    from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
        AcquisitionExecutionResult,
        VerifiedAcquisitionReport,
        _EXECUTION_RESULT_SENTINEL,
        _VERIFIED_REPORT_SENTINEL,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        VaultHandle,
        _HANDLE_SENTINEL,
    )

    if type(material) is not dict or set(material) != {
        "handle",
        "verified_report",
        "public_summary",
        "request_accounting",
        "model_slice_sha256",
        "stage_slice_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker execution material is invalid"
        )
    handle_material = material["handle"]
    if type(handle_material) is not dict or set(handle_material) != {
        "vault_id",
        "entry_id",
        "stage",
        "attempt_id",
        "generation",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "payload_sha256",
        "production_authority",
        "seal_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker handle material is invalid"
        )
    if handle_material["vault_id"] != vault.vault_id:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker crossed its durable vault"
        )
    handle = VaultHandle(
        **handle_material,
        _sentinel=_HANDLE_SENTINEL,
    )
    report = VerifiedAcquisitionReport(
        material["verified_report"],
        vault=vault,
        handle=handle,
        _sentinel=_VERIFIED_REPORT_SENTINEL,
    )
    return AcquisitionExecutionResult(
        vault=vault,
        handle=handle,
        verified_report=report,
        public_summary=material["public_summary"],
        request_accounting=material["request_accounting"],
        model_slice_sha256=material["model_slice_sha256"],
        stage_slice_sha256=material["stage_slice_sha256"],
        _sentinel=_EXECUTION_RESULT_SENTINEL,
    )


def _acquisition_worker_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
        STAGES,
        acquire_stage_to_quarantine,
        build_acquisition_plan,
        create_production_market_transport,
    )

    if type(request) is not dict or set(request) != {
        "command",
        "stage",
        "deadline_monotonic",
        "authority",
        "vault",
        "capability",
        "sec_user_agent",
        "predecessors",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker request is invalid"
        )
    command = request["command"]
    stage = request["stage"]
    if _STAGE_BY_COMMAND.get(command) != stage:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker command crossed its stage"
        )
    try:
        deadline = float(request["deadline_monotonic"])
        now = float(time.monotonic())
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker deadline clock failed"
        ) from None
    if (
        not math.isfinite(now)
        or not math.isfinite(deadline)
        or not 0.0 < deadline - now <= float(MAX_SEC_SECONDS)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker deadline is outside the frozen cap"
        )
    authority_material = request["authority"]
    if type(authority_material) is not dict or set(authority_material) != {
        "payload",
        "repo_root",
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker authority material is invalid"
        )
    authority = VerifiedProductionAuthority(
        payload=authority_material["payload"],
        repo_root=Path(authority_material["repo_root"]).resolve(),
        _sentinel=_AUTHORITY_SENTINEL,
    )
    vault_material = request["vault"]
    capability_material = request["capability"]
    predecessors_material = request["predecessors"]
    if (
        type(vault_material) is not dict
        or type(capability_material) is not dict
        or capability_material.get("store_instance_id")
        != vault_material.get("bound_store_instance_id")
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker store and vault authorities differ"
        )
    if (
        type(predecessors_material) is not list
        or len(predecessors_material) != STAGES.index(stage)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker predecessor chain is foreign or incomplete"
        )
    vault = _open_worker_vault(vault_material)
    capability, store = _worker_capability_and_store(
        capability_material
    )
    if (
        capability.attempt_id
        != build_acquisition_plan(stage)["attempt_id"]
        or getattr(vault, "_bound_store_instance_id", None)
        != getattr(store, "_store_instance_id", None)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Acquisition worker authorities crossed their attempt"
        )
    predecessors = tuple(
        _worker_execution_from_material(item, vault=vault)
        for item in predecessors_material
    )
    plan = build_acquisition_plan(stage)
    market_transport = create_production_market_transport(
        plan,
        deadline_monotonic=deadline,
    )
    user_agent = request["sec_user_agent"]
    sec_transport = create_production_sec_transport(
        authority,
        user_agent=user_agent,
        clock=time.monotonic,
        sleep=time.sleep,
        deadline_monotonic=deadline,
    )
    try:
        execution = acquire_stage_to_quarantine(
            plan=plan,
            store=store,
            capability=capability,
            vault=vault,
            sec_user_agent=user_agent,
            sec_transport=sec_transport,
            market_transport=market_transport,
            predecessor_executions=predecessors,
            allow_test_authorities=False,
            deadline_monotonic=deadline,
            clock=time.monotonic,
            sleeper=time.sleep,
        )
        return _worker_execution_material(execution)
    finally:
        session = getattr(sec_transport, "_session", None)
        if type(session) is requests.Session:
            try:
                session.close()
            except Exception:
                pass
        connection = getattr(store, "_connection", None)
        if type(connection) is sqlite3.Connection:
            try:
                connection.close()
            except Exception:
                pass
            store._connection = None


_STDIO_WORKER_MAX_REQUEST_BYTES: Final[int] = 64 * 1024 * 1024
_STDIO_WORKER_MAX_RESPONSE_BYTES: Final[int] = 1024 * 1024 * 1024
_PUBLICATION_RECOVERY_CLEANUP_RESERVE_SECONDS: Final[float] = 5.0
_FINAL_REGISTRY_CLEANUP_RESERVE_SECONDS: Final[float] = 5.0
_FINAL_REGISTRY_MAX_COMMAND_IO_BYTES: Final[int] = 16 * 1024 * 1024
_PUBLICATION_RECOVERY_HOST_ENVIRONMENT_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "SystemRoot",
            "WINDIR",
            "USERPROFILE",
            "LOCALAPPDATA",
            "APPDATA",
            "HOME",
            "TEMP",
            "TMP",
            "COMSPEC",
            "PATHEXT",
        }
    )
)
_STDIO_WORKER_FLAG_RE: Final[re.Pattern[str]] = re.compile(
    r"--[a-z][a-z0-9-]{1,62}-worker\Z"
)
_WORKER_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "--acquisition-worker",
        "--phase-worker",
        "--deterministic-evaluation-worker",
        "--publication-recovery-worker",
    }
)
_WORKER_HTTP_IMPORT_ROOTS: Final[tuple[str, ...]] = (
    "certifi",
    "charset_normalizer",
    "idna",
    "requests",
    "urllib3",
)
_WORKER_ALLOWED_EXTERNAL_IMPORT_ROOTS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            *_WORKER_HTTP_IMPORT_ROOTS,
            *NUMERICAL_TIME_IMPORT_ROOTS,
        }
    )
)
_WORKER_MODULE: Final[str] = (
    "agent_benchmark.sec_gemma_online_risk_overlay_production"
)
_WORKER_BOOTSTRAP: Final[str] = r"""
import json
import importlib.abc
import os
from pathlib import Path
import runpy
import sys

def fail():
    raise SystemExit(91)

if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.ignore_environment != 1
    or sys.flags.safe_path is not True
    or len(sys.argv) != 5
):
    fail()
try:
    root = Path(sys.argv[1])
    material = json.loads(sys.argv[2])
    module_name = sys.argv[3]
    worker_flag = sys.argv[4]
except BaseException:
    fail()
if (
    not root.is_absolute()
    or root.resolve(strict=True) != root
    or Path.cwd().resolve(strict=True) != root
    or module_name != "agent_benchmark.sec_gemma_online_risk_overlay_production"
    or worker_flag not in {
        "--acquisition-worker",
        "--phase-worker",
        "--deterministic-evaluation-worker",
        "--publication-recovery-worker",
    }
    or type(material) is not dict
    or set(material) != {
        "allowed_external_import_roots",
        "external_paths",
        "isolated_base_paths",
    }
    or type(material["allowed_external_import_roots"]) is not list
    or type(material["external_paths"]) is not list
    or type(material["isolated_base_paths"]) is not list
):
    fail()
base_paths = material["isolated_base_paths"]
external_paths = material["external_paths"]
allowed_external_import_roots = material[
    "allowed_external_import_roots"
]
if (
    not base_paths
    or any(type(value) is not str or not value for value in base_paths)
    or any(type(value) is not str or not value for value in external_paths)
    or allowed_external_import_roots != [
        "certifi",
        "charset_normalizer",
        "idna",
        "numpy",
        "requests",
        "tzdata",
        "urllib3",
    ]
):
    fail()
normalize = lambda value: os.path.normcase(os.path.abspath(value))
if (
    [normalize(value) for value in sys.path]
    != [normalize(value) for value in base_paths]
    or len({normalize(value) for value in external_paths})
    != len(external_paths)
):
    fail()
for value in external_paths:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or not path.is_dir()
    ):
        fail()
production_file = (
    root
    / "agent_benchmark"
    / "sec_gemma_online_risk_overlay_production.py"
)
if (
    not production_file.is_file()
    or production_file.resolve(strict=True) != production_file
):
    fail()
sys.path[:] = [*base_paths, str(root), *external_paths]

class FrozenWorkerImportGuard(importlib.abc.MetaPathFinder):
    marker = "aapl-sec-gemma-online-risk-overlay-v2-2-worker-import-guard-v1"

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        import_root = fullname.split(".", 1)[0]
        if (
            import_root in allowed_external_import_roots
            or import_root == "agent_benchmark"
            or import_root == "__future__"
            or import_root in sys.stdlib_module_names
        ):
            return None
        raise ModuleNotFoundError(
            "External import is outside the verified worker runtime closure",
            name=fullname,
        )

sys.meta_path.insert(0, FrozenWorkerImportGuard())
sys._sec_gemma_worker_external_import_roots = tuple(
    allowed_external_import_roots
)
sys.argv[:] = [str(production_file), worker_flag]
runpy.run_module(module_name, run_name="__main__", alter_sys=True)
"""


def _worker_isolated_base_paths() -> tuple[str, ...]:
    base = Path(sys.base_prefix).resolve(strict=True)
    version = f"python{sys.version_info.major}{sys.version_info.minor}"
    if sys.platform == "win32":
        candidates = (
            base / f"{version}.zip",
            base / "DLLs",
            base / "Lib",
            base,
        )
    else:
        library = base / "lib" / (
            f"python{sys.version_info.major}.{sys.version_info.minor}"
        )
        candidates = (
            base / f"{version}.zip",
            library,
            library / "lib-dynload",
        )
    return tuple(str(path.resolve(strict=False)) for path in candidates)


def _worker_external_paths() -> tuple[str, ...]:
    observed: set[Path] = set()
    for module_name in _WORKER_HTTP_IMPORT_ROOTS:
        module = sys.modules.get(module_name)
        raw_file = getattr(module, "__file__", None)
        if type(raw_file) is not str or not raw_file:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production worker dependency path is unavailable"
            )
        try:
            module_file = Path(raw_file).resolve(strict=True)
            package_root = module_file.parent.parent.resolve(strict=True)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production worker dependency path is unavailable"
            ) from exc
        if not package_root.is_dir():
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production worker dependency root is invalid"
            )
        observed.add(package_root)
    for distribution_name in sorted(NUMERICAL_TIME_IMPORT_ROOTS):
        try:
            distribution_root = Path(
                importlib.metadata.distribution(
                    distribution_name
                ).locate_file("")
            ).resolve(strict=True)
        except (importlib.metadata.PackageNotFoundError, OSError) as exc:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production worker numerical/time dependency path is "
                "unavailable"
            ) from exc
        if not distribution_root.is_dir():
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production worker numerical/time dependency root is invalid"
            )
        observed.add(distribution_root)
    return tuple(
        str(path) for path in sorted(observed, key=lambda item: str(item).casefold())
    )


def _sanitized_worker_environment() -> dict[str, str]:
    if sys.platform != "win32":
        return {}
    environment: dict[str, str] = {}
    for required in ("SYSTEMROOT", "WINDIR"):
        value = next(
            (
                candidate
                for key, candidate in os.environ.items()
                if key.casefold() == required.casefold()
            ),
            None,
        )
        if (
            type(value) is not str
            or not value
            or "\x00" in value
            or not Path(value).resolve(strict=True).is_dir()
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production worker environment cannot be sanitized"
            )
        environment[required] = value
    if Path(environment["SYSTEMROOT"]).resolve(
        strict=True
    ) != Path(environment["WINDIR"]).resolve(strict=True):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker environment roots disagree"
        )
    return environment


def _validate_publication_recovery_host_environment(
    value: Mapping[str, Any],
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(
        _PUBLICATION_RECOVERY_HOST_ENVIRONMENT_KEYS
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery host environment fields changed"
        )
    observed = dict(value)
    if any(
        type(item) is not str or not item or "\x00" in item
        for item in observed.values()
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery host environment is incomplete"
        )
    try:
        system_root = Path(observed["SystemRoot"]).resolve(strict=True)
        windir = Path(observed["WINDIR"]).resolve(strict=True)
        profile = Path(observed["USERPROFILE"]).resolve(strict=True)
        home = Path(observed["HOME"]).resolve(strict=True)
        local_appdata = Path(
            observed["LOCALAPPDATA"]
        ).resolve(strict=True)
        appdata = Path(observed["APPDATA"]).resolve(strict=True)
        temporary = Path(observed["TEMP"]).resolve(strict=True)
        temporary_alias = Path(observed["TMP"]).resolve(strict=True)
        comspec = Path(observed["COMSPEC"]).resolve(strict=True)
    except OSError:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery host environment paths are invalid"
        ) from None
    extensions = observed["PATHEXT"].split(";")
    if (
        not system_root.is_dir()
        or Path(observed["SystemRoot"]) != system_root
        or windir != system_root
        or Path(observed["WINDIR"]) != windir
        or not profile.is_dir()
        or Path(observed["USERPROFILE"]) != profile
        or home != profile
        or Path(observed["HOME"]) != home
        or not local_appdata.is_dir()
        or Path(observed["LOCALAPPDATA"]) != local_appdata
        or not appdata.is_dir()
        or Path(observed["APPDATA"]) != appdata
        or not temporary.is_dir()
        or Path(observed["TEMP"]) != temporary
        or temporary_alias != temporary
        or Path(observed["TMP"]) != temporary_alias
        or not comspec.is_file()
        or Path(observed["COMSPEC"]) != comspec
        or comspec.parent != system_root / "System32"
        or not extensions
        or any(
            re.fullmatch(r"\.[A-Za-z0-9]{1,8}", extension) is None
            for extension in extensions
        )
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery host environment identity is invalid"
        )
    return observed


def _publication_recovery_host_environment() -> dict[str, str]:
    if os.name != "nt":
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery host environment requires Windows"
        )
    ambient = {
        key.casefold(): value for key, value in os.environ.items()
    }

    def required(name: str) -> str:
        value = ambient.get(name.casefold())
        if type(value) is not str or not value:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery host environment is unavailable"
            )
        return value

    profile = required("USERPROFILE")
    return _validate_publication_recovery_host_environment(
        {
            "SystemRoot": required("SYSTEMROOT"),
            "WINDIR": required("WINDIR"),
            "USERPROFILE": profile,
            "LOCALAPPDATA": required("LOCALAPPDATA"),
            "APPDATA": required("APPDATA"),
            "HOME": ambient.get("home", profile),
            "TEMP": required("TEMP"),
            "TMP": required("TMP"),
            "COMSPEC": required("COMSPEC"),
            "PATHEXT": required("PATHEXT"),
        }
    )


def _install_publication_recovery_host_environment(
    value: Mapping[str, Any],
) -> None:
    observed = _validate_publication_recovery_host_environment(value)
    expected_sanitized = _sanitized_worker_environment()
    current = {
        key.casefold(): item for key, item in os.environ.items()
    }
    if current != {
        key.casefold(): item
        for key, item in expected_sanitized.items()
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker ambient environment changed"
        )
    os.environ.clear()
    os.environ.update(observed)
    installed = {
        key.casefold(): item for key, item in os.environ.items()
    }
    if installed != {
        key.casefold(): item for key, item in observed.items()
    }:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery host environment could not be installed"
        )


def _worker_bootstrap_material() -> dict[str, list[str]]:
    return {
        "allowed_external_import_roots": list(
            _WORKER_ALLOWED_EXTERNAL_IMPORT_ROOTS
        ),
        "external_paths": list(_worker_external_paths()),
        "isolated_base_paths": list(_worker_isolated_base_paths()),
    }


def _verify_isolated_worker_bootstrap() -> None:
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.safe_path is not True
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker did not start in isolated no-site mode"
        )
    root = Path(__file__).resolve(strict=True).parents[1]
    if Path.cwd().resolve(strict=True) != root:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker current directory changed"
        )
    expected_path = [
        *_worker_isolated_base_paths(),
        str(root),
        *_worker_external_paths(),
    ]
    normalize = lambda value: os.path.normcase(os.path.abspath(value))
    if [normalize(value) for value in sys.path] != [
        normalize(value) for value in expected_path
    ]:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker sys.path bootstrap changed"
        )
    observed_import_roots = getattr(
        sys,
        "_sec_gemma_worker_external_import_roots",
        None,
    )
    import_guard = sys.meta_path[0] if sys.meta_path else None
    if (
        observed_import_roots
        != _WORKER_ALLOWED_EXTERNAL_IMPORT_ROOTS
        or getattr(import_guard, "marker", None)
        != (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "worker-import-guard-v1"
        )
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker external import guard changed"
        )
    expected_environment = _sanitized_worker_environment()
    observed_environment = {
        key.upper(): value for key, value in os.environ.items()
    }
    if observed_environment != expected_environment:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production worker inherited ambient environment state"
        )


def _terminate_stdio_worker(process: Any) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except BaseException:
                pass
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2.0)
            except BaseException:
                pass
    except BaseException:
        pass


def _run_stdio_worker_subprocess(
    request: Mapping[str, Any],
    *,
    worker_flag: str,
    deadline_monotonic: float,
    clock: Callable[[], float] = time.monotonic,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    try:
        now = float(clock())
        deadline = float(deadline_monotonic)
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker deadline clock failed"
        ) from None
    if (
        not math.isfinite(now)
        or not math.isfinite(deadline)
        or deadline <= now
        or type(worker_flag) is not str
        or _STDIO_WORKER_FLAG_RE.fullmatch(worker_flag) is None
        or worker_flag not in _WORKER_FLAGS
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker authority is invalid or expired"
        )
    try:
        encoded_request = canonical_json_bytes(
            copy.deepcopy(dict(request))
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker request is not canonical JSON"
        ) from None
    if len(encoded_request) > _STDIO_WORKER_MAX_REQUEST_BYTES:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker request exceeds its fixed byte cap"
        )
    creationflags = (
        int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if sys.platform == "win32"
        else 0
    )
    try:
        executable = Path(sys.executable).resolve(strict=True)
        repo_root = Path(__file__).resolve(strict=True).parents[1]
        bootstrap_material = canonical_json_bytes(
            _worker_bootstrap_material()
        ).decode("ascii", errors="strict")
        worker_environment = _sanitized_worker_environment()
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker bootstrap could not be sealed"
        ) from None
    command = (
        str(executable),
        "-I",
        "-S",
        "-c",
        _WORKER_BOOTSTRAP,
        str(repo_root),
        bootstrap_material,
        _WORKER_MODULE,
        worker_flag,
    )
    try:
        process = popen_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
            cwd=str(repo_root),
            env=worker_environment,
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker could not be started"
        ) from None
    try:
        remaining = deadline - float(clock())
        if remaining <= 0.0:
            _terminate_stdio_worker(process)
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production stdio worker exceeded its hard deadline"
            )
        try:
            stdout, _stderr = process.communicate(
                input=encoded_request,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            _terminate_stdio_worker(process)
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production stdio worker exceeded its hard deadline"
            ) from None
        except Exception:
            _terminate_stdio_worker(process)
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production stdio worker communication failed"
            ) from None
        if (
            float(clock()) >= deadline
            or process.returncode != 0
            or type(stdout) is not bytes
            or not stdout
            or len(stdout) > _STDIO_WORKER_MAX_RESPONSE_BYTES
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production stdio worker failed closed"
            )
        try:
            decoded = json.loads(stdout.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production stdio worker returned invalid evidence"
            ) from None
        if (
            type(decoded) is not dict
            or set(decoded) != {"payload", "status"}
            or decoded["status"] != "ok"
            or type(decoded["payload"]) is not dict
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production stdio worker failed closed"
            )
        return copy.deepcopy(decoded["payload"])
    finally:
        if getattr(process, "poll", lambda: 0)() is None:
            _terminate_stdio_worker(process)


def _run_phase_subprocess(
    request: Mapping[str, Any],
    *,
    deadline_monotonic: float,
    clock: Callable[[], float] = time.monotonic,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    result = _run_stdio_worker_subprocess(
        request,
        worker_flag="--phase-worker",
        deadline_monotonic=deadline_monotonic,
        clock=clock,
        popen_factory=popen_factory,
    )
    if set(result) != {"counters", "payload"}:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production phase worker result fields changed"
        )
    return result


def _run_deterministic_evaluation_subprocess(
    *,
    stage: str,
    input_bundle: Mapping[str, Any],
    deadline_monotonic: float,
    clock: Callable[[], float] = time.monotonic,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    if stage not in {"development", "confirmation", "final"}:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Deterministic evaluation stage changed"
        )
    return _run_stdio_worker_subprocess(
        {
            "stage": stage,
            "input_bundle": copy.deepcopy(dict(input_bundle)),
        },
        worker_flag="--deterministic-evaluation-worker",
        deadline_monotonic=deadline_monotonic,
        clock=clock,
        popen_factory=popen_factory,
    )


def _run_acquisition_subprocess(
    request: Mapping[str, Any],
    *,
    deadline_monotonic: float,
    clock: Callable[[], float] = time.monotonic,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    try:
        now = float(clock())
        deadline = float(deadline_monotonic)
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production acquisition worker deadline clock failed"
        ) from None
    if (
        not math.isfinite(now)
        or not math.isfinite(deadline)
        or not 0.0 < deadline - now <= float(MAX_SEC_SECONDS)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production acquisition worker deadline is outside the frozen cap"
        )
    return _run_stdio_worker_subprocess(
        request,
        worker_flag="--acquisition-worker",
        deadline_monotonic=deadline,
        clock=clock,
        popen_factory=popen_factory,
    )


def _read_stdio_worker_request(
    *,
    require_canonical: bool = False,
) -> dict[str, Any]:
    raw = sys.stdin.buffer.read(_STDIO_WORKER_MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > _STDIO_WORKER_MAX_REQUEST_BYTES:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker input is absent or oversized"
        )
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker input is invalid"
        ) from None
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker input must be one mapping"
        )
    if require_canonical and canonical_json_bytes(value) != raw:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker input is not canonical JSON"
        )
    return value


def _write_stdio_worker_result(payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(
        {"status": "ok", "payload": copy.deepcopy(dict(payload))}
    )
    if len(encoded) > _STDIO_WORKER_MAX_RESPONSE_BYTES:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production stdio worker output exceeds its fixed byte cap"
        )
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


class _PrivateSecUserAgent:
    """Hold the required contact identity without a public reveal surface."""

    __slots__ = ("__value", "_sha256")

    def __init__(self, value: str) -> None:
        try:
            audit = validate_sec_user_agent(value)
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Private SEC contact identity is invalid"
            ) from None
        self.__value = value
        self._sha256 = audit.sha256

    def _reveal_for_sec_transport(self) -> str:
        return self.__value

    def _reverify(self) -> None:
        try:
            audit = validate_sec_user_agent(self.__value)
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Private SEC contact identity is no longer valid"
            ) from None
        if audit.sha256 != self._sha256:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Private SEC contact identity changed"
            )

    def __repr__(self) -> str:
        return "_PrivateSecUserAgent(<redacted>)"


class _AcquisitionExecutionLedger:
    """Retain only exact opaque predecessor executions in stage order."""

    __slots__ = ("_executions",)

    def __init__(self) -> None:
        self._executions: dict[str, Any] = {}

    def predecessors(self, stage: str) -> tuple[Any, ...]:
        from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
            STAGES,
        )

        if stage not in STAGES:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Acquisition stage is not preregistered"
            )
        prior = STAGES[: STAGES.index(stage)]
        if set(self._executions) != set(prior):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Opaque acquisition predecessor chain is incomplete"
            )
        return tuple(self._executions[item] for item in prior)

    def record(self, stage: str, execution: Any) -> None:
        from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
            AcquisitionExecutionResult,
            STAGES,
        )

        if (
            stage not in STAGES
            or type(execution) is not AcquisitionExecutionResult
            or execution.production_authority is not True
            or stage in self._executions
            or set(self._executions) != set(
                STAGES[: STAGES.index(stage)]
            )
            or execution.public_summary().get("stage") != stage
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Opaque acquisition execution is foreign or out of order"
            )
        self._executions[stage] = execution

    def resolve_for_scoring(
        self,
        command: str,
        current: Any | None,
    ) -> Any:
        from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
            AcquisitionExecutionResult,
        )

        stage = _STAGE_BY_COMMAND.get(command)
        if command not in _SCORING_COMMANDS or stage is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Scoring acquisition command is not preregistered"
            )
        expected = self._executions.get(stage)
        if expected is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Scoring lacks its exact opaque acquisition"
            )
        if command == "development":
            if current is not None:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Development scoring received a foreign current acquisition"
                )
        elif (
            type(current) is not AcquisitionExecutionResult
            or current is not expected
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Scoring crossed its current opaque acquisition"
            )
        return expected

    def retained_stages(self) -> tuple[str, ...]:
        from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
            STAGES,
        )

        return tuple(
            stage for stage in STAGES if stage in self._executions
        )

    def assert_ready_for_command(
        self,
        command: str,
        *,
        store: Any,
    ) -> None:
        from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
            TERMINAL_PASS,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
            CONFIRMATION_ATTEMPT_ID,
            DEVELOPMENT_ACQUISITION_ID,
            FINAL_ATTEMPT_ID,
        )

        expected = {
            "development_acquisition": (),
            "development": ("development",),
            "confirmation": ("development",),
            "final": ("development", "confirmation"),
        }.get(command)
        if expected is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production readiness command is not preregistered"
            )
        retained = self.retained_stages()
        durable_passes: list[str] = []
        for stage, attempt_id in (
            ("development", DEVELOPMENT_ACQUISITION_ID),
            ("confirmation", CONFIRMATION_ATTEMPT_ID),
            ("final", FINAL_ATTEMPT_ID),
        ):
            try:
                history = store.attempt_history(attempt_id)
            except Exception:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Durable attempt history cannot prove acquisition readiness"
                ) from None
            if history and history[-1].get("status") == TERMINAL_PASS:
                durable_passes.append(stage)
        durable_required = tuple(
            stage
            for stage in ("development", "confirmation", "final")
            if stage in durable_passes
        )
        if retained != expected:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Opaque acquisition state is absent, restarted, or out of order"
            )
        if command == "development_acquisition":
            if durable_required:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Development acquisition already exists durably but cannot be replayed"
                )
        elif command == "development":
            if durable_required != ("development",):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Development scoring lacks its matching durable acquisition"
                )
        elif command == "confirmation":
            if durable_required != ("development",):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Confirmation readiness differs from its durable predecessor"
                )
        elif durable_required != ("development", "confirmation"):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Final readiness differs from its durable predecessors"
            )


def _rehydrate_production_acquisition_ledger(
    *,
    vault: Any,
    store: Any,
) -> _AcquisitionExecutionLedger:
    from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
        STAGES,
        _rehydrate_one_production_acquisition,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        SecGemmaOnlineRiskOverlayStoreError,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        SecGemmaOnlineRiskOverlayVaultError,
        _production_recovery_sealed_stages,
    )

    attempt_by_stage = {
        "development": DEVELOPMENT_ACQUISITION_ID,
        "confirmation": CONFIRMATION_ATTEMPT_ID,
        "final": FINAL_ATTEMPT_ID,
    }
    materials: list[dict[str, Any] | None] = []
    missing = False
    for stage in STAGES:
        attempt_id = attempt_by_stage[stage]
        try:
            material = store.terminal_acquisition_phase_evidence(
                attempt_id
            )
        except SecGemmaOnlineRiskOverlayStoreError as exc:
            if str(exc) != (
                "Acquisition recovery attempt did not terminal-pass"
            ):
                raise
            material = None
        if material is None:
            missing = True
        elif missing:
            raise SecGemmaOnlineRiskOverlayProductionError(
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
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production acquisition recovery vault index is corrupt"
        ) from None
    allowed_sealed = durable_stages
    if len(durable_stages) < len(STAGES):
        pending_stage = STAGES[len(durable_stages)]
        pending_attempt = attempt_by_stage[pending_stage]
        pending_proven = False
        try:
            snapshot = store.verify_chain()
            pending_proven = (
                type(snapshot) is dict
                and snapshot.get("attempt_states", {}).get(
                    pending_attempt
                )
                == "consumed"
            )
        except Exception:
            pending_proven = False
        if pending_proven:
            try:
                store.pending_acquisition_phase_evidence(
                    pending_attempt
                )
            except SecGemmaOnlineRiskOverlayStoreError:
                pass
            else:
                allowed_sealed = durable_stages + (pending_stage,)
    if sealed_stages not in {durable_stages, allowed_sealed}:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Durable acquisition evidence and sealed vault stages differ"
        )
    recovered: list[Any] = []
    for stage, material in zip(STAGES, materials, strict=True):
        if material is None:
            break
        recovered.append(
            _rehydrate_one_production_acquisition(
                vault=vault,
                store=store,
                stage=stage,
                public_material=material,
                predecessor_executions=tuple(recovered),
            )
        )
    ledger = _AcquisitionExecutionLedger()
    for execution in recovered:
        ledger.record(
            execution.public_summary()["stage"],
            execution,
        )
    return ledger


class _ProductionAcquisitionAdapter:
    __slots__ = (
        "_authority",
        "_clock",
        "_ledger",
        "_private_identity",
        "_repo_root",
        "_sentinel",
        "_sleeper",
        "_store",
        "_vault",
    )

    def __init__(
        self,
        *,
        authority: VerifiedProductionAuthority,
        repo_root: Path,
        store: Any,
        vault: Any,
        private_identity: _PrivateSecUserAgent,
        ledger: _AcquisitionExecutionLedger,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
        _sentinel: object,
    ) -> None:
        from agent_benchmark.sec_gemma_online_risk_overlay_store import (
            SecGemmaOnlineRiskOverlayStore,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
            ProductionAcquisitionVault,
        )

        if (
            _sentinel is not _ADAPTER_SENTINEL
            or not is_verified_production_authority(authority)
            or type(store) is not SecGemmaOnlineRiskOverlayStore
            or type(vault) is not ProductionAcquisitionVault
            or type(private_identity) is not _PrivateSecUserAgent
            or type(ledger) is not _AcquisitionExecutionLedger
            or clock is not time.monotonic
            or sleeper is not time.sleep
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production acquisition adapter requires exact authorities"
            )
        self._authority = authority
        self._repo_root = repo_root
        self._store = store
        self._vault = vault
        self._private_identity = private_identity
        self._ledger = ledger
        self._clock = clock
        self._sleeper = sleeper
        self._sentinel = _sentinel

    def acquire(
        self,
        *,
        command: str,
        store: Any,
        capability: Any,
        deadline_monotonic: float,
    ) -> Any:
        from agent_benchmark.sec_gemma_online_risk_overlay_store import (
            EffectCapability,
            SecGemmaOnlineRiskOverlayStore,
        )

        if (
            command not in {
                "development_acquisition",
                "confirmation",
                "final",
            }
            or type(store) is not SecGemmaOnlineRiskOverlayStore
            or store is not self._store
            or type(capability) is not EffectCapability
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production acquisition received foreign runner authorities"
            )
        stage = _STAGE_BY_COMMAND[command]
        try:
            now = float(self._clock())
            deadline = float(deadline_monotonic)
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production acquisition deadline clock failed"
            ) from None
        if (
            not math.isfinite(now)
            or not math.isfinite(deadline)
            or not 0.0 < deadline - now <= float(MAX_SEC_SECONDS)
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production acquisition deadline is outside the frozen cap"
            )
        predecessors = self._ledger.predecessors(stage)
        for effect in ("official_sec_network", "market_network"):
            try:
                store.authorize_effect(capability, effect)
            except Exception:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Production acquisition capability is no longer current"
                ) from None
        user_agent = self._private_identity._reveal_for_sec_transport()
        request = {
            "command": command,
            "stage": stage,
            "deadline_monotonic": deadline,
            "authority": _worker_authority_material(self._authority),
            "vault": _worker_vault_material(self._vault),
            "capability": _worker_capability_material(
                capability,
                store=store,
            ),
            "sec_user_agent": user_agent,
            "predecessors": [
                _worker_execution_material(execution)
                for execution in predecessors
            ],
        }
        material = _run_acquisition_subprocess(
            request,
            deadline_monotonic=deadline,
            clock=time.monotonic,
        )
        if float(self._clock()) >= deadline:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production acquisition worker crossed its parent deadline"
            )
        for effect in ("official_sec_network", "market_network"):
            try:
                store.authorize_effect(capability, effect)
            except Exception:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Production acquisition capability changed while its worker ran"
                ) from None
        execution = _worker_execution_from_material(
            material,
            vault=self._vault,
        )
        summary = execution.public_summary()
        if (
            summary.get("stage") != stage
            or summary.get("attempt_id") != capability.attempt_id
            or float(self._clock()) >= deadline
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production acquisition result crossed its attempt or deadline"
            )
        self._ledger.record(stage, execution)
        return execution

    def rehydrate_pending_acquisition_report(
        self,
        *,
        attempt_id: str,
    ) -> Any:
        """Reissue only an already sealed publication-pending report."""

        from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
            _rehydrate_one_production_acquisition,
            is_verified_acquisition_report,
        )

        stage_by_attempt = {
            DEVELOPMENT_ACQUISITION_ID: "development",
            CONFIRMATION_ATTEMPT_ID: "confirmation",
            FINAL_ATTEMPT_ID: "final",
        }
        stage = stage_by_attempt.get(attempt_id)
        if stage is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Pending acquisition report attempt is not preregistered"
            )
        try:
            public_material = (
                self._store.pending_acquisition_phase_evidence(
                    attempt_id
                )
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Pending acquisition report lacks exact durable evidence"
            ) from None
        existing = self._ledger._executions.get(stage)
        if existing is None:
            try:
                predecessors = self._ledger.predecessors(stage)
                execution = _rehydrate_one_production_acquisition(
                    vault=self._vault,
                    store=self._store,
                    stage=stage,
                    public_material=public_material,
                    predecessor_executions=predecessors,
                )
                self._ledger.record(stage, execution)
            except Exception:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Pending acquisition report could not be rehydrated"
                ) from None
        else:
            execution = existing
        report = execution.verified_report
        if (
            not is_verified_acquisition_report(report)
            or report.as_dict()
            != public_material["verified_acquisition_report"]
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Pending acquisition report differs from durable evidence"
            )
        return report

    def __repr__(self) -> str:
        return "_ProductionAcquisitionAdapter(<source-bound, identity-redacted>)"


class _ProductionPhaseExecutor:
    __slots__ = ("_clock", "_ledger", "_sentinel")

    def __init__(
        self,
        *,
        ledger: _AcquisitionExecutionLedger,
        clock: Callable[[], float],
        _sentinel: object,
    ) -> None:
        if (
            _sentinel is not _EXECUTOR_SENTINEL
            or type(ledger) is not _AcquisitionExecutionLedger
            or clock is not time.monotonic
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production phase executor requires exact authorities"
            )
        self._ledger = ledger
        self._clock = clock
        self._sentinel = _sentinel

    def execute(
        self,
        *,
        command: str,
        phase: str,
        permit: Any,
        deadline_monotonic: float,
        prior_phase_outputs: Mapping[str, Mapping[str, Any]],
        acquisition_execution: Any | None,
    ) -> Mapping[str, Any]:
        from agent_benchmark.sec_gemma_online_risk_overlay_runner import (
            LOCAL_PREFLIGHT,
            build_phase_output,
            is_phase_permit,
        )

        if (
            command not in {
                LOCAL_PREFLIGHT,
                "development",
                "confirmation",
                "final",
            }
            or phase not in {
                "runtime_identity",
                "gemma",
                "deterministic",
            }
            or type(prior_phase_outputs) is not dict
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production executor phase is not preregistered"
            )
        if command == LOCAL_PREFLIGHT:
            if phase != "runtime_identity" or permit is not None:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Local preflight can only probe runtime identity"
                )
        elif (
            not is_phase_permit(permit)
            or permit.command != command
            or permit.phase != phase
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production executor requires an exact consumed phase permit"
            )
        try:
            now = float(self._clock())
            deadline = float(deadline_monotonic)
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production executor deadline clock failed"
            ) from None
        if (
            not math.isfinite(now)
            or not math.isfinite(deadline)
            or deadline <= now
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production executor deadline has expired"
            )
        released: Mapping[str, Any] | None = None
        if phase in {"gemma", "deterministic"}:
            execution = self._ledger.resolve_for_scoring(
                command,
                acquisition_execution,
            )
            if phase == "gemma":
                released = permit.release_acquisition_model_slice(
                    execution
                )
            else:
                released = permit.release_acquisition_stage_slice(
                    execution
                )
            if type(released) is not dict:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Phase permit returned a foreign acquisition slice"
                )
        request = {
            "command": command,
            "phase": phase,
            "prior_phase_outputs": copy.deepcopy(
                dict(prior_phase_outputs)
            ),
            "released_slice": copy.deepcopy(released),
        }
        worker = _run_phase_subprocess(
            request,
            deadline_monotonic=deadline,
            clock=time.monotonic,
        )
        return build_phase_output(
            command=command,
            phase=phase,
            counters=worker["counters"],
            payload=worker["payload"],
        )

    def evaluate_deterministic(
        self,
        *,
        stage: str,
        input_bundle: Mapping[str, Any],
        deadline_monotonic: float,
    ) -> Mapping[str, Any]:
        if stage not in {"development", "confirmation", "final"}:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production deterministic evaluation stage changed"
            )
        try:
            now = float(self._clock())
            deadline = float(deadline_monotonic)
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production deterministic evaluation clock failed"
            ) from None
        if (
            not math.isfinite(now)
            or not math.isfinite(deadline)
            or deadline <= now
            or deadline - now > float(MAX_DETERMINISTIC_SECONDS)
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production deterministic evaluation deadline is outside "
                "the frozen cap"
            )
        return _run_deterministic_evaluation_subprocess(
            stage=stage,
            input_bundle=input_bundle,
            deadline_monotonic=deadline,
            clock=time.monotonic,
        )

    def __repr__(self) -> str:
        return "_ProductionPhaseExecutor(<source-bound, subprocess-isolated>)"


def _portable_publication_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _publication_object_name(kind: str, owner_nonce_sha256: str) -> str:
    if kind not in {"job", "mutex"}:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication kernel object kind changed"
        )
    nonce = _sha256(owner_nonce_sha256, "publication owner nonce")
    return (
        "Local\\AaplSecGemmaOnlineRiskOverlayV22"
        f"-{kind}-{nonce}"
    )


def _publication_object_name_sha256(name: str) -> str:
    return canonical_sha256({"windows_object_name": name})


def _publication_owner_process_identity() -> tuple[int, str]:
    if os.name != "nt":
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication containment requires Windows"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not kernel32.GetProcessTimes(
        kernel32.GetCurrentProcess(),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication owner process identity is unavailable"
        )
    value = (int(creation.dwHighDateTime) << 32) | int(
        creation.dwLowDateTime
    )
    return os.getpid(), hex(value)


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _JobObjectBasicProcessIdList(ctypes.Structure):
    _fields_ = [
        ("NumberOfAssignedProcesses", wintypes.DWORD),
        ("NumberOfProcessIdsInList", wintypes.DWORD),
        ("ProcessIdList", ctypes.c_size_t * 1),
    ]


class _StartupInfoW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _StartupInfoExW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _StartupInfoW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _WindowsAtomicPublicationProcess:
    """Small process facade for a child created atomically inside one Job."""

    __slots__ = (
        "_closed",
        "_kernel32",
        "_stderr_limit_bytes",
        "_stderr_file",
        "_stdin_file",
        "_stdout_limit_bytes",
        "_stdout_file",
        "_handle",
        "pid",
        "returncode",
    )

    def __init__(
        self,
        *,
        kernel32: Any,
        process_handle: int,
        process_id: int,
        stdin_file: Any,
        stdout_file: Any,
        stderr_file: Any,
        stdout_limit_bytes: int | None,
        stderr_limit_bytes: int | None,
    ) -> None:
        self._kernel32 = kernel32
        self._handle = process_handle
        self.pid = process_id
        self._stdin_file = stdin_file
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file
        self._stdout_limit_bytes = stdout_limit_bytes
        self._stderr_limit_bytes = stderr_limit_bytes
        self.returncode: int | None = None
        self._closed = False

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        wait = self._kernel32.WaitForSingleObject(
            wintypes.HANDLE(self._handle),
            0,
        )
        if wait == _WindowsPublicationKernel._WAIT_TIMEOUT:
            return None
        if wait != _WindowsPublicationKernel._WAIT_OBJECT_0:
            raise OSError("Publication process wait failed")
        code = wintypes.DWORD()
        if not self._kernel32.GetExitCodeProcess(
            wintypes.HANDLE(self._handle),
            ctypes.byref(code),
        ):
            raise OSError("Publication process exit code is unavailable")
        self.returncode = int(code.value)
        return self.returncode

    def _bounded_output_size(
        self,
        stream: Any,
        limit_bytes: int | None,
        *,
        label: str,
    ) -> int:
        size = int(os.fstat(stream.fileno()).st_size)
        if limit_bytes is not None and size > limit_bytes:
            raise SecGemmaOnlineRiskOverlayProductionError(
                f"Publication process {label} exceeded its fixed byte cap"
            )
        return size

    def communicate(self, timeout: float) -> tuple[bytes, bytes]:
        try:
            timeout_seconds = float(timeout)
            deadline = time.monotonic() + timeout_seconds
        except Exception:
            raise subprocess.TimeoutExpired(
                "contained-publication-process",
                timeout,
            ) from None
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise subprocess.TimeoutExpired(
                "contained-publication-process",
                timeout,
            )
        while True:
            self._bounded_output_size(
                self._stdout_file,
                self._stdout_limit_bytes,
                label="stdout",
            )
            self._bounded_output_size(
                self._stderr_file,
                self._stderr_limit_bytes,
                label="stderr",
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(
                    "contained-publication-process",
                    timeout,
                )
            milliseconds = max(
                1,
                min(10, math.ceil(remaining * 1000.0)),
            )
            wait = self._kernel32.WaitForSingleObject(
                wintypes.HANDLE(self._handle),
                milliseconds,
            )
            if wait == _WindowsPublicationKernel._WAIT_TIMEOUT:
                continue
            if wait != _WindowsPublicationKernel._WAIT_OBJECT_0:
                raise OSError("Publication process wait failed")
            break
        self.poll()
        self._bounded_output_size(
            self._stdout_file,
            self._stdout_limit_bytes,
            label="stdout",
        )
        self._bounded_output_size(
            self._stderr_file,
            self._stderr_limit_bytes,
            label="stderr",
        )
        self._stdout_file.seek(0)
        self._stderr_file.seek(0)
        stdout_read_limit = (
            -1
            if self._stdout_limit_bytes is None
            else self._stdout_limit_bytes + 1
        )
        stderr_read_limit = (
            -1
            if self._stderr_limit_bytes is None
            else self._stderr_limit_bytes + 1
        )
        return (
            self._stdout_file.read(stdout_read_limit),
            self._stderr_file.read(stderr_read_limit),
        )

    def kill(self) -> None:
        if self.poll() is not None:
            return
        if not self._kernel32.TerminateProcess(
            wintypes.HANDLE(self._handle),
            125,
        ):
            raise OSError("Publication process could not be terminated")

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        try:
            if not self._kernel32.CloseHandle(
                wintypes.HANDLE(self._handle)
            ):
                failures.append(
                    OSError("Publication process handle could not be closed")
                )
        except BaseException as exc:
            failures.append(exc)
        for stream in (
            self._stdin_file,
            self._stdout_file,
            self._stderr_file,
        ):
            try:
                stream.close()
            except BaseException as exc:
                failures.append(exc)
        self._handle = 0
        self._closed = True
        if failures:
            raise failures[0]


def _atomic_publication_environment_block(
    environment: Mapping[str, str],
) -> ctypes.Array[Any]:
    entries = sorted(
        environment.items(),
        key=lambda item: item[0].casefold(),
    )
    return ctypes.create_unicode_buffer(
        "\0".join(f"{name}={value}" for name, value in entries) + "\0\0"
    )


def _create_atomic_publication_process(
    *,
    kernel32: Any,
    job_handle: Any,
    argv: tuple[str, ...],
    cwd: Path,
    env: Mapping[str, str],
    input_bytes: bytes | None = None,
    stdout_limit_bytes: int | None = None,
    stderr_limit_bytes: int | None = None,
) -> _WindowsAtomicPublicationProcess:
    """Create suspended and atomically job-assigned via STARTUPINFOEX."""

    if os.name != "nt" or msvcrt is None:
        raise OSError("Atomic publication process creation requires Windows")
    if (
        input_bytes is not None
        and (
            type(input_bytes) is not bytes
            or len(input_bytes) > _STDIO_WORKER_MAX_REQUEST_BYTES
        )
    ):
        raise OSError("Atomic publication process input is invalid")
    for limit_bytes in (stdout_limit_bytes, stderr_limit_bytes):
        if limit_bytes is not None and (
            type(limit_bytes) is not int or limit_bytes <= 0
        ):
            raise OSError("Atomic publication process output cap is invalid")
    files: list[Any] = []
    handles: tuple[int, ...] = ()
    attribute_buffer: Any | None = None
    attribute_list: Any | None = None
    attribute_initialized = False
    process_information = _ProcessInformation()
    created = False
    try:
        if input_bytes is None:
            files.append(open(os.devnull, "rb", buffering=0))
        else:
            stdin_file = tempfile.TemporaryFile()
            stdin_file.write(input_bytes)
            stdin_file.flush()
            stdin_file.seek(0)
            files.append(stdin_file)
        files.append(tempfile.TemporaryFile())
        files.append(tempfile.TemporaryFile())
        handles = tuple(
            int(msvcrt.get_osfhandle(stream.fileno()))
            for stream in files
        )
        for handle in handles:
            os.set_handle_inheritable(handle, True)
        size = ctypes.c_size_t()
        kernel32.InitializeProcThreadAttributeList(
            None,
            2,
            0,
            ctypes.byref(size),
        )
        if size.value <= 0:
            raise OSError(
                "Publication process attribute-list sizing failed"
            )
        attribute_buffer = ctypes.create_string_buffer(size.value)
        attribute_list = ctypes.cast(
            attribute_buffer,
            ctypes.c_void_p,
        )
        if not kernel32.InitializeProcThreadAttributeList(
            attribute_list,
            2,
            0,
            ctypes.byref(size),
        ):
            raise OSError(
                "Publication process attribute-list initialization failed"
            )
        attribute_initialized = True
        job_values = (wintypes.HANDLE * 1)(job_handle)
        if not kernel32.UpdateProcThreadAttribute(
            attribute_list,
            0,
            _WindowsPublicationKernel._PROC_THREAD_ATTRIBUTE_JOB_LIST,
            ctypes.cast(job_values, ctypes.c_void_p),
            ctypes.sizeof(job_values),
            None,
            None,
        ):
            raise OSError(
                "Publication process atomic Job assignment failed"
            )
        inherited_values = (wintypes.HANDLE * len(handles))(*handles)
        if not kernel32.UpdateProcThreadAttribute(
            attribute_list,
            0,
            _WindowsPublicationKernel._PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            ctypes.cast(inherited_values, ctypes.c_void_p),
            ctypes.sizeof(inherited_values),
            None,
            None,
        ):
            raise OSError(
                "Publication process inherited-handle restriction failed"
            )
        startup = _StartupInfoExW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = (
            _WindowsPublicationKernel._STARTF_USESTDHANDLES
        )
        startup.StartupInfo.hStdInput = wintypes.HANDLE(handles[0])
        startup.StartupInfo.hStdOutput = wintypes.HANDLE(handles[1])
        startup.StartupInfo.hStdError = wintypes.HANDLE(handles[2])
        startup.lpAttributeList = attribute_list
        command_line = ctypes.create_unicode_buffer(
            subprocess.list2cmdline(argv)
        )
        environment_block = _atomic_publication_environment_block(env)
        if not kernel32.CreateProcessW(
            argv[0],
            command_line,
            None,
            None,
            True,
            (
                _WindowsPublicationKernel._CREATE_SUSPENDED
                | _WindowsPublicationKernel._CREATE_NO_WINDOW
                | _WindowsPublicationKernel._CREATE_UNICODE_ENVIRONMENT
                | _WindowsPublicationKernel._EXTENDED_STARTUPINFO_PRESENT
            ),
            environment_block,
            str(cwd),
            ctypes.byref(startup),
            ctypes.byref(process_information),
        ):
            raise OSError(
                ctypes.get_last_error(),
                "Publication CreateProcessW failed",
            )
        created = True
        prior = kernel32.ResumeThread(process_information.hThread)
        if prior != 1:
            raise OSError(
                "Publication child was not exactly once suspended"
            )
        if not kernel32.CloseHandle(process_information.hThread):
            raise OSError(
                "Publication primary thread handle could not be closed"
            )
        process_information.hThread = None
        return _WindowsAtomicPublicationProcess(
            kernel32=kernel32,
            process_handle=int(process_information.hProcess),
            process_id=int(process_information.dwProcessId),
            stdin_file=files[0],
            stdout_file=files[1],
            stderr_file=files[2],
            stdout_limit_bytes=stdout_limit_bytes,
            stderr_limit_bytes=stderr_limit_bytes,
        )
    except BaseException:
        if created:
            try:
                kernel32.TerminateJobObject(job_handle, 125)
            except BaseException:
                pass
            try:
                kernel32.WaitForSingleObject(
                    process_information.hProcess,
                    5000,
                )
            except BaseException:
                pass
        if process_information.hThread:
            try:
                kernel32.CloseHandle(process_information.hThread)
            except BaseException:
                pass
        if process_information.hProcess:
            try:
                kernel32.CloseHandle(process_information.hProcess)
            except BaseException:
                pass
        for stream in files:
            try:
                stream.close()
            except BaseException:
                pass
        raise
    finally:
        for handle in handles:
            try:
                os.set_handle_inheritable(handle, False)
            except BaseException:
                pass
        if attribute_list is not None and attribute_initialized:
            try:
                kernel32.DeleteProcThreadAttributeList(attribute_list)
            except BaseException:
                pass


_WINDOWS_JOB_OBJECT_BASIC_PROCESS_ID_LIST: Final[int] = 3
_WINDOWS_ERROR_MORE_DATA: Final[int] = 234
_WINDOWS_ERROR_INVALID_PARAMETER: Final[int] = 87
_WINDOWS_SYNCHRONIZE: Final[int] = 0x00100000
_WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION: Final[int] = 0x00001000
_WINDOWS_MAX_JOB_PROCESS_IDS: Final[int] = 1_048_576


def _windows_job_process_ids(
    *,
    kernel32: Any,
    job: Any,
) -> tuple[int, ...]:
    """Return a complete, dynamically sized Job process-id snapshot."""

    process_id_offset = _JobObjectBasicProcessIdList.ProcessIdList.offset
    pointer_size = ctypes.sizeof(ctypes.c_size_t)
    capacity = 16
    while capacity <= _WINDOWS_MAX_JOB_PROCESS_IDS:
        buffer_size = process_id_offset + (capacity * pointer_size)
        buffer = ctypes.create_string_buffer(buffer_size)
        returned = wintypes.DWORD()
        ctypes.set_last_error(0)
        queried = bool(
            kernel32.QueryInformationJobObject(
                job,
                _WINDOWS_JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                buffer,
                buffer_size,
                ctypes.byref(returned),
            )
        )
        header = ctypes.cast(
            buffer,
            ctypes.POINTER(_JobObjectBasicProcessIdList),
        ).contents
        assigned = int(header.NumberOfAssignedProcesses)
        listed = int(header.NumberOfProcessIdsInList)
        if listed > capacity or listed > assigned:
            raise OSError("Contained Job process-id evidence is malformed")
        if queried and listed == assigned:
            process_ids_array = (ctypes.c_size_t * listed).from_address(
                ctypes.addressof(buffer) + process_id_offset
            )
            process_ids = tuple(int(value) for value in process_ids_array)
            if any(process_id <= 0 for process_id in process_ids) or len(
                set(process_ids)
            ) != len(process_ids):
                raise OSError(
                    "Contained Job process-id evidence is malformed"
                )
            return process_ids
        if queried:
            capacity = max(capacity * 2, assigned, listed + 1)
            continue
        error = ctypes.get_last_error()
        if error != _WINDOWS_ERROR_MORE_DATA:
            raise OSError(
                error,
                "Contained Job process-id evidence is unavailable",
            )
        returned_capacity = 0
        if int(returned.value) > process_id_offset:
            returned_capacity = math.ceil(
                (int(returned.value) - process_id_offset) / pointer_size
            )
        capacity = max(
            capacity * 2,
            capacity + 1,
            assigned,
            listed + 1,
            returned_capacity,
        )
    raise OSError("Contained Job process-id evidence is oversized")


class _WindowsJobProcessSignalWitnesses:
    """Hold process objects until every Job member signals termination."""

    __slots__ = ("_closed", "_handles", "_job", "_kernel32")

    def __init__(self, *, kernel32: Any, job: Any) -> None:
        self._kernel32 = kernel32
        self._job = job
        self._handles: dict[int, Any] = {}
        self._closed = False

    def capture(self) -> None:
        if self._closed:
            raise OSError("Contained Job process witnesses are closed")
        for process_id in _windows_job_process_ids(
            kernel32=self._kernel32,
            job=self._job,
        ):
            if process_id in self._handles:
                continue
            ctypes.set_last_error(0)
            handle = self._kernel32.OpenProcess(
                _WINDOWS_SYNCHRONIZE
                | _WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                process_id,
            )
            if not handle:
                error = ctypes.get_last_error()
                if error == _WINDOWS_ERROR_INVALID_PARAMETER:
                    continue
                raise OSError(
                    error,
                    "Contained Job process witness could not be opened",
                )
            keep_handle = False
            try:
                inside = wintypes.BOOL()
                if not self._kernel32.IsProcessInJob(
                    handle,
                    self._job,
                    ctypes.byref(inside),
                ):
                    raise OSError(
                        ctypes.get_last_error(),
                        "Contained Job process membership is unavailable",
                    )
                if bool(inside.value):
                    self._handles[process_id] = handle
                    keep_handle = True
            finally:
                if not keep_handle and not self._kernel32.CloseHandle(handle):
                    raise OSError(
                        "Rejected contained-process witness could not be closed"
                    )

    def all_signaled(self) -> bool:
        if self._closed:
            raise OSError("Contained Job process witnesses are closed")
        all_signaled = True
        for handle in self._handles.values():
            wait = self._kernel32.WaitForSingleObject(handle, 0)
            if wait == _WindowsPublicationKernel._WAIT_TIMEOUT:
                all_signaled = False
            elif wait != _WindowsPublicationKernel._WAIT_OBJECT_0:
                raise OSError("Contained Job process wait failed")
        return all_signaled

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        for handle in self._handles.values():
            try:
                closed = bool(self._kernel32.CloseHandle(handle))
            except BaseException as exc:
                if failure is None:
                    failure = exc
            else:
                if not closed and failure is None:
                    failure = OSError(
                        "Contained Job process witness could not be closed"
                    )
        self._handles.clear()
        self._closed = True
        if failure is not None:
            raise failure


def _windows_terminate_job_and_wait_for_process_signals(
    *,
    kernel32: Any,
    job: Any,
    terminate_job: Callable[[], None],
    active_process_count: Callable[[], int],
    deadline_monotonic: float,
    context: str,
) -> None:
    """Terminate a Job and prove accounting plus process-object quiescence."""

    witnesses = _WindowsJobProcessSignalWitnesses(
        kernel32=kernel32,
        job=job,
    )
    failure: BaseException | None = None
    complete = False

    def checked_now() -> float:
        try:
            now = float(time.monotonic())
        except Exception:
            raise OSError(f"{context} cleanup clock failed") from None
        if not math.isfinite(now) or now >= deadline_monotonic:
            raise OSError(f"{context} did not become quiescent")
        return now

    try:
        checked_now()
        witnesses.capture()
        checked_now()
        terminate_job()
        while True:
            now = checked_now()
            # Capture on both sides of accounting. A child assigned after the
            # pre-termination snapshot is therefore retained before success.
            witnesses.capture()
            active_before = active_process_count()
            witnesses.capture()
            active_after = active_process_count()
            all_signaled = witnesses.all_signaled()
            checked_now()
            if (
                type(active_before) is not int
                or type(active_after) is not int
                or active_before < 0
                or active_after < 0
            ):
                raise OSError(f"{context} accounting is malformed")
            if active_before == 0 and active_after == 0 and all_signaled:
                complete = True
                break
            time.sleep(min(0.01, deadline_monotonic - now))
    except BaseException as exc:
        failure = exc
    finally:
        try:
            witnesses.close()
        except BaseException as close_error:
            close_error.__cause__ = failure
            failure = close_error
    if failure is not None:
        raise failure
    if not complete:
        raise OSError(f"{context} did not become quiescent")


class _WindowsPublicationKernel:
    """Own one kill-on-close Windows Job Object and exclusive owner mutex."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    _JOB_OBJECT_QUERY = 0x0004
    _PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    _PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
    _STARTF_USESTDHANDLES = 0x00000100
    _CREATE_SUSPENDED = 0x00000004
    _CREATE_UNICODE_ENVIRONMENT = 0x00000400
    _EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    _CREATE_NO_WINDOW = 0x08000000
    _ERROR_ALREADY_EXISTS = 183
    _ERROR_FILE_NOT_FOUND = 2
    _ERROR_INVALID_PARAMETER = 87
    _WAIT_OBJECT_0 = 0
    _WAIT_ABANDONED = 0x00000080
    _WAIT_TIMEOUT = 0x00000102
    _INFINITE = 0xFFFFFFFF

    def __init__(self, *, job_name: str, mutex_name: str) -> None:
        if os.name != "nt":
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication containment requires Windows"
            )
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()
        ctypes.set_last_error(0)
        self._job = self._kernel32.CreateJobObjectW(None, job_name)
        if not self._job:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication Job Object could not be created"
            )
        if ctypes.get_last_error() == self._ERROR_ALREADY_EXISTS:
            self._kernel32.CloseHandle(self._job)
            self._job = None
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication Job Object identity already exists"
            )
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self._kernel32.SetInformationJobObject(
            self._job,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            self._kernel32.CloseHandle(self._job)
            self._job = None
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication Job Object kill-on-close policy failed"
            )
        ctypes.set_last_error(0)
        self._mutex = self._kernel32.CreateMutexW(
            None,
            True,
            mutex_name,
        )
        if not self._mutex:
            self._kernel32.CloseHandle(self._job)
            self._job = None
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication owner mutex could not be created"
            )
        if ctypes.get_last_error() == self._ERROR_ALREADY_EXISTS:
            self._kernel32.CloseHandle(self._mutex)
            self._kernel32.CloseHandle(self._job)
            self._mutex = None
            self._job = None
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication owner mutex identity already exists"
            )
        self._closed = False
        self._terminated = False
        self._recorded_processes: list[tuple[int, str]] = []

    def _configure_signatures(self) -> None:
        kernel32 = self._kernel32
        kernel32.CreateJobObjectW.argtypes = [
            ctypes.c_void_p,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
        ]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.InitializeProcThreadAttributeList.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        kernel32.UpdateProcThreadAttribute.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
        kernel32.DeleteProcThreadAttributeList.argtypes = [
            ctypes.c_void_p
        ]
        kernel32.DeleteProcThreadAttributeList.restype = None
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.POINTER(_StartupInfoExW),
            ctypes.POINTER(_ProcessInformation),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
        ]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.IsProcessInJob.restype = wintypes.BOOL

    def _active_process_count(self) -> int:
        if self._job is None:
            return 0
        accounting = _JobObjectBasicAccountingInformation()
        returned = wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            self._job,
            self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned),
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication Job Object accounting is unavailable"
            )
        return int(accounting.ActiveProcesses)

    def _process_creation_filetime_hex(self, process: Any) -> str:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        handle = wintypes.HANDLE(int(process._handle))
        if not self._kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise OSError("Publication child identity is unavailable")
        value = (int(creation.dwHighDateTime) << 32) | int(
            creation.dwLowDateTime
        )
        return hex(value)

    def _guarded_terminate_process(
        self,
        process: Any | None,
        *,
        exit_code: int,
    ) -> None:
        if self._job is not None:
            try:
                self._kernel32.TerminateJobObject(
                    self._job,
                    exit_code,
                )
            except BaseException:
                pass
            self._terminated = True
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
        except BaseException:
            pass
        try:
            process.communicate(timeout=5.0)
        except BaseException:
            pass
        try:
            process.close()
        except BaseException:
            pass

    def execute(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> Any:
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            PublicationProcessExecution,
        )

        if self._closed or self._terminated or self._job is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication worker containment is no longer active"
            )
        process: Any | None = None
        try:
            process = _create_atomic_publication_process(
                kernel32=self._kernel32,
                job_handle=self._job,
                argv=argv,
                cwd=cwd,
                env=dict(env),
            )
            self._recorded_processes.append(
                (
                    int(process.pid),
                    self._process_creation_filetime_hex(process),
                )
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=timeout_seconds
                )
            except subprocess.TimeoutExpired:
                self._guarded_terminate_process(process, exit_code=124)
                return PublicationProcessExecution(
                    process_exit_status="deadline",
                    process_exit_code=None,
                    stdout=b"",
                    stderr=b"",
                )
            except KeyboardInterrupt:
                self._guarded_terminate_process(process, exit_code=130)
                return PublicationProcessExecution(
                    process_exit_status="parent_interrupted",
                    process_exit_code=None,
                    stdout=b"",
                    stderr=b"",
                )
            accounting_deadline = time.monotonic() + 1.0
            active = self._active_process_count()
            while active and time.monotonic() < accounting_deadline:
                time.sleep(0.01)
                active = self._active_process_count()
            if active:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "A publication descendant outlived its top-level Git command"
                )
            result = PublicationProcessExecution(
                process_exit_status="exited",
                process_exit_code=int(process.returncode),
                stdout=bytes(stdout),
                stderr=bytes(stderr),
            )
            process.close()
            return result
        except KeyboardInterrupt:
            self._guarded_terminate_process(process, exit_code=130)
            return PublicationProcessExecution(
                process_exit_status="parent_interrupted",
                process_exit_code=None,
                stdout=b"",
                stderr=b"",
            )
        except OSError:
            self._guarded_terminate_process(process, exit_code=125)
            return PublicationProcessExecution(
                process_exit_status="spawn_failed",
                process_exit_code=None,
                stdout=b"",
                stderr=b"",
            )
        except BaseException:
            self._guarded_terminate_process(process, exit_code=125)
            raise

    def quiesce(self) -> tuple[int, int]:
        if self._closed:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication worker containment was already released"
            )
        if self._job is None or self._mutex is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication worker containment is incomplete"
            )
        job = self._job
        mutex = self._mutex
        failure: BaseException | None = None
        mutex_released = False
        alive = 0
        try:
            if not self._kernel32.TerminateJobObject(job, 0):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication Job Object could not be terminated"
                )
            self._terminated = True
            deadline = time.monotonic() + 5.0
            active = self._active_process_count()
            while active and time.monotonic() < deadline:
                time.sleep(0.01)
                active = self._active_process_count()
            if active:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication Job Object did not become quiescent"
                )
            alive = sum(
                not _windows_prior_process_is_dead(
                    process_id,
                    creation,
                )
                for process_id, creation in self._recorded_processes
            )
            if alive:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "A recorded publication process remains live"
                )
            if not self._kernel32.ReleaseMutex(mutex):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication owner mutex could not be released"
                )
            mutex_released = True
        except BaseException as exc:
            failure = exc
        finally:
            if failure is not None:
                try:
                    self._kernel32.TerminateJobObject(job, 125)
                except BaseException:
                    pass
            self._terminated = True
            if not mutex_released:
                try:
                    mutex_released = bool(
                        self._kernel32.ReleaseMutex(mutex)
                    )
                except BaseException:
                    mutex_released = False
            if not mutex_released and failure is None:
                failure = SecGemmaOnlineRiskOverlayProductionError(
                    "Publication owner mutex could not be released"
                )
            for handle, label in (
                (mutex, "owner mutex"),
                (job, "Job Object"),
            ):
                try:
                    closed = bool(self._kernel32.CloseHandle(handle))
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                else:
                    if not closed and failure is None:
                        failure = (
                            SecGemmaOnlineRiskOverlayProductionError(
                                f"Publication {label} could not be closed"
                            )
                        )
            self._mutex = None
            self._job = None
            self._closed = True
        if failure is not None:
            raise failure
        return 0, alive

    def close_uncommitted(self) -> None:
        self.quiesce()


class _WindowsPublicationRecoverySupervisor:
    """Own the outer kill-on-close Job for one full recovery invocation."""

    __slots__ = (
        "_cleanup_deadline",
        "_closed",
        "_job",
        "_job_name",
        "_kernel32",
        "_terminated",
    )

    def __init__(self, *, job_name: str) -> None:
        if (
            os.name != "nt"
            or type(job_name) is not str
            or not job_name.startswith(
                "Local\\CodexPublicationRecoveryJob-"
            )
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervision requires Windows "
                "and one exact Job identity"
            )
        self._kernel32 = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )
        _WindowsPublicationKernel._configure_signatures(self)
        ctypes.set_last_error(0)
        self._job = self._kernel32.CreateJobObjectW(None, job_name)
        if not self._job:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery Job Object could not be created"
            )
        if (
            ctypes.get_last_error()
            == _WindowsPublicationKernel._ERROR_ALREADY_EXISTS
        ):
            self._kernel32.CloseHandle(self._job)
            self._job = None
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery Job Object identity already exists"
            )
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = (
            _WindowsPublicationKernel._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self._kernel32.SetInformationJobObject(
            self._job,
            _WindowsPublicationKernel._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            self._kernel32.CloseHandle(self._job)
            self._job = None
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery Job kill-on-close policy failed"
            )
        self._job_name = job_name
        self._cleanup_deadline: float | None = None
        self._closed = False
        self._terminated = False

    def _active_process_count(self) -> int:
        if self._job is None:
            return 0
        accounting = _JobObjectBasicAccountingInformation()
        returned = wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            self._job,
            _WindowsPublicationKernel._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned),
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery Job accounting is unavailable"
            )
        return int(accounting.ActiveProcesses)

    def _terminate_tree(self, *, exit_code: int) -> None:
        if self._job is None or self._terminated:
            return
        try:
            terminated = bool(
                self._kernel32.TerminateJobObject(
                    self._job,
                    exit_code,
                )
            )
        except BaseException:
            terminated = False
        if not terminated:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery process tree could not be terminated"
            )
        self._terminated = True

    def _wait_for_zero_active(
        self,
        *,
        deadline_monotonic: float,
    ) -> None:
        while True:
            try:
                now = float(time.monotonic())
            except Exception:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication-recovery cleanup clock failed"
                ) from None
            if not math.isfinite(now) or now >= deadline_monotonic:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication-recovery Job did not become quiescent"
                )
            if self._active_process_count() == 0:
                return
            time.sleep(min(0.01, deadline_monotonic - now))

    def _terminate_and_wait(
        self,
        *,
        exit_code: int,
        deadline_monotonic: float,
    ) -> None:
        if self._job is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery Job is unavailable"
            )
        try:
            _windows_terminate_job_and_wait_for_process_signals(
                kernel32=self._kernel32,
                job=self._job,
                terminate_job=lambda: self._terminate_tree(
                    exit_code=exit_code
                ),
                active_process_count=self._active_process_count,
                deadline_monotonic=deadline_monotonic,
                context="Publication-recovery Job",
            )
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayProductionError(str(exc)) from exc

    def run(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        env: Mapping[str, str],
        input_bytes: bytes,
        deadline_monotonic: float,
    ) -> bytes:
        if (
            self._closed
            or self._terminated
            or self._job is None
            or type(input_bytes) is not bytes
            or not input_bytes
            or len(input_bytes) > _STDIO_WORKER_MAX_REQUEST_BYTES
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervisor is unavailable"
            )
        try:
            deadline = float(deadline_monotonic)
            remaining = deadline - float(time.monotonic())
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervisor clock failed"
            ) from None
        if (
            not math.isfinite(deadline)
            or not (
                _PUBLICATION_RECOVERY_CLEANUP_RESERVE_SECONDS
                < remaining
                <= MAX_PUBLICATION_RECOVERY_SECONDS
            )
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervisor deadline is invalid"
            )
        self._cleanup_deadline = deadline
        process: _WindowsAtomicPublicationProcess | None = None
        output: bytes | None = None
        primary_error: BaseException | None = None
        cleanup_exit_code = 125
        try:
            process = _create_atomic_publication_process(
                kernel32=self._kernel32,
                job_handle=self._job,
                argv=argv,
                cwd=cwd,
                env=dict(env),
                input_bytes=input_bytes,
                stdout_limit_bytes=_STDIO_WORKER_MAX_RESPONSE_BYTES,
                stderr_limit_bytes=_STDIO_WORKER_MAX_RESPONSE_BYTES,
            )
            remaining = deadline - float(time.monotonic())
            worker_timeout = (
                remaining
                - _PUBLICATION_RECOVERY_CLEANUP_RESERVE_SECONDS
            )
            if worker_timeout <= 0.0:
                cleanup_exit_code = 124
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication recovery exceeded its 300-second deadline"
                )
            try:
                stdout, stderr = process.communicate(
                    timeout=worker_timeout
                )
            except subprocess.TimeoutExpired:
                cleanup_exit_code = 124
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication recovery exceeded its 300-second deadline"
                ) from None
            except KeyboardInterrupt:
                cleanup_exit_code = 130
                raise
            if float(time.monotonic()) >= (
                deadline
                - _PUBLICATION_RECOVERY_CLEANUP_RESERVE_SECONDS
            ):
                cleanup_exit_code = 124
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication recovery exceeded its 300-second deadline"
                )
            if (
                process.returncode != 0
                or type(stdout) is not bytes
                or not stdout
                or len(stdout) > _STDIO_WORKER_MAX_RESPONSE_BYTES
                or type(stderr) is not bytes
                or stderr
            ):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Supervised publication-recovery worker failed closed"
                )
            accounting_deadline = min(
                deadline,
                float(time.monotonic()) + 1.0,
            )
            active = self._active_process_count()
            while (
                active
                and float(time.monotonic()) < accounting_deadline
            ):
                time.sleep(0.01)
                active = self._active_process_count()
            if active:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "A publication-recovery descendant outlived its worker"
                )
            output = bytes(stdout)
        except BaseException as exc:
            primary_error = exc
        finally:
            if primary_error is not None:
                try:
                    self._terminate_and_wait(
                        exit_code=cleanup_exit_code,
                        deadline_monotonic=deadline,
                    )
                except BaseException as cleanup_error:
                    cleanup_error.__cause__ = primary_error
                    primary_error = cleanup_error
            if process is not None:
                try:
                    process.close()
                except BaseException as close_error:
                    close_error.__cause__ = primary_error
                    primary_error = close_error
                    if not self._terminated and self._job is not None:
                        try:
                            self._terminate_and_wait(
                                exit_code=125,
                                deadline_monotonic=deadline,
                            )
                        except BaseException:
                            pass
        if primary_error is not None:
            raise primary_error
        if output is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Supervised publication-recovery worker returned no output"
            )
        return output

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        job = self._job
        try:
            now = float(time.monotonic())
        except Exception:
            now = 0.0
        cleanup_deadline = (
            self._cleanup_deadline
            if self._cleanup_deadline is not None
            else now + _PUBLICATION_RECOVERY_CLEANUP_RESERVE_SECONDS
        )
        if job is not None:
            try:
                self._terminate_and_wait(
                    exit_code=0,
                    deadline_monotonic=cleanup_deadline,
                )
            except BaseException as exc:
                failure = exc
        if job is not None:
            try:
                closed = bool(self._kernel32.CloseHandle(job))
            except BaseException as exc:
                if failure is None:
                    failure = exc
            else:
                if not closed and failure is None:
                    failure = SecGemmaOnlineRiskOverlayProductionError(
                        "Publication-recovery Job could not be closed"
                    )
        self._job = None
        self._closed = True
        self._terminated = True
        if failure is not None:
            raise failure


class _WindowsFinalRegistrySubprocessRunner:
    """Run each final-registry Git command in one ephemeral atomic Job."""

    __slots__ = ("_kernel32", "_token_bytes")

    def __init__(
        self,
        *,
        token_bytes: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        if os.name != "nt" or not callable(token_bytes):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Final-registry containment requires Windows"
            )
        self._kernel32 = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )
        _WindowsPublicationKernel._configure_signatures(self)
        self._token_bytes = token_bytes

    def _create_job(self) -> Any:
        token = self._token_bytes(32)
        if type(token) is not bytes or len(token) != 32:
            raise OSError(
                "Final-registry Job nonce source changed"
            )
        name = (
            "Local\\CodexFinalRegistryCommandJob-"
            + hashlib.sha256(token).hexdigest()
        )
        ctypes.set_last_error(0)
        job = self._kernel32.CreateJobObjectW(None, name)
        if not job:
            raise OSError("Final-registry Job could not be created")
        if (
            ctypes.get_last_error()
            == _WindowsPublicationKernel._ERROR_ALREADY_EXISTS
        ):
            self._kernel32.CloseHandle(job)
            raise OSError(
                "Final-registry Job identity already exists"
            )
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = (
            _WindowsPublicationKernel._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self._kernel32.SetInformationJobObject(
            job,
            _WindowsPublicationKernel._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            self._kernel32.CloseHandle(job)
            raise OSError(
                "Final-registry Job kill-on-close policy failed"
            )
        return job

    def _active_process_count(self, job: Any) -> int:
        accounting = _JobObjectBasicAccountingInformation()
        returned = wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            job,
            _WindowsPublicationKernel._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned),
        ):
            raise OSError(
                "Final-registry Job accounting is unavailable"
            )
        return int(accounting.ActiveProcesses)

    def _terminate_and_wait(
        self,
        job: Any,
        *,
        exit_code: int,
        deadline_monotonic: float,
    ) -> None:
        def terminate_job() -> None:
            if not self._kernel32.TerminateJobObject(job, exit_code):
                raise OSError(
                    "Final-registry process tree could not be terminated"
                )

        _windows_terminate_job_and_wait_for_process_signals(
            kernel32=self._kernel32,
            job=job,
            terminate_job=terminate_job,
            active_process_count=lambda: self._active_process_count(job),
            deadline_monotonic=deadline_monotonic,
            context="Final-registry process tree",
        )

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        input: bytes | None,
        stdout: int,
        stderr: int,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        if (
            type(args) not in {list, tuple}
            or not args
            or any(
                type(argument) is not str
                or not argument
                or "\x00" in argument
                for argument in args
            )
            or not isinstance(cwd, Path)
            or not cwd.is_absolute()
            or type(env) is not dict
            or any(
                type(name) is not str
                or not name
                or "\x00" in name
                or type(value) is not str
                or "\x00" in value
                for name, value in env.items()
            )
            or (input is not None and type(input) is not bytes)
            or (
                type(input) is bytes
                and len(input) > _STDIO_WORKER_MAX_REQUEST_BYTES
            )
            or stdout != subprocess.PIPE
            or stderr != subprocess.PIPE
            or check is not False
            or type(timeout) not in {int, float}
        ):
            raise OSError(
                "Final-registry contained command profile changed"
            )
        requested_timeout = float(timeout)
        if (
            not math.isfinite(requested_timeout)
            or requested_timeout
            <= _FINAL_REGISTRY_CLEANUP_RESERVE_SECONDS
            or requested_timeout > 300.0
        ):
            raise subprocess.TimeoutExpired(args, requested_timeout)
        argv = tuple(args)
        try:
            executable = Path(argv[0])
            resolved_executable = executable.resolve(strict=True)
            resolved_cwd = cwd.resolve(strict=True)
        except OSError as exc:
            raise OSError(
                "Final-registry contained command path is unavailable"
            ) from exc
        if (
            not executable.is_absolute()
            or executable != resolved_executable
            or cwd != resolved_cwd
            or not resolved_cwd.is_dir()
        ):
            raise OSError(
                "Final-registry contained command path is not exact"
            )
        try:
            started = float(time.monotonic())
        except Exception as exc:
            raise OSError(
                "Final-registry containment clock failed"
            ) from exc
        if not math.isfinite(started) or started < 0.0:
            raise OSError(
                "Final-registry containment clock is invalid"
            )
        deadline = started + requested_timeout
        job = self._create_job()
        process: _WindowsAtomicPublicationProcess | None = None
        result: subprocess.CompletedProcess[bytes] | None = None
        primary_error: BaseException | None = None
        exit_code = 125
        try:
            remaining = deadline - float(time.monotonic())
            command_timeout = (
                remaining - _FINAL_REGISTRY_CLEANUP_RESERVE_SECONDS
            )
            if command_timeout <= 0.0:
                exit_code = 124
                raise subprocess.TimeoutExpired(
                    argv,
                    requested_timeout,
                )
            process = _create_atomic_publication_process(
                kernel32=self._kernel32,
                job_handle=job,
                argv=argv,
                cwd=resolved_cwd,
                env=dict(env),
                input_bytes=input,
                stdout_limit_bytes=_FINAL_REGISTRY_MAX_COMMAND_IO_BYTES,
                stderr_limit_bytes=_FINAL_REGISTRY_MAX_COMMAND_IO_BYTES,
            )
            try:
                command_stdout, command_stderr = process.communicate(
                    timeout=command_timeout
                )
            except subprocess.TimeoutExpired:
                exit_code = 124
                raise subprocess.TimeoutExpired(
                    argv,
                    requested_timeout,
                ) from None
            except SecGemmaOnlineRiskOverlayProductionError as exc:
                raise OSError(
                    "Final-registry command output is malformed or oversized"
                ) from exc
            if (
                type(command_stdout) is not bytes
                or type(command_stderr) is not bytes
                or len(command_stdout)
                > _FINAL_REGISTRY_MAX_COMMAND_IO_BYTES
                or len(command_stderr)
                > _FINAL_REGISTRY_MAX_COMMAND_IO_BYTES
                or type(process.returncode) is not int
            ):
                raise OSError(
                    "Final-registry command output is malformed or oversized"
                )
            result = subprocess.CompletedProcess(
                args=argv,
                returncode=process.returncode,
                stdout=bytes(command_stdout),
                stderr=bytes(command_stderr),
            )
        except BaseException as exc:
            primary_error = exc
        finally:
            try:
                self._terminate_and_wait(
                    job,
                    exit_code=exit_code,
                    deadline_monotonic=deadline,
                )
            except BaseException as cleanup_error:
                cleanup_error.__cause__ = primary_error
                primary_error = cleanup_error
            if process is not None:
                try:
                    process.close()
                except BaseException as close_error:
                    close_error.__cause__ = primary_error
                    primary_error = close_error
            try:
                closed = bool(self._kernel32.CloseHandle(job))
            except BaseException as close_job_error:
                close_job_error.__cause__ = primary_error
                primary_error = close_job_error
            else:
                if not closed:
                    close_job_error = OSError(
                        "Final-registry Job could not be closed"
                    )
                    close_job_error.__cause__ = primary_error
                    primary_error = close_job_error
        if primary_error is not None:
            raise primary_error
        if result is None:
            raise OSError(
                "Final-registry contained command returned no result"
            )
        return result

    def __repr__(self) -> str:
        return "_WindowsFinalRegistrySubprocessRunner(<atomic-job>)"


def _current_process_is_in_recovery_job(job_name: str) -> bool:
    if (
        os.name != "nt"
        or type(job_name) is not str
        or not job_name.startswith(
            "Local\\CodexPublicationRecoveryJob-"
        )
    ):
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenJobObjectW.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.OpenJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.OpenJobObjectW(
        _WindowsPublicationKernel._JOB_OBJECT_QUERY,
        False,
        job_name,
    )
    if not job:
        return False
    try:
        inside = wintypes.BOOL()
        if not kernel32.IsProcessInJob(
            kernel32.GetCurrentProcess(),
            job,
            ctypes.byref(inside),
        ):
            return False
        limits = _JobObjectExtendedLimitInformation()
        returned = wintypes.DWORD()
        if not kernel32.QueryInformationJobObject(
            job,
            _WindowsPublicationKernel._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
            ctypes.byref(returned),
        ):
            return False
        return bool(inside.value) and bool(
            limits.BasicLimitInformation.LimitFlags
            & _WindowsPublicationKernel._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
    finally:
        kernel32.CloseHandle(job)


def _windows_prior_process_is_dead(
    owner_process_id: int,
    owner_process_creation_filetime_hex: str,
) -> bool:
    if os.name != "nt":
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication restart reconciliation requires Windows"
        )
    try:
        expected_creation = int(
            owner_process_creation_filetime_hex,
            16,
        )
    except (TypeError, ValueError):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Prior publication owner creation identity is invalid"
        ) from None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        0x00100000 | 0x00001000,
        False,
        owner_process_id,
    )
    if not handle:
        if ctypes.get_last_error() == 87:
            return True
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Prior publication owner process cannot be verified"
        )
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Prior publication owner creation identity is unavailable"
            )
        observed_creation = (
            int(creation.dwHighDateTime) << 32
        ) | int(creation.dwLowDateTime)
        if observed_creation != expected_creation:
            return True
        return (
            kernel32.WaitForSingleObject(handle, 0)
            == _WindowsPublicationKernel._WAIT_OBJECT_0
        )
    finally:
        kernel32.CloseHandle(handle)


def _windows_named_mutex_is_unowned(name: str) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Prior publication owner mutex cannot be verified"
        )
    try:
        wait = kernel32.WaitForSingleObject(handle, 0)
        if wait not in {
            _WindowsPublicationKernel._WAIT_OBJECT_0,
            _WindowsPublicationKernel._WAIT_ABANDONED,
        }:
            return False
        if not kernel32.ReleaseMutex(handle):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Prior publication owner mutex verification failed"
            )
        return True
    finally:
        kernel32.CloseHandle(handle)


def _windows_named_job_active_process_count(name: str) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenJobObjectW.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.OpenJobObjectW.restype = wintypes.HANDLE
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenJobObjectW(
        _WindowsPublicationKernel._JOB_OBJECT_QUERY,
        False,
        name,
    )
    if not handle:
        if ctypes.get_last_error() == (
            _WindowsPublicationKernel._ERROR_FILE_NOT_FOUND
        ):
            return 0
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Prior publication Job Object cannot be verified"
        )
    try:
        accounting = _JobObjectBasicAccountingInformation()
        returned = wintypes.DWORD()
        if not kernel32.QueryInformationJobObject(
            handle,
            _WindowsPublicationKernel._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned),
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Prior publication Job Object accounting is unavailable"
            )
        return int(accounting.ActiveProcesses)
    finally:
        kernel32.CloseHandle(handle)


def _verify_prior_publication_owner_quiescence(
    ownership: Mapping[str, Any],
) -> dict[str, Any]:
    owner = copy.deepcopy(dict(ownership))
    nonce = _sha256(
        owner.get("owner_nonce_sha256"),
        "prior publication owner nonce",
    )
    job_name = _publication_object_name("job", nonce)
    mutex_name = _publication_object_name("mutex", nonce)
    if (
        owner.get("job_object_name_sha256")
        != _publication_object_name_sha256(job_name)
        or owner.get("owner_mutex_name_sha256")
        != _publication_object_name_sha256(mutex_name)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Prior publication kernel object identity changed"
        )
    process_id = owner.get("owner_process_id")
    if type(process_id) is not int or process_id <= 0:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Prior publication owner PID is invalid"
        )
    prior_dead = _windows_prior_process_is_dead(
        process_id,
        owner.get("owner_process_creation_filetime_hex"),
    )
    mutex_unowned = _windows_named_mutex_is_unowned(mutex_name)
    active = _windows_named_job_active_process_count(job_name)
    if not prior_dead or not mutex_unowned or active != 0:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Prior publication owner or worker is still live"
        )
    return {
        "prior_owner_process_dead": True,
        "owner_mutex_unowned": True,
        "job_object_active_process_count": 0,
        "recorded_git_ssh_processes_alive_count": 0,
    }


def _identity_command(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> bytes:
    try:
        completed = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30.0,
            shell=False,
            env=dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication executable identity discovery failed"
        ) from exc
    if (
        completed.returncode != 0
        or type(completed.stdout) is not bytes
        or type(completed.stderr) is not bytes
        or completed.stderr
        or not completed.stdout
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication executable identity command failed closed"
        )
    return bytes(completed.stdout)


def _publication_host_environment(
    *,
    git_executable: Path,
) -> dict[str, str]:
    system_root = os.environ.get("SystemRoot") or "C:\\Windows"
    windir = os.environ.get("WINDIR") or system_root
    profile = os.environ.get("USERPROFILE") or str(Path.home())
    local_appdata = os.environ.get("LOCALAPPDATA") or str(
        Path(profile) / "AppData" / "Local"
    )
    appdata = os.environ.get("APPDATA") or str(
        Path(profile) / "AppData" / "Roaming"
    )
    temporary = tempfile.gettempdir()
    comspec = os.environ.get("COMSPEC") or str(
        Path(system_root) / "System32" / "cmd.exe"
    )
    pathext = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    values = {
        "SystemRoot": system_root,
        "WINDIR": windir,
        "COMSPEC": comspec,
        "PATH": str(git_executable.parent),
        "PATHEXT": pathext,
        "TEMP": temporary,
        "TMP": temporary,
        "USERPROFILE": profile,
        "LOCALAPPDATA": local_appdata,
        "APPDATA": appdata,
        "HOME": profile,
    }
    if any(type(value) is not str or not value for value in values.values()):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication child host environment is incomplete"
        )
    return values


def _discover_publication_runtime_material(
) -> tuple[dict[str, Any], dict[str, str], str]:
    from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
        FROZEN_CREDENTIAL_HELPER_PATH,
        build_transport_executable_identity_pins,
    )

    if os.name != "nt":
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production publication runtime requires Windows"
        )
    try:
        credential_helper = Path(
            FROZEN_CREDENTIAL_HELPER_PATH
        ).resolve(strict=True)
        git_root = credential_helper.parents[2]
        git_executable = (
            git_root / "mingw64" / "bin" / "git.exe"
        ).resolve(strict=True)
        command_interpreter = (
            git_root / "usr" / "bin" / "sh.exe"
        ).resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production publication real executable closure is unavailable"
        ) from exc
    host = _publication_host_environment(git_executable=git_executable)
    discovery_environment = {
        "SystemRoot": host["SystemRoot"],
        "WINDIR": host["WINDIR"],
        "COMSPEC": host["COMSPEC"],
        "PATH": host["PATH"],
        "PATHEXT": host["PATHEXT"],
        "TEMP": host["TEMP"],
        "TMP": host["TMP"],
        "HOME": host["HOME"],
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "NUL",
        "GIT_CONFIG_GLOBAL": "NUL",
        "GIT_CONFIG_COUNT": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    git_version = _identity_command(
        (str(git_executable), "--version"),
        environment=discovery_environment,
    )
    raw_exec_path = _identity_command(
        (str(git_executable), "--exec-path"),
        environment=discovery_environment,
    )
    raw_shell_path = _identity_command(
        (str(git_executable), "var", "GIT_SHELL_PATH"),
        environment=discovery_environment,
    )
    try:
        git_exec_path = Path(
            raw_exec_path.decode("utf-8", errors="strict").strip()
        ).resolve(strict=True)
        observed_shell = Path(
            raw_shell_path.decode("utf-8", errors="strict").strip()
        ).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production Git runtime path identity is invalid"
        ) from exc
    if observed_shell != command_interpreter:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production Git compiled shell path changed"
        )
    git_remote_https = (
        git_exec_path / "git-remote-https.exe"
    ).resolve(strict=True)
    helper_version = _identity_command(
        (str(credential_helper), "--version"),
        environment=discovery_environment,
    )
    pins = build_transport_executable_identity_pins(
        git_executable_path=git_executable,
        git_version_stdout=git_version,
        git_exec_path=git_exec_path,
        git_remote_https_executable_path=git_remote_https,
        credential_helper_executable_path=credential_helper,
        credential_helper_version_stdout=helper_version,
        command_interpreter_executable_path=command_interpreter,
    )
    return (
        pins,
        host,
        pins["transport_executable_closure_manifest_sha256"],
    )


class _PublicationWorkerHandle:
    __slots__ = (
        "_active",
        "_child_environment",
        "_consumed_push_authorizations",
        "_context",
        "_kernel",
        "_ownership",
        "_pins",
        "_runtime",
        "_sentinel",
        "_source_objects",
        "_transport_root",
    )

    def __init__(
        self,
        *,
        runtime: "_ProductionPublicationRuntime",
        kernel: Any,
        transport_root: Path,
        source_object_directory: Path,
        executable_pins: Mapping[str, Any],
        child_environment: Mapping[str, str],
        context: Mapping[str, Any],
        ownership: Mapping[str, Any],
        _sentinel: object,
    ) -> None:
        if _sentinel is not _PUBLICATION_HANDLE_SENTINEL:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication worker handles can only be issued by the runtime"
            )
        self._runtime = runtime
        self._kernel = kernel
        self._transport_root = transport_root
        self._source_objects = source_object_directory
        self._pins = copy.deepcopy(dict(executable_pins))
        self._child_environment = dict(child_environment)
        self._consumed_push_authorizations: set[str] = set()
        self._context = copy.deepcopy(dict(context))
        self._ownership = copy.deepcopy(dict(ownership))
        self._active = True
        self._sentinel = _sentinel

    @property
    def transport_root(self) -> Path:
        return self._transport_root

    @property
    def source_object_directory(self) -> Path:
        return self._source_objects

    @property
    def executable_pins(self) -> dict[str, Any]:
        return copy.deepcopy(self._pins)

    @property
    def host_environment_values(self) -> dict[str, str]:
        return dict(self._runtime._host_environment)

    @property
    def ownership_material(self) -> dict[str, Any]:
        return copy.deepcopy(self._ownership)

    def verify_executable_pins(
        self,
        pins: Mapping[str, Any],
    ) -> None:
        if not self._active or dict(pins) != self._pins:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication executable pin authority changed"
            )
        self._runtime._pin_verifier(
            copy.deepcopy(self._pins),
            expected_dependency_closure_sha256=(
                self._runtime._dependency_closure_sha256
            ),
        )
        return None

    def _execute_authorized(self, request: Any) -> Any:
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            ALLOWED_REMOTE_URL,
            contained_publication_execution_request_material,
        )

        if not self._active or self._runtime._active_handle is not self:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication worker handle is no longer current"
            )
        try:
            material = contained_publication_execution_request_material(
                request
            )
            intent_authority = (
                self._runtime._store.publication_intent_authority(
                    self._context["attempt_id"]
                )
            )
            intent = intent_authority.material
            transport_authority = (
                self._runtime._store.transport_manifest_authority(
                    self._context["attempt_id"],
                    material["transport_manifest_sha256"],
                )
            )
            transport = transport_authority.material
            ownership_authority = (
                self._runtime._store
                .publication_worker_ownership_authority(
                    self._context["attempt_id"],
                    self._ownership["worker_ownership_sha256"],
                )
            )
            durable_ownership = ownership_authority.material
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication process lacks exact durable command authorities"
            ) from None
        if durable_ownership != self._ownership:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Durable publication worker ownership changed"
            )
        tag_ref = intent["tag_ref"]
        expected_object = intent["expected_tag_object_sha1"]
        expected_directory = self._transport_root / "transport.git"
        readback = (
            self._pins["git_executable_path"],
            "ls-remote",
            "--tags",
            ALLOWED_REMOTE_URL,
            tag_ref,
            f"{tag_ref}^{{}}",
        )
        push = (
            self._pins["git_executable_path"],
            "push",
            "--porcelain",
            "--no-verify",
            ALLOWED_REMOTE_URL,
            f"{expected_object}:{tag_ref}",
        )
        profile_kind = material["profile_kind"]
        expected_argv = readback if profile_kind == "readback" else push
        authorization_hash = material["pre_push_authorization_sha256"]
        if (
            profile_kind not in {"readback", "push"}
            or material["argv"] != expected_argv
            or material["cwd"] != expected_directory
            or not expected_directory.is_absolute()
            or dict(material["environment"]) != self._child_environment
            or type(material["timeout_seconds"]) not in {int, float}
            or not math.isfinite(float(material["timeout_seconds"]))
            or float(material["timeout_seconds"]) <= 0.0
            or float(material["timeout_seconds"]) > 300.0
            or material["publication_intent_sha256"]
            != intent["publication_intent_sha256"]
            or material["publication_intent_sha256"]
            != self._context["publication_intent_sha256"]
            or material["operation_kind"] != self._context["operation_kind"]
            or material["operation_sha256"]
            != self._context["operation_sha256"]
            or material["worker_ownership_sha256"]
            != self._ownership["worker_ownership_sha256"]
            or material["expected_tag_object_sha1"] != expected_object
            or material["tag_ref"] != tag_ref
            or transport["publication_intent_sha256"]
            != intent["publication_intent_sha256"]
            or transport["operation_kind"] != self._context["operation_kind"]
            or transport["operation_sha256"]
            != self._context["operation_sha256"]
            or transport["attempt_id"] != self._context["attempt_id"]
            or transport["isolated_transport_git_directory_manifest_sha256"]
            != material["transport_manifest_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication contained process request changed"
            )
        if profile_kind == "readback":
            if authorization_hash is not None:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication readback unexpectedly carries push authority"
                )
        else:
            try:
                authorization_authority = (
                    self._runtime._store.pre_push_authorization_authority(
                        self._context["attempt_id"],
                        authorization_hash,
                    )
                )
                authorization = authorization_authority.material
            except Exception:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication push lacks durable one-use authorization"
                ) from None
            if (
                authorization["publication_intent_sha256"]
                != intent["publication_intent_sha256"]
                or authorization["authorization_operation_kind"]
                != self._context["operation_kind"]
                or authorization["authorization_operation_sha256"]
                != self._context["operation_sha256"]
                or authorization["worker_ownership_sha256"]
                != self._ownership["worker_ownership_sha256"]
                or authorization["tag_ref"] != tag_ref
                or authorization["expected_tag_object_sha1"]
                != expected_object
                or authorization["authorization_status"]
                != "push_authorized_once"
                or authorization["push_command_limit"] != 1
                or authorization_hash
                in self._consumed_push_authorizations
            ):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Publication push authorization changed or was reused"
                )
            self._consumed_push_authorizations.add(authorization_hash)
        self.verify_executable_pins(self._pins)
        return self._kernel.execute(
            argv=material["argv"],
            cwd=material["cwd"],
            env=dict(material["environment"]),
            timeout_seconds=float(material["timeout_seconds"]),
        )

    def abort_uncommitted(self) -> None:
        if not self._active:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication worker handle was already released"
            )
        self._kernel.close_uncommitted()
        self._active = False
        self._runtime._release(self)
        return None

    def quiesce(
        self,
        *,
        worker_ownership_sha256: str,
    ) -> dict[str, Any]:
        expected = _sha256(
            worker_ownership_sha256,
            "publication worker ownership",
        )
        if (
            not self._active
            or expected
            != self._ownership["worker_ownership_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication quiescence lost its worker binding"
            )
        active_count, alive_count = self._kernel.quiesce()
        if active_count != 0 or alive_count != 0:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication worker containment did not become quiescent"
            )
        self._active = False
        self._runtime._release(self)
        body = {
            "schema_version": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-quiescence-v1"
            ),
            "quiescence_verifier_id": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-quiescence-verifier-v1"
            ),
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "implementation_manifest_sha256": self._context[
                "implementation_manifest_sha256"
            ],
            "implementation_commit": self._context[
                "implementation_commit"
            ],
            "store_instance_id": self._context["store_instance_id"],
            "store_session_nonce_sha256": self._context[
                "store_session_nonce_sha256"
            ],
            "attempt_id": self._context["attempt_id"],
            "publication_intent_sha256": self._context[
                "publication_intent_sha256"
            ],
            "worker_ownership_sha256": expected,
            "verification_mode": "same_session_clean_release",
            "prior_owner_process_dead": True,
            "owner_mutex_unowned": True,
            "job_object_active_process_count": 0,
            "recorded_git_ssh_processes_alive_count": 0,
            "quiescence_status": "verified_no_live_owner_or_worker",
        }
        return {
            **body,
            "worker_quiescence_sha256": canonical_sha256(body),
        }


class _ProductionPublicationRuntime:
    """Source-bound publication supervisor and contained process executor."""

    __slots__ = (
        "_active_handle",
        "_child_environment",
        "_dependency_closure_sha256",
        "_host_environment",
        "_implementation",
        "_kernel_factory",
        "_pin_verifier",
        "_pins",
        "_repo_root",
        "_restart_verifier",
        "_runtime_root",
        "_sentinel",
        "_store",
        "_token_bytes",
    )

    def __init__(
        self,
        *,
        repo_root: Path,
        implementation_manifest: Mapping[str, Any],
        store: Any,
        executable_pins: Mapping[str, Any],
        dependency_closure_sha256: str,
        host_environment_values: Mapping[str, str],
        pin_verifier: Callable[..., None],
        kernel_factory: Callable[..., Any],
        restart_verifier: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ],
        token_bytes: Callable[[int], bytes],
        _sentinel: object,
    ) -> None:
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            build_exact_child_environment,
        )

        if (
            _sentinel is not _PUBLICATION_RUNTIME_SENTINEL
            or not isinstance(repo_root, Path)
            or not repo_root.is_absolute()
            or not callable(pin_verifier)
            or not callable(kernel_factory)
            or not callable(restart_verifier)
            or not callable(token_bytes)
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication runtime requires exact authorities"
            )
        implementation = copy.deepcopy(dict(implementation_manifest))
        for field, length in (
            ("implementation_manifest_sha256", 64),
            ("implementation_commit", 40),
        ):
            value = implementation.get(field)
            if (
                type(value) is not str
                or len(value) != length
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
            ):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Production publication implementation identity is invalid"
                )
        self._repo_root = repo_root
        self._implementation = implementation
        self._store = store
        self._pins = copy.deepcopy(dict(executable_pins))
        self._dependency_closure_sha256 = _sha256(
            dependency_closure_sha256,
            "publication Git dependency closure",
        )
        self._host_environment = dict(host_environment_values)
        self._pin_verifier = pin_verifier
        self._kernel_factory = kernel_factory
        self._restart_verifier = restart_verifier
        self._token_bytes = token_bytes
        self._child_environment = build_exact_child_environment(
            host_values=self._host_environment,
            executable_pins=self._pins,
        )
        self._runtime_root = (
            repo_root
            / ".git"
            / "sec-gemma-online-risk-overlay-v2-2-publication-workers"
        )
        try:
            self._runtime_root.mkdir(exist_ok=True)
            source_objects = (repo_root / ".git" / "objects").resolve(
                strict=True
            )
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication runtime paths are unavailable"
            ) from exc
        if not source_objects.is_dir():
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication source object directory is unavailable"
            )
        self._active_handle: _PublicationWorkerHandle | None = None
        self._sentinel = _sentinel

    def reverify(self) -> None:
        if self._active_handle is not None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication runtime still has an active worker"
            )
        self._pin_verifier(
            copy.deepcopy(self._pins),
            expected_dependency_closure_sha256=(
                self._dependency_closure_sha256
            ),
        )
        expected = _publication_host_environment(
            git_executable=Path(self._pins["git_executable_path"])
        )
        if expected != self._host_environment:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication host environment changed"
            )
        return None

    def begin(
        self,
        *,
        implementation_manifest: Mapping[str, Any],
        store_instance_id: str,
        store_session_nonce_sha256: str,
        attempt_id: str,
        publication_intent_sha256: str,
        operation_kind: str,
        operation_sha256: str,
    ) -> _PublicationWorkerHandle:
        if (
            self._active_handle is not None
            or dict(implementation_manifest) != self._implementation
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication runtime is busy or source identity changed"
            )
        store_id = _sha256(store_instance_id, "publication store instance")
        session = _sha256(
            store_session_nonce_sha256,
            "publication store session",
        )
        intent = _sha256(
            publication_intent_sha256,
            "publication intent",
        )
        operation = _sha256(
            operation_sha256,
            "publication operation",
        )
        if (
            type(attempt_id) is not str
            or not attempt_id
            or operation_kind
            not in {"normal_publication", "publication_recovery"}
            or (
                operation_kind == "normal_publication"
                and operation
                != PUBLICATION_NORMAL_OPERATION_SHA256
            )
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication operation identity changed"
            )
        self._pin_verifier(
            copy.deepcopy(self._pins),
            expected_dependency_closure_sha256=(
                self._dependency_closure_sha256
            ),
        )
        nonce_bytes = self._token_bytes(32)
        if type(nonce_bytes) is not bytes or len(nonce_bytes) != 32:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication owner nonce source changed"
            )
        owner_nonce = hashlib.sha256(nonce_bytes).hexdigest()
        job_name = _publication_object_name("job", owner_nonce)
        mutex_name = _publication_object_name("mutex", owner_nonce)
        kernel = self._kernel_factory(
            job_name=job_name,
            mutex_name=mutex_name,
        )
        try:
            owner_pid, owner_creation = (
                _publication_owner_process_identity()
            )
        except BaseException:
            try:
                kernel.close_uncommitted()
            except BaseException:
                pass
            raise
        context = {
            "implementation_manifest_sha256": self._implementation[
                "implementation_manifest_sha256"
            ],
            "implementation_commit": self._implementation[
                "implementation_commit"
            ],
            "store_instance_id": store_id,
            "store_session_nonce_sha256": session,
            "attempt_id": attempt_id,
            "publication_intent_sha256": intent,
            "operation_kind": operation_kind,
            "operation_sha256": operation,
        }
        body = {
            "schema_version": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-ownership-v1"
            ),
            "owner_verifier_id": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-owner-verifier-v1"
            ),
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            **context,
            "owner_nonce_sha256": owner_nonce,
            "owner_process_id": owner_pid,
            "owner_process_creation_filetime_hex": owner_creation,
            "job_object_name_sha256": (
                _publication_object_name_sha256(job_name)
            ),
            "owner_mutex_name_sha256": (
                _publication_object_name_sha256(mutex_name)
            ),
            "kill_on_parent_exit": True,
            "child_assignment_before_resume_required": True,
            "ownership_status": "claimed",
        }
        ownership = {
            **body,
            "worker_ownership_sha256": canonical_sha256(body),
        }
        binding = canonical_sha256(
            {
                "attempt_id": attempt_id,
                "publication_intent_sha256": intent,
                "operation_sha256": operation,
                "owner_nonce_sha256": owner_nonce,
            }
        )
        transport_root = self._runtime_root / binding
        try:
            handle = _PublicationWorkerHandle(
                runtime=self,
                kernel=kernel,
                transport_root=transport_root,
                source_object_directory=(
                    self._repo_root / ".git" / "objects"
                ).resolve(strict=True),
                executable_pins=self._pins,
                child_environment=self._child_environment,
                context=context,
                ownership=ownership,
                _sentinel=_PUBLICATION_HANDLE_SENTINEL,
            )
        except BaseException:
            try:
                kernel.close_uncommitted()
            except BaseException:
                pass
            raise
        self._active_handle = handle
        return handle

    def _release(self, handle: _PublicationWorkerHandle) -> None:
        if self._active_handle is not handle:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication runtime released a foreign worker"
            )
        self._active_handle = None

    def __call__(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> Any:
        del argv, cwd, env, timeout_seconds
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Raw publication process execution is forbidden"
        )

    def execute_authorized(self, request: Any) -> Any:
        if self._active_handle is None:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication process lacks an active worker"
            )
        return self._active_handle._execute_authorized(request)

    def restarted_quiescence_material(
        self,
        *,
        ownership: Mapping[str, Any],
        current_store_session_nonce_sha256: str,
    ) -> dict[str, Any]:
        owner = copy.deepcopy(dict(ownership))
        if (
            tuple(owner) != PUBLICATION_WORKER_OWNERSHIP_FIELDS
            or owner.get("schema_version")
            != (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-ownership-v1"
            )
            or owner.get("owner_verifier_id")
            != (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-owner-verifier-v1"
            )
            or owner.get("contract_version") != CONTRACT_VERSION
            or owner.get("contract_sha256") != CONTRACT_SHA256
            or owner.get("implementation_manifest_sha256")
            != self._implementation[
                "implementation_manifest_sha256"
            ]
            or owner.get("implementation_commit")
            != self._implementation["implementation_commit"]
            or owner.get("worker_ownership_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in owner.items()
                    if key != "worker_ownership_sha256"
                }
            )
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Restarted publication ownership authority changed"
            )
        observed = dict(self._restart_verifier(owner))
        if observed != {
            "prior_owner_process_dead": True,
            "owner_mutex_unowned": True,
            "job_object_active_process_count": 0,
            "recorded_git_ssh_processes_alive_count": 0,
        }:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Restarted publication owner is not exactly quiescent"
            )
        session = _sha256(
            current_store_session_nonce_sha256,
            "restarted publication store session",
        )
        body = {
            "schema_version": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-quiescence-v1"
            ),
            "quiescence_verifier_id": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-worker-quiescence-verifier-v1"
            ),
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "implementation_manifest_sha256": owner[
                "implementation_manifest_sha256"
            ],
            "implementation_commit": owner[
                "implementation_commit"
            ],
            "store_instance_id": owner["store_instance_id"],
            "store_session_nonce_sha256": session,
            "attempt_id": owner["attempt_id"],
            "publication_intent_sha256": owner[
                "publication_intent_sha256"
            ],
            "worker_ownership_sha256": owner[
                "worker_ownership_sha256"
            ],
            "verification_mode": "post_restart_prior_owner_dead",
            **observed,
            "quiescence_status": "verified_no_live_owner_or_worker",
        }
        return {
            **body,
            "worker_quiescence_sha256": canonical_sha256(body),
        }

    def __repr__(self) -> str:
        return "_ProductionPublicationRuntime(<kill-on-close, source-bound>)"


class VerifiedProductionAuthorities:
    """Opaque identity-equal production composition for the runner."""

    __slots__ = (
        "_acquisition_adapter",
        "_authority",
        "_final_registry_authority",
        "_implementation",
        "_phase_executor",
        "_private_identity",
        "_publication_runtime",
        "_repo_root",
        "_report_publisher",
        "_sentinel",
        "_store",
        "_vault",
    )

    def __init__(
        self,
        *,
        repo_root: Path,
        implementation_manifest: Mapping[str, Any],
        store: Any,
        authority: VerifiedProductionAuthority,
        private_identity: _PrivateSecUserAgent,
        vault: Any,
        phase_executor: _ProductionPhaseExecutor,
        acquisition_adapter: _ProductionAcquisitionAdapter,
        report_publisher: Any,
        publication_runtime: _ProductionPublicationRuntime,
        final_registry_authority: Any,
        _sentinel: object,
    ) -> None:
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            ExternalGitTagPublisher,
            verify_pinned_transport_executables,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_registry import (
            FinalRegistryAuthorizer,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_store import (
            SecGemmaOnlineRiskOverlayStore,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
            ProductionAcquisitionVault,
        )

        if (
            _sentinel is not _AUTHORITIES_SENTINEL
            or type(store) is not SecGemmaOnlineRiskOverlayStore
            or not is_verified_production_authority(authority)
            or type(private_identity) is not _PrivateSecUserAgent
            or type(vault) is not ProductionAcquisitionVault
            or type(phase_executor) is not _ProductionPhaseExecutor
            or type(acquisition_adapter)
            is not _ProductionAcquisitionAdapter
            or type(report_publisher) is not ExternalGitTagPublisher
            or type(publication_runtime)
            is not _ProductionPublicationRuntime
            or type(final_registry_authority)
            is not FinalRegistryAuthorizer
            or acquisition_adapter._store is not store
            or acquisition_adapter._vault is not vault
            or final_registry_authority._publisher
            is not report_publisher
            or final_registry_authority._store is not store
            or report_publisher._executor is not publication_runtime
            or type(report_publisher._final_subprocess_runner)
            is not _WindowsFinalRegistrySubprocessRunner
            or report_publisher._final_subprocess_runner._token_bytes
            is not secrets.token_bytes
            or report_publisher._final_allow_url_rewrite is not False
            or report_publisher._final_test_url_rewrite_target is not None
            or report_publisher._final_test_url_rewrite_target_identity_sha256
            is not None
            or publication_runtime._store is not store
            or publication_runtime._pin_verifier
            is not verify_pinned_transport_executables
            or publication_runtime._kernel_factory
            is not _WindowsPublicationKernel
            or publication_runtime._restart_verifier
            is not _verify_prior_publication_owner_quiescence
            or publication_runtime._token_bytes is not secrets.token_bytes
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production authority bundle requires exact identity-equal components"
            )
        self._repo_root = repo_root
        self._implementation = copy.deepcopy(
            dict(implementation_manifest)
        )
        self._store = store
        self._authority = authority
        self._private_identity = private_identity
        self._vault = vault
        self._phase_executor = phase_executor
        self._acquisition_adapter = acquisition_adapter
        self._report_publisher = report_publisher
        self._publication_runtime = publication_runtime
        self._final_registry_authority = final_registry_authority
        self._sentinel = _sentinel

    @property
    def phase_executor(self) -> _ProductionPhaseExecutor:
        return self._phase_executor

    @property
    def acquisition_adapter(self) -> _ProductionAcquisitionAdapter:
        return self._acquisition_adapter

    @property
    def report_publisher(self) -> Any:
        return self._report_publisher

    @property
    def publication_runtime(self) -> _ProductionPublicationRuntime:
        return self._publication_runtime

    @property
    def final_registry_authority(self) -> Any:
        return self._final_registry_authority

    @property
    def store(self) -> Any:
        return self._store

    def rehydrate_pending_acquisition_report(
        self,
        *,
        attempt_id: str,
    ) -> Any:
        return self._acquisition_adapter.rehydrate_pending_acquisition_report(
            attempt_id=attempt_id
        )

    def assert_ready_for_command(self, command: str) -> None:
        """Fail before one-shot registration when opaque state was not retained."""

        self._phase_executor._ledger.assert_ready_for_command(
            command,
            store=self._store,
        )
        return None

    def reverify(self) -> None:
        from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
            build_implementation_manifest,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            ExternalGitTagPublisher,
            verify_pinned_transport_executables,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_registry import (
            FinalRegistryAuthorizer,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
            verify_live_source_tree,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_store import (
            SecGemmaOnlineRiskOverlayStore,
        )
        from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
            ProductionAcquisitionVault,
        )

        if (
            type(self._store) is not SecGemmaOnlineRiskOverlayStore
            or type(self._vault) is not ProductionAcquisitionVault
            or type(self._phase_executor)
            is not _ProductionPhaseExecutor
            or type(self._acquisition_adapter)
            is not _ProductionAcquisitionAdapter
            or type(self._report_publisher)
            is not ExternalGitTagPublisher
            or type(self._publication_runtime)
            is not _ProductionPublicationRuntime
            or type(self._final_registry_authority)
            is not FinalRegistryAuthorizer
            or self._acquisition_adapter._store is not self._store
            or self._acquisition_adapter._vault is not self._vault
            or self._acquisition_adapter._authority
            is not self._authority
            or self._acquisition_adapter._private_identity
            is not self._private_identity
            or self._acquisition_adapter._ledger
            is not self._phase_executor._ledger
            or self._acquisition_adapter._clock is not time.monotonic
            or self._acquisition_adapter._sleeper is not time.sleep
            or self._phase_executor._clock is not time.monotonic
            or self._report_publisher._final_clock is not time.monotonic
            or type(self._report_publisher._final_subprocess_runner)
            is not _WindowsFinalRegistrySubprocessRunner
            or self._report_publisher._final_subprocess_runner._token_bytes
            is not secrets.token_bytes
            or self._report_publisher._final_allow_url_rewrite is not False
            or self._report_publisher._final_test_url_rewrite_target is not None
            or self._report_publisher._final_test_url_rewrite_target_identity_sha256
            is not None
            or self._final_registry_authority._clock
            is not time.monotonic
            or self._final_registry_authority._publisher
            is not self._report_publisher
            or self._final_registry_authority._store is not self._store
            or self._report_publisher._executor
            is not self._publication_runtime
            or self._publication_runtime._pin_verifier
            is not verify_pinned_transport_executables
            or self._publication_runtime._store is not self._store
            or self._publication_runtime._kernel_factory
            is not _WindowsPublicationKernel
            or self._publication_runtime._restart_verifier
            is not _verify_prior_publication_owner_quiescence
            or self._publication_runtime._token_bytes
            is not secrets.token_bytes
            or self._acquisition_adapter._repo_root != self._repo_root
            or self._authority.repo_root != self._repo_root
            or self._publication_runtime._repo_root != self._repo_root
            or self._report_publisher._final_repo_root != self._repo_root
            or self._final_registry_authority._repo_root
            != self._repo_root
            or self._publication_runtime._implementation
            != self._implementation
            or self._report_publisher._final_implementation
            != self._implementation
            or getattr(self._vault, "_production_authority", None)
            is not True
            or getattr(self._vault, "_bound_store_instance_id", None)
            != getattr(self._store, "_store_instance_id", None)
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production authority component identity changed"
            )
        self._publication_runtime.reverify()
        try:
            self._report_publisher._verify_final_registry_mode()
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production final-registry publisher identity changed"
            ) from None
        try:
            verified = verify_live_source_tree(self._repo_root)
            live_implementation = build_implementation_manifest(
                verified_sources=verified
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Live source verification failed for production authorities"
            ) from None
        if live_implementation != self._implementation:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Live implementation differs from production authorities"
            )
        refreshed = issue_verified_production_authority(verified)
        if (
            refreshed.authority_sha256
            != self._authority.authority_sha256
            or refreshed.head_commit != self._authority.head_commit
            or refreshed.repo_root != self._repo_root
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production source authority changed"
            )
        if (
            getattr(self._store, "_repo_root", None)
            != self._repo_root
            or getattr(self._store, "_implementation_manifest", None)
            != self._implementation
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production store binding changed"
            )
        try:
            snapshot = self._store.snapshot()
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production store integrity verification failed"
            ) from None
        if (
            type(snapshot) is not dict
            or snapshot.get("chain_valid") is not True
            or snapshot.get("anchor_valid") is not True
            or snapshot.get("implementation_manifest_sha256")
            != self._implementation[
                "implementation_manifest_sha256"
            ]
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production store is not bound to the live implementation"
            )
        self._private_identity._reverify()
        return None

    def __repr__(self) -> str:
        return "VerifiedProductionAuthorities(<source/store-bound, redacted>)"


def is_verified_production_authorities(value: Any) -> bool:
    return (
        type(value) is VerifiedProductionAuthorities
        and getattr(value, "_sentinel", None) is _AUTHORITIES_SENTINEL
    )


class VerifiedSupervisedPublicationRecoveryWorkerAuthority:
    """Opaque one-use proof that recovery runs in the exact outer Job child."""

    __slots__ = (
        "_attempt_id",
        "_implementation",
        "_invocation_nonce_sha256",
        "_job_name",
        "_production_authorities",
        "_repo_root",
        "_sentinel",
        "_supervisor_deadline",
        "_supervisor_process_creation_filetime_hex",
        "_supervisor_process_id",
        "_supervisor_started_at",
        "_used",
        "_worker_process_creation_filetime_hex",
        "_worker_process_id",
    )

    def __init__(
        self,
        *,
        repo_root: Path,
        implementation_manifest: Mapping[str, Any],
        production_authorities: VerifiedProductionAuthorities,
        attempt_id: str,
        invocation_nonce_sha256: str,
        job_name: str,
        supervisor_started_at: float,
        supervisor_deadline: float,
        supervisor_process_id: int,
        supervisor_process_creation_filetime_hex: str,
        worker_process_id: int,
        worker_process_creation_filetime_hex: str,
        _sentinel: object,
    ) -> None:
        if (
            _sentinel is not _RECOVERY_WORKER_AUTHORITY_SENTINEL
            or not is_verified_production_authorities(
                production_authorities
            )
            or not isinstance(repo_root, Path)
            or not repo_root.is_absolute()
            or attempt_id not in _RECOVERY_COMMAND_BY_ATTEMPT_ID
            or type(invocation_nonce_sha256) is not str
            or _SHA256_RE.fullmatch(invocation_nonce_sha256) is None
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Supervised publication-recovery authority cannot be forged"
            )
        self._repo_root = repo_root
        self._implementation = copy.deepcopy(
            dict(implementation_manifest)
        )
        self._production_authorities = production_authorities
        self._attempt_id = attempt_id
        self._invocation_nonce_sha256 = invocation_nonce_sha256
        self._job_name = job_name
        self._supervisor_started_at = supervisor_started_at
        self._supervisor_deadline = supervisor_deadline
        self._supervisor_process_id = supervisor_process_id
        self._supervisor_process_creation_filetime_hex = (
            supervisor_process_creation_filetime_hex
        )
        self._worker_process_id = worker_process_id
        self._worker_process_creation_filetime_hex = (
            worker_process_creation_filetime_hex
        )
        self._used = False
        self._sentinel = _sentinel

    def authorize_runner_recovery(
        self,
        *,
        repo_root: Path,
        implementation_manifest: Mapping[str, Any],
        production_authorities: Any,
        attempt_id: str,
        worker_entry_monotonic: float,
        worker_deadline_monotonic: float,
    ) -> tuple[float, float]:
        try:
            current_process_id, current_creation = (
                _publication_owner_process_identity()
            )
            now = float(time.monotonic())
            worker_entry = float(worker_entry_monotonic)
            worker_deadline = float(worker_deadline_monotonic)
            supervisor_deadline = float(self._supervisor_deadline)
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Supervised publication-recovery authority could not "
                "reverify its process or clock"
            ) from None
        if (
            self._used
            or production_authorities is not self._production_authorities
            or not is_verified_production_authorities(
                production_authorities
            )
            or repo_root != self._repo_root
            or dict(implementation_manifest) != self._implementation
            or attempt_id != self._attempt_id
            or current_process_id != self._worker_process_id
            or current_creation
            != self._worker_process_creation_filetime_hex
            or not math.isfinite(worker_entry)
            or not math.isfinite(worker_deadline)
            or worker_deadline
            != worker_entry + MAX_PUBLICATION_RECOVERY_SECONDS
            or supervisor_deadline
            != (
                self._supervisor_started_at
                + MAX_PUBLICATION_RECOVERY_SECONDS
            )
            or not worker_entry < supervisor_deadline <= worker_deadline
            or not now < supervisor_deadline
            or not _current_process_is_in_recovery_job(
                self._job_name
            )
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Supervised publication-recovery authority is foreign, "
                "expired, reused, or outside its Job"
            )
        try:
            supervisor_dead = _windows_prior_process_is_dead(
                self._supervisor_process_id,
                self._supervisor_process_creation_filetime_hex,
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervisor identity cannot be verified"
            ) from None
        if supervisor_dead:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervisor is no longer alive"
            )
        self._used = True
        return self._supervisor_started_at, supervisor_deadline

    def __repr__(self) -> str:
        return (
            "VerifiedSupervisedPublicationRecoveryWorkerAuthority("
            "<source/process/deadline-bound>)"
        )


def is_verified_supervised_publication_recovery_worker_authority(
    value: Any,
) -> bool:
    return (
        type(value)
        is VerifiedSupervisedPublicationRecoveryWorkerAuthority
        and getattr(value, "_sentinel", None)
        is _RECOVERY_WORKER_AUTHORITY_SENTINEL
    )


def verify_production_authorities(
    *,
    repo_root: Path,
    implementation_manifest: Mapping[str, Any],
    store: Any,
    sec_user_agent: str,
    clock: Callable[[], float] = time.monotonic,
) -> VerifiedProductionAuthorities:
    """Compose exact source-, store-, transport-, and publisher-bound authority."""

    from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
        build_implementation_manifest,
        validate_implementation_manifest,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
        ExternalGitTagPublisher,
        verify_pinned_transport_executables,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_registry import (
        FinalRegistryAuthorizer,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
        verify_live_source_tree,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        SecGemmaOnlineRiskOverlayStore,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        open_production_acquisition_vault,
    )

    if clock is not time.monotonic:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production authorities require the exact monotonic clock"
        )
    if type(store) is not SecGemmaOnlineRiskOverlayStore:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production authorities require the exact durable store"
        )
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    try:
        verified = verify_live_source_tree(repo_root)
        live_implementation = build_implementation_manifest(
            verified_sources=verified
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production live source verification failed"
        ) from None
    if live_implementation != implementation:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production manifest differs from live source verification"
        )
    root = verified.repo_root
    if (
        getattr(store, "_repo_root", None) != root
        or getattr(store, "_implementation_manifest", None)
        != implementation
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production store differs from the live manifest"
        )
    authority = issue_verified_production_authority(verified)
    private_identity = _PrivateSecUserAgent(sec_user_agent)
    vault = open_production_acquisition_vault(
        repo_root=root,
        store=store,
    )
    try:
        ledger = _rehydrate_production_acquisition_ledger(
            vault=vault,
            store=store,
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production acquisition restart recovery failed closed"
        ) from None
    adapter = _ProductionAcquisitionAdapter(
        authority=authority,
        repo_root=root,
        store=store,
        vault=vault,
        private_identity=private_identity,
        ledger=ledger,
        clock=time.monotonic,
        sleeper=time.sleep,
        _sentinel=_ADAPTER_SENTINEL,
    )
    executor = _ProductionPhaseExecutor(
        ledger=ledger,
        clock=time.monotonic,
        _sentinel=_EXECUTOR_SENTINEL,
    )
    executable_pins, host_environment, dependency_closure = (
        _discover_publication_runtime_material()
    )
    publication_runtime = _ProductionPublicationRuntime(
        repo_root=root,
        implementation_manifest=implementation,
        store=store,
        executable_pins=executable_pins,
        dependency_closure_sha256=dependency_closure,
        host_environment_values=host_environment,
        pin_verifier=verify_pinned_transport_executables,
        kernel_factory=_WindowsPublicationKernel,
        restart_verifier=(
            _verify_prior_publication_owner_quiescence
        ),
        token_bytes=secrets.token_bytes,
        _sentinel=_PUBLICATION_RUNTIME_SENTINEL,
    )
    for attempt_id in (
        DEVELOPMENT_ACQUISITION_ID,
        DEVELOPMENT_ATTEMPT_ID,
        CONFIRMATION_ATTEMPT_ID,
        FINAL_ATTEMPT_ID,
    ):
        try:
            unresolved = (
                store.unresolved_publication_worker_ownership_authorities(
                    attempt_id
                )
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production publication ownership recovery failed closed"
            ) from None
        for ownership in unresolved:
            try:
                material = publication_runtime.restarted_quiescence_material(
                    ownership=ownership.material,
                    current_store_session_nonce_sha256=(
                        store.store_session_nonce_sha256
                    ),
                )
                store.commit_restarted_worker_quiescence(
                    ownership,
                    material,
                )
            except Exception:
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Production publication ownership is not restart-safe"
                ) from None
    publisher = ExternalGitTagPublisher(
        process_executor=publication_runtime,
        repo_root=root,
        implementation_manifest=implementation,
        clock=time.monotonic,
        final_registry_subprocess_runner=(
            _WindowsFinalRegistrySubprocessRunner(
                token_bytes=secrets.token_bytes
            )
        ),
    )
    registry = FinalRegistryAuthorizer(
        repo_root=root,
        publisher=publisher,
        store=store,
        clock=time.monotonic,
    )
    bundle = VerifiedProductionAuthorities(
        repo_root=root,
        implementation_manifest=implementation,
        store=store,
        authority=authority,
        private_identity=private_identity,
        vault=vault,
        phase_executor=executor,
        acquisition_adapter=adapter,
        report_publisher=publisher,
        publication_runtime=publication_runtime,
        final_registry_authority=registry,
        _sentinel=_AUTHORITIES_SENTINEL,
    )
    bundle.reverify()
    return bundle


def _build_publication_recovery_worker_request(
    *,
    repo_root: Path,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    sec_user_agent: str,
    supervisor_started_at: float,
    supervisor_deadline: float,
) -> dict[str, Any]:
    if (
        not isinstance(repo_root, Path)
        or not repo_root.is_absolute()
        or type(implementation_manifest) is not dict
        or attempt_id not in _RECOVERY_COMMAND_BY_ATTEMPT_ID
        or type(sec_user_agent) is not str
        or type(supervisor_started_at) is not float
        or type(supervisor_deadline) is not float
        or not math.isfinite(supervisor_started_at)
        or supervisor_started_at <= 0.0
        or supervisor_deadline
        != supervisor_started_at + MAX_PUBLICATION_RECOVERY_SECONDS
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production publication-recovery request is invalid"
        )
    try:
        nonce = secrets.token_bytes(32).hex()
        supervisor_process_id, supervisor_creation = (
            _publication_owner_process_identity()
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production publication-recovery supervisor identity failed"
        ) from None
    nonce_sha256 = canonical_sha256(
        {"publication_recovery_invocation_nonce": nonce}
    )
    job_name = (
        "Local\\CodexPublicationRecoveryJob-" + nonce_sha256
    )
    body = {
        "schema_version": (
            PRODUCTION_PUBLICATION_RECOVERY_REQUEST_SCHEMA_VERSION
        ),
        "worker_authority_schema_version": (
            SUPERVISED_PUBLICATION_RECOVERY_WORKER_AUTHORITY_SCHEMA_VERSION
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "repo_root": str(repo_root),
        "implementation_manifest": copy.deepcopy(
            dict(implementation_manifest)
        ),
        "attempt_id": attempt_id,
        "sec_user_agent": sec_user_agent,
        "recovery_host_environment": (
            _publication_recovery_host_environment()
        ),
        "invocation_nonce": nonce,
        "invocation_nonce_sha256": nonce_sha256,
        "supervisor_job_name": job_name,
        "supervisor_process_id": supervisor_process_id,
        "supervisor_process_creation_filetime_hex": supervisor_creation,
        "supervisor_started_monotonic_hex": (
            supervisor_started_at.hex()
        ),
        "supervisor_deadline_monotonic_hex": (
            supervisor_deadline.hex()
        ),
    }
    return {
        **body,
        "publication_recovery_request_sha256": canonical_sha256(body),
    }


def _validate_publication_recovery_worker_request(
    value: Mapping[str, Any],
    *,
    require_worker_containment: bool,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker request must be one exact mapping"
        )
    observed = copy.deepcopy(dict(value))
    expected_fields = {
        "schema_version",
        "worker_authority_schema_version",
        "contract_version",
        "contract_sha256",
        "repo_root",
        "implementation_manifest",
        "attempt_id",
        "sec_user_agent",
        "recovery_host_environment",
        "invocation_nonce",
        "invocation_nonce_sha256",
        "supervisor_job_name",
        "supervisor_process_id",
        "supervisor_process_creation_filetime_hex",
        "supervisor_started_monotonic_hex",
        "supervisor_deadline_monotonic_hex",
        "publication_recovery_request_sha256",
    }
    if set(observed) != expected_fields:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker request fields changed"
        )
    request_sha256 = observed.pop(
        "publication_recovery_request_sha256"
    )
    if (
        observed["schema_version"]
        != PRODUCTION_PUBLICATION_RECOVERY_REQUEST_SCHEMA_VERSION
        or observed["worker_authority_schema_version"]
        != (
            SUPERVISED_PUBLICATION_RECOVERY_WORKER_AUTHORITY_SCHEMA_VERSION
        )
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["attempt_id"]
        not in _RECOVERY_COMMAND_BY_ATTEMPT_ID
        or type(observed["implementation_manifest"]) is not dict
        or type(observed["sec_user_agent"]) is not str
        or type(observed["recovery_host_environment"]) is not dict
        or type(observed["invocation_nonce"]) is not str
        or _SHA256_RE.fullmatch(observed["invocation_nonce"]) is None
        or type(observed["invocation_nonce_sha256"]) is not str
        or _SHA256_RE.fullmatch(
            observed["invocation_nonce_sha256"]
        )
        is None
        or observed["invocation_nonce_sha256"]
        != canonical_sha256(
            {
                "publication_recovery_invocation_nonce": observed[
                    "invocation_nonce"
                ]
            }
        )
        or observed["supervisor_job_name"]
        != (
            "Local\\CodexPublicationRecoveryJob-"
            + observed["invocation_nonce_sha256"]
        )
        or type(observed["supervisor_process_id"]) is not int
        or observed["supervisor_process_id"] <= 0
        or type(
            observed[
                "supervisor_process_creation_filetime_hex"
            ]
        )
        is not str
        or type(request_sha256) is not str
        or request_sha256 != canonical_sha256(observed)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker request is not exact"
        )
    _validate_publication_recovery_host_environment(
        observed["recovery_host_environment"]
    )
    started = _positive_float_hex(
        observed["supervisor_started_monotonic_hex"],
        "Publication-recovery supervisor start",
    )
    deadline = _positive_float_hex(
        observed["supervisor_deadline_monotonic_hex"],
        "Publication-recovery supervisor deadline",
    )
    if (
        deadline != started + MAX_PUBLICATION_RECOVERY_SECONDS
        or float(time.monotonic()) >= deadline
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker request is expired"
        )
    try:
        root = Path(observed["repo_root"])
        expected_root = Path(__file__).resolve(strict=True).parents[1]
        if (
            type(observed["repo_root"]) is not str
            or not root.is_absolute()
            or root.resolve(strict=True) != root
            or root != expected_root
        ):
            raise ValueError
    except (OSError, ValueError, TypeError):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker source root changed"
        ) from None
    if require_worker_containment:
        if not _current_process_is_in_recovery_job(
            observed["supervisor_job_name"]
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery worker is outside its outer Job"
            )
        try:
            supervisor_dead = _windows_prior_process_is_dead(
                observed["supervisor_process_id"],
                observed[
                    "supervisor_process_creation_filetime_hex"
                ],
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervisor cannot be verified"
            ) from None
        if supervisor_dead:
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication-recovery supervisor is no longer alive"
            )
    return {
        **observed,
        "publication_recovery_request_sha256": request_sha256,
    }


def _issue_supervised_publication_recovery_worker_authority(
    *,
    request: Mapping[str, Any],
    production_authorities: VerifiedProductionAuthorities,
) -> VerifiedSupervisedPublicationRecoveryWorkerAuthority:
    observed = _validate_publication_recovery_worker_request(
        request,
        require_worker_containment=True,
    )
    if not is_verified_production_authorities(
        production_authorities
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker lacks production authorities"
        )
    worker_process_id, worker_creation = (
        _publication_owner_process_identity()
    )
    return VerifiedSupervisedPublicationRecoveryWorkerAuthority(
        repo_root=Path(observed["repo_root"]),
        implementation_manifest=observed[
            "implementation_manifest"
        ],
        production_authorities=production_authorities,
        attempt_id=observed["attempt_id"],
        invocation_nonce_sha256=observed[
            "invocation_nonce_sha256"
        ],
        job_name=observed["supervisor_job_name"],
        supervisor_started_at=float.fromhex(
            observed["supervisor_started_monotonic_hex"]
        ),
        supervisor_deadline=float.fromhex(
            observed["supervisor_deadline_monotonic_hex"]
        ),
        supervisor_process_id=observed[
            "supervisor_process_id"
        ],
        supervisor_process_creation_filetime_hex=observed[
            "supervisor_process_creation_filetime_hex"
        ],
        worker_process_id=worker_process_id,
        worker_process_creation_filetime_hex=worker_creation,
        _sentinel=_RECOVERY_WORKER_AUTHORITY_SENTINEL,
    )


def _validate_publication_recovery_output(
    value: Mapping[str, Any],
    *,
    attempt_id: str,
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or attempt_id not in _RECOVERY_COMMAND_BY_ATTEMPT_ID
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery output is not one exact mapping"
        )
    observed = copy.deepcopy(dict(value))
    command = _RECOVERY_COMMAND_BY_ATTEMPT_ID[attempt_id]
    if "sealed_stage_result_sha256" in observed:
        expected_fields = {
            "schema_version",
            "contract_version",
            "contract_sha256",
            "command",
            "attempt_id",
            "terminal_status",
            "passed",
            "diagnostic_code",
            "terminal_transition",
            "terminal_transition_sha256",
            "terminal_evidence",
            "external_publication",
            "terminal_artifact",
            "terminal_artifact_receipt",
            "sealed_stage_result_sha256",
        }
        sealed_sha256 = observed.pop(
            "sealed_stage_result_sha256"
        )
        if (
            set(value) != expected_fields
            or observed["schema_version"]
            != (
                "aapl-sec-gemma-online-risk-overlay-v2-2-"
                "sealed-stage-result-v1"
            )
            or observed["contract_version"] != CONTRACT_VERSION
            or observed["contract_sha256"] != CONTRACT_SHA256
            or observed["command"] != command
            or observed["attempt_id"] != attempt_id
            or observed["terminal_status"]
            not in {"terminal_pass", "terminal_fail"}
            or type(observed["passed"]) is not bool
            or (
                observed["terminal_status"] == "terminal_pass"
            )
            is not observed["passed"]
            or type(sealed_sha256) is not str
            or sealed_sha256 != canonical_sha256(observed)
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Supervised recovery returned an invalid sealed result"
            )
        return {
            **observed,
            "sealed_stage_result_sha256": sealed_sha256,
        }
    expected_fields = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "command",
        "attempt_id",
        "result_status",
        "publication_intent_sha256",
        "publication_intent_store_receipt_sha256",
        "semantic_result_released",
        "next_stage_authority_blocked",
        "publication_recovery_required",
        "external_cost_usd",
        "publication_pending_result_sha256",
    }
    pending_sha256 = observed.pop(
        "publication_pending_result_sha256",
        None,
    )
    if (
        set(value) != expected_fields
        or observed["schema_version"]
        != (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "publication-pending-result-v1"
        )
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["command"] != command
        or observed["attempt_id"] != attempt_id
        or observed["result_status"] != "publication_pending"
        or observed["semantic_result_released"] is not False
        or observed["next_stage_authority_blocked"] is not True
        or observed["publication_recovery_required"] is not True
        or observed["external_cost_usd"] != 0
        or type(pending_sha256) is not str
        or pending_sha256 != canonical_sha256(observed)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Supervised recovery returned an invalid pending result"
        )
    return {
        **observed,
        "publication_pending_result_sha256": pending_sha256,
    }


def _publication_recovery_worker_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _validate_publication_recovery_worker_request(
        request,
        require_worker_containment=True,
    )
    _install_publication_recovery_host_environment(
        observed["recovery_host_environment"]
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
        validate_implementation_manifest,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_runner import (
        SecGemmaOnlineRiskOverlayRunner,
        validate_sealed_stage_result,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        SecGemmaOnlineRiskOverlayStore,
        publication_intent_material,
    )

    implementation = validate_implementation_manifest(
        observed["implementation_manifest"]
    )
    root = Path(observed["repo_root"])
    attempt_id = observed["attempt_id"]
    with SecGemmaOnlineRiskOverlayStore(
        root,
        implementation_manifest=implementation,
    ) as store:
        authorities = verify_production_authorities(
            repo_root=root,
            implementation_manifest=implementation,
            store=store,
            sec_user_agent=observed["sec_user_agent"],
            clock=time.monotonic,
        )
        worker_authority = (
            _issue_supervised_publication_recovery_worker_authority(
                request=observed,
                production_authorities=authorities,
            )
        )
        runner = SecGemmaOnlineRiskOverlayRunner(
            repo_root=root,
            implementation_manifest=implementation,
            store=store,
            phase_executor=authorities.phase_executor,
            acquisition_adapter=authorities.acquisition_adapter,
            report_publisher=authorities.report_publisher,
            publication_runtime=authorities.publication_runtime,
            final_registry_authority=(
                authorities.final_registry_authority
            ),
            production_authorities=authorities,
            clock=time.monotonic,
            test_only_allow_effects=False,
        )
        result = runner.recover_publication(
            attempt_id,
            supervised_worker_authority=worker_authority,
        )
        result = _validate_publication_recovery_output(
            result,
            attempt_id=attempt_id,
        )
        if "sealed_stage_result_sha256" in result:
            validate_sealed_stage_result(
                result,
                implementation_manifest=implementation,
                attempt_plan=store.attempt_plan(attempt_id),
            )
        else:
            intent = publication_intent_material(
                store.publication_intent_authority(attempt_id)
            )
            if (
                result["publication_intent_sha256"]
                != intent["publication_intent_sha256"]
            ):
                raise SecGemmaOnlineRiskOverlayProductionError(
                    "Pending recovery output crossed its durable intent"
                )
        if float(time.monotonic()) >= float.fromhex(
            observed["supervisor_deadline_monotonic_hex"]
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Publication recovery exceeded its 300-second deadline"
            )
        return result


def _decode_publication_recovery_worker_output(
    payload: bytes,
    *,
    attempt_id: str,
) -> dict[str, Any]:
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > _STDIO_WORKER_MAX_RESPONSE_BYTES
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker output is absent or oversized"
        )
    try:
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker output is invalid"
        ) from None
    if canonical_json_bytes(decoded) != payload:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker output is not canonical JSON"
        )
    if (
        type(decoded) is not dict
        or set(decoded) != {"payload", "status"}
        or decoded["status"] != "ok"
        or type(decoded["payload"]) is not dict
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker failed closed"
        )
    return _validate_publication_recovery_output(
        decoded["payload"],
        attempt_id=attempt_id,
    )


def _supervise_publication_recovery_worker(
    request: Mapping[str, Any],
    *,
    supervisor_deadline: float,
    supervisor_factory: Callable[..., Any] = (
        _WindowsPublicationRecoverySupervisor
    ),
) -> dict[str, Any]:
    observed = _validate_publication_recovery_worker_request(
        request,
        require_worker_containment=False,
    )
    try:
        encoded_request = canonical_json_bytes(observed)
        root = Path(observed["repo_root"])
        bootstrap_material = canonical_json_bytes(
            _worker_bootstrap_material()
        ).decode("ascii", errors="strict")
        command = (
            str(Path(sys.executable).resolve(strict=True)),
            "-I",
            "-S",
            "-c",
            _WORKER_BOOTSTRAP,
            str(root),
            bootstrap_material,
            _WORKER_MODULE,
            "--publication-recovery-worker",
        )
        environment = _sanitized_worker_environment()
    except Exception:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker bootstrap could not be sealed"
        ) from None
    if (
        not encoded_request
        or len(encoded_request) > _STDIO_WORKER_MAX_REQUEST_BYTES
        or float(time.monotonic()) >= supervisor_deadline
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery worker request exceeded its deadline or cap"
        )
    supervisor = supervisor_factory(
        job_name=observed["supervisor_job_name"]
    )
    primary_error: BaseException | None = None
    output: bytes | None = None
    try:
        output = supervisor.run(
            argv=command,
            cwd=root,
            env=environment,
            input_bytes=encoded_request,
            deadline_monotonic=supervisor_deadline,
        )
    except BaseException as exc:
        primary_error = exc
    try:
        supervisor.close()
    except BaseException as exc:
        exc.__cause__ = primary_error
        primary_error = exc
    if primary_error is not None:
        raise primary_error
    if output is None or float(time.monotonic()) >= supervisor_deadline:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication recovery exceeded its 300-second deadline"
        )
    result = _decode_publication_recovery_worker_output(
        output,
        attempt_id=observed["attempt_id"],
    )
    if float(time.monotonic()) >= supervisor_deadline:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication recovery exceeded its 300-second deadline"
        )
    return result


def _recover_production_publication_from_entry(
    *,
    repo_root: Path,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    sec_user_agent: str,
    invocation_started_at: float,
) -> dict[str, Any]:
    if (
        type(invocation_started_at) is not float
        or not math.isfinite(invocation_started_at)
        or invocation_started_at <= 0.0
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication-recovery entry clock failed"
        )
    deadline = (
        invocation_started_at + MAX_PUBLICATION_RECOVERY_SECONDS
    )
    request = _build_publication_recovery_worker_request(
        repo_root=repo_root,
        implementation_manifest=implementation_manifest,
        attempt_id=attempt_id,
        sec_user_agent=sec_user_agent,
        supervisor_started_at=invocation_started_at,
        supervisor_deadline=deadline,
    )
    if float(time.monotonic()) >= deadline:
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Publication recovery exceeded its 300-second deadline"
        )
    return _supervise_publication_recovery_worker(
        request,
        supervisor_deadline=deadline,
    )


def recover_production_publication(
    *,
    repo_root: Path,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    sec_user_agent: str,
) -> dict[str, Any]:
    """Run one full non-effectful recovery in a fresh killable child."""

    invocation_started_at = time.monotonic()
    return _recover_production_publication_from_entry(
        repo_root=repo_root,
        implementation_manifest=implementation_manifest,
        attempt_id=attempt_id,
        sec_user_agent=sec_user_agent,
        invocation_started_at=invocation_started_at,
    )


def _production_worker_cli(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    workers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
        "--acquisition-worker": _acquisition_worker_payload,
        "--phase-worker": _phase_worker_dispatch,
        "--deterministic-evaluation-worker": (
            _deterministic_evaluation_worker_payload
        ),
        "--publication-recovery-worker": (
            _publication_recovery_worker_payload
        ),
    }
    if len(arguments) != 1 or arguments[0] not in workers:
        return 2
    try:
        _verify_isolated_worker_bootstrap()
        request = _read_stdio_worker_request(
            require_canonical=(
                arguments[0] == "--publication-recovery-worker"
            )
        )
        result = workers[arguments[0]](request)
        _write_stdio_worker_result(result)
    except BaseException:
        return 1
    return 0


__all__ = [
    "LATENCY_PREFLIGHT_RECEIPT_SCHEMA_VERSION",
    "PRODUCTION_ACQUISITION_ADAPTER_SCHEMA_VERSION",
    "PRODUCTION_AUTHORITIES_SCHEMA_VERSION",
    "PRODUCTION_AUTHORITY_SCHEMA_VERSION",
    "PRODUCTION_AUTHORITY_VERIFIER_ID",
    "PRODUCTION_PHASE_EXECUTOR_SCHEMA_VERSION",
    "PRODUCTION_PUBLICATION_RECOVERY_REQUEST_SCHEMA_VERSION",
    "PRODUCTION_PUBLICATION_RUNTIME_SCHEMA_VERSION",
    "PRODUCTION_SEC_SESSION_SCHEMA_VERSION",
    "SEMANTIC_BATCH_RECEIPT_SCHEMA_VERSION",
    "SEMANTIC_EVENT_RECEIPT_SCHEMA_VERSION",
    "SEMANTIC_EXTRACTION_ROW_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayProductionError",
    "SUPERVISED_PUBLICATION_RECOVERY_WORKER_AUTHORITY_SCHEMA_VERSION",
    "VerifiedProductionAuthority",
    "VerifiedProductionAuthorities",
    "VerifiedSupervisedPublicationRecoveryWorkerAuthority",
    "create_production_sec_transport",
    "is_verified_production_authority",
    "is_verified_production_authorities",
    "is_verified_supervised_publication_recovery_worker_authority",
    "issue_verified_production_authority",
    "recover_production_publication",
    "verify_production_authorities",
]


if __name__ == "__main__":
    raise SystemExit(_production_worker_cli())
