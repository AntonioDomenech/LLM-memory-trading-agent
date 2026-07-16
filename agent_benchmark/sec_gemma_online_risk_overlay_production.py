"""Source-bound production authorities for the v2.1 overlay.

This module owns the concrete external HTTP session used by the reviewed SEC
transport.  It deliberately does not expose a generic transport factory:
production objects can be created only from an opaque live source-tree
verification for the clean pushed implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
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
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    MAX_DETERMINISTIC_SECONDS,
    MAX_MODEL_SECONDS,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    MODEL_NAME,
    NEW_SOURCE_FILES,
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
    "aapl-sec-gemma-online-risk-overlay-v2-1-production-authority-v2"
)
PRODUCTION_AUTHORITY_VERIFIER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-production-authority-verifier-v1"
)
PRODUCTION_SEC_SESSION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-sec-session-v1"
)
PRODUCTION_AUTHORITIES_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-production-authorities-v1"
)
PRODUCTION_ACQUISITION_ADAPTER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-production-acquisition-adapter-v1"
)
PRODUCTION_PHASE_EXECUTOR_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-production-phase-executor-v1"
)
SEMANTIC_EXTRACTION_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-semantic-extraction-row-v1"
)
SEMANTIC_EVENT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-semantic-event-receipt-v1"
)
SEMANTIC_BATCH_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-semantic-batch-receipt-v1"
)
LATENCY_PREFLIGHT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-latency-preflight-v1"
)

_AUTHORITY_SENTINEL = object()
_AUTHORITIES_SENTINEL = object()
_ADAPTER_SENTINEL = object()
_EXECUTOR_SENTINEL = object()
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
    """Issue authority only for the exact clean pushed v2.1 implementation."""

    if not is_verified_source_tree(verified_source_tree):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Production authority requires live source verification"
        )
    material = source_verification_material(verified_source_tree)
    if (
        material["schema_version"] != SOURCE_VERIFICATION_SCHEMA_VERSION
        or material["contract_sha256"] != CONTRACT_SHA256
        or material["branch"] != (
            "codex/aapl-sec-gemma-online-risk-overlay-v2-1"
        )
        or material["head_commit"] != material["upstream_commit"]
        or {
            item["role"]: item["path"]
            for item in material["new_sources"]
        }
        != dict(NEW_SOURCE_FILES)
    ):
        raise SecGemmaOnlineRiskOverlayProductionError(
            "Live source verification is not the exact v2.1 implementation"
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
_STDIO_WORKER_FLAG_RE: Final[re.Pattern[str]] = re.compile(
    r"--[a-z][a-z0-9-]{1,62}-worker\Z"
)
_WORKER_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "--acquisition-worker",
        "--phase-worker",
        "--deterministic-evaluation-worker",
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
    marker = "aapl-sec-gemma-online-risk-overlay-v2-1-worker-import-guard-v1"

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
            "aapl-sec-gemma-online-risk-overlay-v2-1-"
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


def _read_stdio_worker_request() -> dict[str, Any]:
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
        rehydrate_contiguous_production_acquisitions,
    )

    recovered = rehydrate_contiguous_production_acquisitions(
        vault=vault,
        store=store,
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


class VerifiedProductionAuthorities:
    """Opaque identity-equal production composition for the runner."""

    __slots__ = (
        "_acquisition_adapter",
        "_authority",
        "_final_registry_authority",
        "_implementation",
        "_phase_executor",
        "_private_identity",
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
        final_registry_authority: Any,
        _sentinel: object,
    ) -> None:
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            ExternalGitTagPublisher,
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
            or type(final_registry_authority)
            is not FinalRegistryAuthorizer
            or acquisition_adapter._store is not store
            or acquisition_adapter._vault is not vault
            or final_registry_authority._publisher
            is not report_publisher
            or final_registry_authority._store is not store
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
    def final_registry_authority(self) -> Any:
        return self._final_registry_authority

    @property
    def store(self) -> Any:
        return self._store

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
            or self._report_publisher._clock is not time.monotonic
            or self._final_registry_authority._clock
            is not time.monotonic
            or self._final_registry_authority._publisher
            is not self._report_publisher
            or self._final_registry_authority._store is not self._store
            or self._acquisition_adapter._repo_root != self._repo_root
            or self._authority.repo_root != self._repo_root
            or self._report_publisher._repo_root != self._repo_root
            or self._final_registry_authority._repo_root
            != self._repo_root
            or self._report_publisher._implementation
            != self._implementation
            or getattr(self._vault, "_production_authority", None)
            is not True
            or getattr(self._vault, "_bound_store_instance_id", None)
            != getattr(self._store, "_store_instance_id", None)
        ):
            raise SecGemmaOnlineRiskOverlayProductionError(
                "Production authority component identity changed"
            )
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
    publisher = ExternalGitTagPublisher(
        repo_root=root,
        implementation_manifest=implementation,
        clock=time.monotonic,
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
        final_registry_authority=registry,
        _sentinel=_AUTHORITIES_SENTINEL,
    )
    bundle.reverify()
    return bundle


def _production_worker_cli(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    workers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
        "--acquisition-worker": _acquisition_worker_payload,
        "--phase-worker": _phase_worker_dispatch,
        "--deterministic-evaluation-worker": (
            _deterministic_evaluation_worker_payload
        ),
    }
    if len(arguments) != 1 or arguments[0] not in workers:
        return 2
    try:
        _verify_isolated_worker_bootstrap()
        request = _read_stdio_worker_request()
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
    "PRODUCTION_SEC_SESSION_SCHEMA_VERSION",
    "SEMANTIC_BATCH_RECEIPT_SCHEMA_VERSION",
    "SEMANTIC_EVENT_RECEIPT_SCHEMA_VERSION",
    "SEMANTIC_EXTRACTION_ROW_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayProductionError",
    "VerifiedProductionAuthority",
    "VerifiedProductionAuthorities",
    "create_production_sec_transport",
    "is_verified_production_authority",
    "is_verified_production_authorities",
    "issue_verified_production_authority",
    "verify_production_authorities",
]


if __name__ == "__main__":
    raise SystemExit(_production_worker_cli())
