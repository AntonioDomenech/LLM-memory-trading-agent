"""Pure, fail-closed contract for the v3.9 SEC-to-science bridge.

This module performs no filesystem, Git, clock, network, SEC, Yahoo, Ollama,
market, model, broker, or real-money I/O.  It contains only immutable public
authority, canonical identity helpers, and validators for evidence gathered by
the I/O-owning v3.9 modules.

The scientific payload is not copied or translated here.  It is projected from
the pinned v2.2 contract through the twelve preregistered keys and must reproduce
the frozen 38,320-byte canonical payload exactly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final


class ContractViolation(ValueError):
    """A redacted, stable rejection raised for a contract mismatch."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


# Public version and Git authority.
CONTRACT_VERSION: Final[str] = "aapl-sec-gemma-lean-science-v3-9"
CONTRACT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-contract-v1"
)
BRANCH_NAME: Final[str] = "codex/aapl-sec-gemma-lean-science-v3-9"
DEVELOPMENT_COMMAND: Final[str] = "development"

BASE_COMMIT: Final[str] = "0c4b01cf5f1ef77548658d9bfa38e76fb1b70635"
BASE_TREE: Final[str] = "68a9176ed22edeb2edc0f00ae1179c8724603539"
BASE_PARENT: Final[str] = "a0f971184ad26630478182be97f01c46c80498e1"
PREREGISTRATION_COMMIT: Final[str] = (
    "d50c33515ed9597b2fc07bb40c922a3f3166fbdd"
)
PREREGISTRATION_TREE: Final[str] = "946a009fb9e479778ea1881bae302d802578248b"
PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_9.md"
)
PREREGISTRATION_GIT_BLOB_SHA1: Final[str] = (
    "2b7042fd948cad91925c9fe9e01784ea0ff739c8"
)
PREREGISTRATION_LITERAL_SHA256: Final[str] = (
    "feb9bd040e0ed65aed125e52e79e10bc8f87cd7e96776dc67353502fc0983446"
)
PREREGISTRATION_LITERAL_BYTES: Final[int] = 43_098

# Pushed v3.8 public authority.
V38_TERMINAL_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3_8/DEVELOPMENT_ACQUISITION.json"
)
V38_TERMINAL_BLOB_SHA1: Final[str] = (
    "28075db4e7cb85bfcdeb92e7b1aecce28f3a49fa"
)
V38_TERMINAL_LITERAL_SHA256: Final[str] = (
    "c176c6fb5e200dd363be656562d72bc40f722912221fe7f5cd52d8e42cc9723a"
)
V38_TERMINAL_INTERNAL_SHA256: Final[str] = (
    "d72059f39eb4b1019ce83799682eadc7eecf3fd77310f43bee8537433a20cb20"
)
V38_SOURCE_AUTHORITY_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3_8/development/"
    "source-authority-d7f87ac22e58ac4b50033af95e6f768692f1e31a6890e9f50e49f20d59ff0d0a.json"
)
V38_SOURCE_AUTHORITY_BLOB_SHA1: Final[str] = (
    "d2d17dc2e1159e79bbcd2ca3f63cb733289b404f"
)
V38_SOURCE_AUTHORITY_LITERAL_SHA256: Final[str] = (
    "d7f87ac22e58ac4b50033af95e6f768692f1e31a6890e9f50e49f20d59ff0d0a"
)
V38_SOURCE_MODULE_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_lean_v38_source.py"
)
V38_SOURCE_MODULE_LITERAL_SHA256: Final[str] = (
    "8f624df2b28425b8e31db81997aeb0387d3c19418c0781e3b04fca66830d0f04"
)
V38_SOURCE_MODULE_GIT_BLOB_SHA1: Final[str] = (
    "b04263d5b3a8bdd74362fd02bc637d773d76ff9a"
)
V38_ACQUISITION_MODULE_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_lean_v38_acquisition.py"
)
V38_ACQUISITION_MODULE_LITERAL_SHA256: Final[str] = (
    "8ba5b3dee285a5e12e631c5d5e8fbafce1a66d1c81b4c5736371ac8ceffbb3f6"
)
V38_ACQUISITION_MODULE_GIT_BLOB_SHA1: Final[str] = (
    "b5fbd94ecaff681ac698e47dd9ea0448d7886ee3"
)

# Authenticated, read-only v3.8 private authority.  These are hashes and
# aggregate counts only; no accession, filename, URL, source body, or contact is
# present in this public contract.
V38_CHECKPOINT_FILE_SHA256: Final[str] = (
    "cd2896d627d0694c9efc202c2c8579569265fb4f61f5f74bb9d7db875fbc981e"
)
V38_LOGICAL_CHECKPOINT_SHA256: Final[str] = (
    "704a4a9554444201ec9468e74f8c77abe82a3e3320acce29cac0c7ace11dd6fc"
)
V38_STAGE_SOURCE_SEAL_SHA256: Final[str] = (
    "0eb7c6a83de59d44b3a5ceeca5918b03070ff1f508cc6a5411cd81871368f413"
)
V38_COMPACT_REPLAY_SHA256: Final[str] = (
    "64a7b7776206b008c0dffe26d4a14be9a092281a7834d878332d4f741db05822"
)
V38_ROLE_MANIFEST_INVENTORY_SHA256: Final[str] = (
    "b6c66fba0d7482ee9b1b1892351b0d68dcb43f4ce2a2916e3ea96a316fbc2ad7"
)
V38_ROLE_PLAN_SHA256: Final[str] = (
    "6a03e4f8c2a444dd0612d7b1ff74cc605e026ace54941696a5ae180dd226110a"
)
V38_ROLE_COUNTS: Final[tuple[tuple[str, int], ...]] = (
    ("I", 97),
    ("U", 75),
    ("D", 75),
)
V38_SEC_REQUEST_COUNT: Final[int] = 199
EXPERIMENT_FAMILY_SEC_REQUEST_COUNT: Final[int] = 964
V38_INVENTORY_FILE_COUNT: Final[int] = 1_410
V38_INVENTORY_BYTE_COUNT: Final[int] = 612_601_642
V38_INVENTORY_SHA256: Final[str] = (
    "850f3a022fcaca5b56156d7843f35b05d3bc5732bf480634bd9872bfac7194f2"
)

# Frozen scientific authority.
SCIENTIFIC_PARENT_COMMIT: Final[str] = (
    "efbc481c57e480d48303763163676e64e87df49d"
)
SCIENTIFIC_PARENT_TREE: Final[str] = (
    "bc91d619a4d04680d9600b1acd74f0a29662d0f9"
)
SCIENTIFIC_CONTRACT_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_online_risk_overlay_contract.py"
)
SCIENTIFIC_CONTRACT_GIT_BLOB_SHA1: Final[str] = (
    "8a1c18728eeb7a31864397bb0c8e992b85d2320b"
)
SCIENTIFIC_CONTRACT_LITERAL_SHA256: Final[str] = (
    "fccb45098f505a2f272f970762f928fbe8b75a23380292924b67d2bdd4696d2e"
)
SCIENTIFIC_CONTRACT_INTERNAL_SHA256: Final[str] = (
    "64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d"
)
SCIENCE_PROJECTION_KEYS: Final[tuple[str, ...]] = (
    "objective",
    "event_availability",
    "chronology",
    "data",
    "features",
    "gemma",
    "learner",
    "policy",
    "ledger",
    "metric_definitions",
    "gates",
    "evidence_classification",
)
SCIENCE_PROJECTION_BYTE_COUNT: Final[int] = 38_320
SCIENCE_PROJECTION_SHA256: Final[str] = (
    "1ee05d2916752752bbef3710d70c7dab664fb82b9ac22829dac8897401058609"
)
FROZEN_SCIENCE_PROJECTION_SHA256: Final[str] = SCIENCE_PROJECTION_SHA256

# Frozen development boundary and scientific chronology.
DEVELOPMENT_DOCUMENT_COUNT: Final[int] = 75
DEVELOPMENT_PILOT_COUNT: Final[int] = 5
DEVELOPMENT_REMAINING_COUNT: Final[int] = 70
DEVELOPMENT_MARKET_START: Final[str] = "1998-01-01"
DEVELOPMENT_MARKET_END_EXCLUSIVE: Final[str] = "2019-01-01"
DEVELOPMENT_CORPUS_END: Final[str] = "2018-12-31"
DEVELOPMENT_WARMUP_END: Final[str] = "2004-12-31"
DEVELOPMENT_BLOCKS: Final[tuple[tuple[str, str, str], ...]] = (
    ("block_1", "2005-01-03", "2007-12-31"),
    ("block_2", "2008-01-02", "2010-12-31"),
    ("block_3", "2011-01-03", "2013-12-31"),
    ("block_4", "2014-01-02", "2016-12-30"),
    ("block_5", "2017-01-03", "2018-12-31"),
)
SOURCE_PROJECTION_ORDER: Final[tuple[str, ...]] = (
    "availability_session",
    "accession",
)
SCIENCE_EVENT_ORDER: Final[tuple[str, ...]] = (
    "availability_session",
    "acceptance_datetime",
    "accession",
)
PILOT_SORT_ORDER: Final[tuple[str, ...]] = (
    "canonical_request_byte_length_descending",
    "accession_ascending",
)

# Frozen local-model identity and request semantics.
MODEL_NAME: Final[str] = "gemma4:12b"
MODEL_MANIFEST_SHA256: Final[str] = (
    "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
)
MODEL_CONFIG_DIGEST: Final[str] = (
    "c805f5b265d8e695c44f4065dfc368206cd8026447604925fef8db57ee32ee23"
)
MODEL_LAYER_DIGESTS: Final[tuple[str, ...]] = (
    "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
    "675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842",
    "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594",
    "56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3",
)
MODEL_ACTIVE_FROM_BLOB_COUNT: Final[int] = 2
OLLAMA_VERSION: Final[str] = "0.32.0"
RUNTIME_FINGERPRINT_SHA256: Final[str] = (
    "816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77"
)
RUNTIME_VERSION_RESPONSE_SHA256: Final[str] = (
    "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
)
RUNTIME_SHOW_SEMANTIC_SHA256: Final[str] = (
    "5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6"
)
RUNTIME_SHOW_SEMANTIC_EXCLUDED_KEYS: Final[tuple[str, ...]] = ("modified_at",)
RUNTIME_SHOW_RAW_SHA256_DIAGNOSTIC_ONLY: Final[str] = (
    "5f56fb0fb2214ddcb9fa21c66aa31e37297f553e8758aeda5958f0f287d70893"
)
RUNTIME_MODEL_INFO_SHA256: Final[str] = (
    "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
)
PROMPT_SHA256: Final[str] = (
    "9ed8496ed101c138cdbee162bdf6dfd53434f6f1e0c64fc93d844405d6eae9f7"
)
SCHEMA_SHA256: Final[str] = (
    "1707ae581abb1a256dfb1ee8f51efd9dd67df5d9b3e8f7c22ee2d92b67b6f82b"
)
MODEL_TEMPERATURE: Final[int] = 0
MODEL_SEED: Final[int] = 0
MODEL_CONTEXT_TOKENS: Final[int] = 6_144
MODEL_OUTPUT_TOKENS: Final[int] = 512
MODEL_INPUT_MAX_BYTES: Final[int] = 20_000
MODEL_INPUT_MAX_SENTENCES: Final[int] = 72
MODEL_INPUT_MAX_SENTENCE_CHARACTERS: Final[int] = 220
MODEL_RESPONSE_MAX_BYTES: Final[int] = 256 * 1024
OLLAMA_VERSION_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/version"
OLLAMA_SHOW_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/show"
OLLAMA_CHAT_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/chat"
RUNTIME_PROBE_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("GET", OLLAMA_VERSION_ENDPOINT),
    ("POST", OLLAMA_SHOW_ENDPOINT),
)
NORMAL_IDENTITY_REQUEST_COUNT: Final[int] = 4
PAUSED_RESUMED_IDENTITY_REQUEST_COUNT: Final[int] = 8

# Frozen Yahoo batch.
YAHOO_ENDPOINT: Final[str] = (
    "https://query1.finance.yahoo.com/v8/finance/chart"
)
YAHOO_SYMBOL_ORDER: Final[tuple[str, ...]] = (
    "AAPL",
    "SPY",
    "QQQ",
    "IWM",
    "VIX",
    "TNX",
)
YAHOO_PROVIDER_SYMBOLS: Final[tuple[tuple[str, str], ...]] = (
    ("AAPL", "AAPL"),
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("IWM", "IWM"),
    ("VIX", "^VIX"),
    ("TNX", "^TNX"),
)
YAHOO_QUERY_ITEMS: Final[tuple[tuple[str, str], ...]] = (
    ("period1", "883612800"),
    ("period2", "1546300800"),
    ("interval", "1d"),
    ("includePrePost", "false"),
    ("includeAdjustedClose", "true"),
    ("events", "div,splits"),
)
YAHOO_URLS: Final[tuple[str, ...]] = (
    f"{YAHOO_ENDPOINT}/AAPL?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/SPY?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/QQQ?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/IWM?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/%5EVIX?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
    f"{YAHOO_ENDPOINT}/%5ETNX?period1=883612800&period2=1546300800&interval=1d"
    "&includePrePost=false&includeAdjustedClose=true&events=div%2Csplits",
)
YAHOO_REQUEST_COUNT: Final[int] = 6
YAHOO_REQUEST_TIMEOUT_SECONDS: Final[int] = 30
YAHOO_MAX_RESPONSE_BYTES: Final[int] = 64 * 1024 * 1024
YAHOO_MAX_TOTAL_RESPONSE_BYTES: Final[int] = 128 * 1024 * 1024
YAHOO_MAX_BATCH_SECONDS: Final[int] = 210
YAHOO_USER_AGENT: Final[str] = (
    "LLM-memory-trading-agent/1.0 market-evidence (no-auth; one-shot)"
)

# Fixed pilot rule.
PILOT_PROJECTED_THRESHOLD_NS: Final[int] = 43_200_000_000_000
PILOT_PROJECTION_MULTIPLIER: Final[int] = 70

# One-shot and publication paths.
DEVELOPMENT_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-development-001"
)
PRIVATE_NAMESPACE: Final[str] = "data/aapl_sec_gemma_lean_science_v3_9"
PRIVATE_PREFLIGHT_NAMESPACE: Final[str] = f"{PRIVATE_NAMESPACE}/preflight"
PRIVATE_DEVELOPMENT_NAMESPACE: Final[str] = f"{PRIVATE_NAMESPACE}/development"
PUBLIC_EVIDENCE_ROOT: Final[str] = "e/aapl_sec_gemma_lean_science_v3_9"
PREFLIGHT_ARTIFACT_PATH: Final[str] = (
    f"{PUBLIC_EVIDENCE_ROOT}/DEVELOPMENT_PREFLIGHT.json"
)
PAUSE_ARTIFACT_PATH: Final[str] = (
    f"{PUBLIC_EVIDENCE_ROOT}/DEVELOPMENT_PAUSE.json"
)
RESULT_ARTIFACT_PATH: Final[str] = (
    f"{PUBLIC_EVIDENCE_ROOT}/DEVELOPMENT_RESULT.json"
)
COMPARISON_PATH: Final[str] = "e/APPROACH_COMPARISON.md"
CONTINUATION_PREREGISTRATION_PATH: Final[str] = (
    "docs/aapl_sec_gemma_lean_science_v3_9_continuation.md"
)

IMPLEMENTATION_PRODUCTION_PATHS: Final[tuple[str, ...]] = (
    "agent_benchmark/sec_gemma_lean_science_v39_contract.py",
    "agent_benchmark/sec_gemma_lean_science_v39_bridge.py",
    "agent_benchmark/sec_gemma_lean_science_v39_journal.py",
    "agent_benchmark/sec_gemma_lean_science_v39_store.py",
    "agent_benchmark/sec_gemma_lean_science_v39_preflight.py",
    "agent_benchmark/sec_gemma_lean_science_v39_runner.py",
)
IMPLEMENTATION_TEST_PATHS: Final[tuple[str, ...]] = (
    "tests/test_sec_gemma_lean_science_v39_contract.py",
    "tests/test_sec_gemma_lean_science_v39_bridge.py",
    "tests/test_sec_gemma_lean_science_v39_journal.py",
    "tests/test_sec_gemma_lean_science_v39_store.py",
    "tests/test_sec_gemma_lean_science_v39_preflight.py",
    "tests/test_sec_gemma_lean_science_v39_runner.py",
)
IMPLEMENTATION_ALLOWED_PATHS: Final[tuple[str, ...]] = (
    *IMPLEMENTATION_PRODUCTION_PATHS,
    *IMPLEMENTATION_TEST_PATHS,
)

EFFECT_COUNT_KEYS: Final[tuple[str, ...]] = (
    "sec_requests",
    "experiment_family_sec_requests",
    "yahoo_requests",
    "ollama_identity_http_requests",
    "ollama_chat_generations",
    "retries",
    "repairs",
    "pulls",
    "fallbacks",
    "paid_calls",
    "confirmation_final_data_opens",
    "broker_effects",
    "real_money_effects",
)


def _effect_budget(*, identity_requests: int, generations: int) -> dict[str, int]:
    return {
        "sec_requests": 0,
        "experiment_family_sec_requests": EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
        "yahoo_requests": YAHOO_REQUEST_COUNT,
        "ollama_identity_http_requests": identity_requests,
        "ollama_chat_generations": generations,
        "retries": 0,
        "repairs": 0,
        "pulls": 0,
        "fallbacks": 0,
        "paid_calls": 0,
        "confirmation_final_data_opens": 0,
        "broker_effects": 0,
        "real_money_effects": 0,
    }


NORMAL_EFFECT_BUDGET: Final[dict[str, int]] = _effect_budget(
    identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
    generations=DEVELOPMENT_DOCUMENT_COUNT,
)
PAUSED_RESUMED_EFFECT_BUDGET: Final[dict[str, int]] = _effect_budget(
    identity_requests=PAUSED_RESUMED_IDENTITY_REQUEST_COUNT,
    generations=DEVELOPMENT_DOCUMENT_COUNT,
)
PILOT_PAUSE_EFFECT_BUDGET: Final[dict[str, int]] = _effect_budget(
    identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
    generations=DEVELOPMENT_PILOT_COUNT,
)
ZERO_EFFECT_BUDGET: Final[dict[str, int]] = {
    key: (EXPERIMENT_FAMILY_SEC_REQUEST_COUNT if key == "experiment_family_sec_requests" else 0)
    for key in EFFECT_COUNT_KEYS
}

HIGH_LEVEL_EFFECT_ORDER: Final[tuple[str, ...]] = (
    "authenticate_preregistration_and_implementation",
    "authenticate_v38_source_read_only",
    "rebuild_projection_requests_and_pilot_commitments",
    "consume_one_shot_attempt",
    "seal_six_yahoo_responses_without_opening_values",
    "model_batch_pre_probe",
    "seal_five_pilots",
    "apply_timing_only_pause_rule",
    "seal_remaining_seventy_if_not_paused",
    "model_batch_post_probe",
    "open_and_validate_complete_market_and_semantic_batches",
    "run_frozen_deterministic_science",
    "independent_replay_and_no_leverage_proofs",
    "seal_redacted_terminal_evidence",
)


def _reject(code: str) -> None:
    raise ContractViolation(code)


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > 128:
        _reject("v39_contract_json_depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _reject("v39_contract_json_nonfinite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _reject("v39_contract_json_key")
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    _reject("v39_contract_json_type")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the preregistered UTF-8 canonical JSON representation."""

    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ContractViolation("v39_contract_json_encoding") from exc


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return a lowercase unprefixed SHA-256 for a bytes-like value."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        _reject("v39_contract_hash_input")
    return hashlib.sha256(bytes(value)).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def add_self_sha256(
    value: Mapping[str, Any],
    *,
    field: str = "manifest_sha256",
) -> dict[str, Any]:
    """Detach a mapping and bind its exact unsigned canonical JSON."""

    if not isinstance(value, Mapping) or not isinstance(field, str) or not field:
        _reject("v39_contract_self_hash_shape")
    if field in value:
        _reject("v39_contract_self_hash_present")
    detached = copy.deepcopy(dict(value))
    detached[field] = canonical_sha256(detached)
    return detached


def validate_self_sha256(
    value: Any,
    *,
    field: str = "manifest_sha256",
) -> dict[str, Any]:
    """Validate a strict lowercase SHA-256 self-hash and detach the mapping."""

    if not isinstance(value, Mapping) or not isinstance(field, str) or not field:
        _reject("v39_contract_self_hash_shape")
    detached = copy.deepcopy(dict(value))
    observed = detached.pop(field, None)
    if not _is_hex(observed, 64) or observed != canonical_sha256(detached):
        _reject("v39_contract_self_hash_mismatch")
    return copy.deepcopy(dict(value))


def project_science_manifest(source_manifest: Any) -> dict[str, Any]:
    """Select exactly the twelve scientific keys from a source manifest."""

    if not isinstance(source_manifest, Mapping):
        _reject("v39_contract_science_source_shape")
    if any(key not in source_manifest for key in SCIENCE_PROJECTION_KEYS):
        _reject("v39_contract_science_source_keys")
    return {
        key: copy.deepcopy(source_manifest[key]) for key in SCIENCE_PROJECTION_KEYS
    }


def verify_frozen_science_projection(value: Any) -> dict[str, Any]:
    """Accept only the exact 38,320-byte frozen scientific projection."""

    if not isinstance(value, Mapping) or set(value) != set(SCIENCE_PROJECTION_KEYS):
        _reject("v39_contract_science_projection_keys")
    detached = {key: copy.deepcopy(value[key]) for key in SCIENCE_PROJECTION_KEYS}
    encoded = canonical_json_bytes(detached)
    if len(encoded) != SCIENCE_PROJECTION_BYTE_COUNT:
        _reject("v39_contract_science_projection_size")
    if sha256_bytes(encoded) != SCIENCE_PROJECTION_SHA256:
        _reject("v39_contract_science_projection_hash")
    return detached


def build_frozen_science_projection() -> dict[str, Any]:
    """Build and authenticate the frozen projection through the pinned adapter."""

    # This import is pure: the predecessor contract has its own literal
    # self-check and performs no external I/O.
    from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
        CONTRACT_SHA256 as predecessor_internal_sha256,
        build_contract_manifest as build_predecessor_manifest,
    )

    if predecessor_internal_sha256 != SCIENTIFIC_CONTRACT_INTERNAL_SHA256:
        _reject("v39_contract_science_parent_internal")
    return verify_frozen_science_projection(
        project_science_manifest(build_predecessor_manifest())
    )


def build_source_authority_pins() -> dict[str, Any]:
    """Return a detached public description of all frozen v3.8 authority pins."""

    return {
        "base": {
            "commit": BASE_COMMIT,
            "tree": BASE_TREE,
            "parent": BASE_PARENT,
        },
        "terminal": {
            "path": V38_TERMINAL_PATH,
            "git_blob_sha1": V38_TERMINAL_BLOB_SHA1,
            "literal_sha256": V38_TERMINAL_LITERAL_SHA256,
            "internal_sha256": V38_TERMINAL_INTERNAL_SHA256,
        },
        "source_authority": {
            "path": V38_SOURCE_AUTHORITY_PATH,
            "git_blob_sha1": V38_SOURCE_AUTHORITY_BLOB_SHA1,
            "literal_sha256": V38_SOURCE_AUTHORITY_LITERAL_SHA256,
        },
        "modules": {
            "source": {
                "path": V38_SOURCE_MODULE_PATH,
                "git_blob_sha1": V38_SOURCE_MODULE_GIT_BLOB_SHA1,
                "literal_sha256": V38_SOURCE_MODULE_LITERAL_SHA256,
            },
            "acquisition": {
                "path": V38_ACQUISITION_MODULE_PATH,
                "git_blob_sha1": V38_ACQUISITION_MODULE_GIT_BLOB_SHA1,
                "literal_sha256": V38_ACQUISITION_MODULE_LITERAL_SHA256,
            },
        },
        "private_aggregates": {
            "checkpoint_file_sha256": V38_CHECKPOINT_FILE_SHA256,
            "logical_checkpoint_sha256": V38_LOGICAL_CHECKPOINT_SHA256,
            "stage_source_seal_sha256": V38_STAGE_SOURCE_SEAL_SHA256,
            "compact_replay_sha256": V38_COMPACT_REPLAY_SHA256,
            "role_manifest_inventory_sha256": V38_ROLE_MANIFEST_INVENTORY_SHA256,
            "role_plan_sha256": V38_ROLE_PLAN_SHA256,
            "role_counts": dict(V38_ROLE_COUNTS),
            "v38_sec_requests": V38_SEC_REQUEST_COUNT,
            "experiment_family_sec_requests": EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
            "inventory_file_count": V38_INVENTORY_FILE_COUNT,
            "inventory_byte_count": V38_INVENTORY_BYTE_COUNT,
            "inventory_sha256": V38_INVENTORY_SHA256,
        },
    }


def build_effect_budgets() -> dict[str, dict[str, int]]:
    return {
        "zero_effect_preflight": {
            key: (
                EXPERIMENT_FAMILY_SEC_REQUEST_COUNT
                if key == "experiment_family_sec_requests"
                else 0
            )
            for key in EFFECT_COUNT_KEYS
        },
        "normal_complete": _effect_budget(
            identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
            generations=DEVELOPMENT_DOCUMENT_COUNT,
        ),
        "pilot_pause": _effect_budget(
            identity_requests=NORMAL_IDENTITY_REQUEST_COUNT,
            generations=DEVELOPMENT_PILOT_COUNT,
        ),
        "paused_resumed_complete": _effect_budget(
            identity_requests=PAUSED_RESUMED_IDENTITY_REQUEST_COUNT,
            generations=DEVELOPMENT_DOCUMENT_COUNT,
        ),
    }


def validate_command(value: Any) -> str:
    if type(value) is not str or value != DEVELOPMENT_COMMAND:
        _reject("v39_contract_command")
    return DEVELOPMENT_COMMAND


def validate_effect_counts(value: Any, *, route: str) -> dict[str, int]:
    budgets = build_effect_budgets()
    if route not in budgets:
        _reject("v39_contract_effect_route")
    if not isinstance(value, Mapping) or set(value) != set(EFFECT_COUNT_KEYS):
        _reject("v39_contract_effect_keys")
    if any(type(item) is not int or item < 0 for item in value.values()):
        _reject("v39_contract_effect_type")
    observed = dict(value)
    if observed != budgets[route]:
        _reject("v39_contract_effect_counts")
    return copy.deepcopy(observed)


def projected_pilot_ns(durations_ns: Any) -> int:
    """Compute the only allowed five-pilot timing projection."""

    if (
        not isinstance(durations_ns, Sequence)
        or isinstance(durations_ns, (str, bytes, bytearray))
        or len(durations_ns) != DEVELOPMENT_PILOT_COUNT
        or any(type(value) is not int or value <= 0 for value in durations_ns)
    ):
        _reject("v39_contract_pilot_durations")
    return sum(durations_ns) + PILOT_PROJECTION_MULTIPLIER * max(durations_ns)


def pilot_pause_required(durations_ns: Any) -> bool:
    return projected_pilot_ns(durations_ns) > PILOT_PROJECTED_THRESHOLD_NS


_HEX_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]+\Z")


def _is_hex(value: Any, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and _HEX_RE.fullmatch(value) is not None
    )


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _reject(code)
    return copy.deepcopy(dict(value))


def _validate_changed_paths(
    value: Any,
    expected: Mapping[str, str],
    *,
    code: str,
) -> None:
    if not isinstance(value, Mapping):
        _reject(code)
    if dict(value) != dict(expected):
        _reject(code)


def validate_preregistration_ancestry(value: Any) -> dict[str, Any]:
    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "git_blob_sha1",
            "literal_sha256",
            "literal_bytes",
        }
    )
    item = _strict_mapping(value, fields, code="v39_contract_prereg_shape")
    expected = {
        "branch": BRANCH_NAME,
        "commit": PREREGISTRATION_COMMIT,
        "tree": PREREGISTRATION_TREE,
        "parent": BASE_COMMIT,
        "local_head": PREREGISTRATION_COMMIT,
        "remote_head": PREREGISTRATION_COMMIT,
        "changed_paths": {PREREGISTRATION_PATH: "A"},
        "clean_worktree": True,
        "git_blob_sha1": PREREGISTRATION_GIT_BLOB_SHA1,
        "literal_sha256": PREREGISTRATION_LITERAL_SHA256,
        "literal_bytes": PREREGISTRATION_LITERAL_BYTES,
    }
    if item != expected:
        _reject("v39_contract_prereg_ancestry")
    return item


def validate_implementation_ancestry(value: Any) -> dict[str, Any]:
    """Validate the exact pushed 12-addition implementation commit shape."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "preregistration_authenticated",
            "predecessor_blobs_unchanged",
        }
    )
    item = _strict_mapping(value, fields, code="v39_contract_impl_shape")
    commit = item["commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or commit == PREREGISTRATION_COMMIT
        or not _is_hex(item["tree"], 40)
        or item["parent"] != PREREGISTRATION_COMMIT
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["preregistration_authenticated"] is not True
        or item["predecessor_blobs_unchanged"] is not True
    ):
        _reject("v39_contract_impl_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {path: "A" for path in IMPLEMENTATION_ALLOWED_PATHS},
        code="v39_contract_impl_delta",
    )
    return item


def validate_preflight_ancestry(value: Any) -> dict[str, Any]:
    """Validate the pushed one-path, one-run, zero-effect preflight commit."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "implementation_commit",
            "implementation_tree",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "implementation_authenticated",
            "private_replay_passed",
            "zero_effects_verified",
            "preflight_run_count",
        }
    )
    item = _strict_mapping(value, fields, code="v39_contract_preflight_shape")
    commit = item["commit"]
    implementation_commit = item["implementation_commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(implementation_commit, 40)
        or not _is_hex(item["implementation_tree"], 40)
        or item["parent"] != implementation_commit
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["implementation_authenticated"] is not True
        or item["private_replay_passed"] is not True
        or item["zero_effects_verified"] is not True
        or type(item["preflight_run_count"]) is not int
        or item["preflight_run_count"] != 1
    ):
        _reject("v39_contract_preflight_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {PREFLIGHT_ARTIFACT_PATH: "A"},
        code="v39_contract_preflight_delta",
    )
    return item


def validate_pause_ancestry(value: Any) -> dict[str, Any]:
    """Validate pause commit S as one added artifact above preflight F."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "preflight_commit",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "pilot_guard_authenticated",
        }
    )
    item = _strict_mapping(value, fields, code="v39_contract_pause_shape")
    commit = item["commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(item["preflight_commit"], 40)
        or item["parent"] != item["preflight_commit"]
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["pilot_guard_authenticated"] is not True
    ):
        _reject("v39_contract_pause_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {PAUSE_ARTIFACT_PATH: "A"},
        code="v39_contract_pause_delta",
    )
    return item


def validate_continuation_ancestry(value: Any) -> dict[str, Any]:
    """Validate continuation commit C as one added document above pause S."""

    fields = frozenset(
        {
            "branch",
            "commit",
            "tree",
            "parent",
            "pause_commit",
            "preflight_commit",
            "local_head",
            "remote_head",
            "changed_paths_from_pause",
            "changed_paths_from_preflight",
            "clean_worktree",
            "earlier_blobs_equal_preflight",
        }
    )
    item = _strict_mapping(value, fields, code="v39_contract_continue_shape")
    commit = item["commit"]
    if (
        item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(item["pause_commit"], 40)
        or not _is_hex(item["preflight_commit"], 40)
        or item["parent"] != item["pause_commit"]
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["earlier_blobs_equal_preflight"] is not True
    ):
        _reject("v39_contract_continue_ancestry")
    _validate_changed_paths(
        item["changed_paths_from_pause"],
        {CONTINUATION_PREREGISTRATION_PATH: "A"},
        code="v39_contract_continue_delta",
    )
    _validate_changed_paths(
        item["changed_paths_from_preflight"],
        {
            PAUSE_ARTIFACT_PATH: "A",
            CONTINUATION_PREREGISTRATION_PATH: "A",
        },
        code="v39_contract_continue_cumulative_delta",
    )
    return item


def validate_result_ancestry(value: Any) -> dict[str, Any]:
    """Validate the exact two-path terminal result commit shape."""

    fields = frozenset(
        {
            "route",
            "branch",
            "commit",
            "tree",
            "parent",
            "authorized_parent",
            "authorized_parent_kind",
            "local_head",
            "remote_head",
            "changed_paths",
            "clean_worktree",
            "terminal_sealed",
            "independent_replay_passed",
            "privacy_passed",
            "v38_unchanged",
        }
    )
    item = _strict_mapping(value, fields, code="v39_contract_result_shape")
    route = item["route"]
    parent_kind = item["authorized_parent_kind"]
    allowed_parent_kinds = {
        "normal": {"preflight"},
        "paused_resumed": {"continuation"},
        "indeterminate_before_pause": {"preflight", "continuation"},
    }
    commit = item["commit"]
    if (
        route not in allowed_parent_kinds
        or parent_kind not in allowed_parent_kinds[route]
        or item["branch"] != BRANCH_NAME
        or not _is_hex(commit, 40)
        or not _is_hex(item["tree"], 40)
        or not _is_hex(item["authorized_parent"], 40)
        or item["parent"] != item["authorized_parent"]
        or item["local_head"] != commit
        or item["remote_head"] != commit
        or item["clean_worktree"] is not True
        or item["terminal_sealed"] is not True
        or item["independent_replay_passed"] is not True
        or item["privacy_passed"] is not True
        or item["v38_unchanged"] is not True
    ):
        _reject("v39_contract_result_ancestry")
    _validate_changed_paths(
        item["changed_paths"],
        {RESULT_ARTIFACT_PATH: "A", COMPARISON_PATH: "M"},
        code="v39_contract_result_delta",
    )
    return item


def _build_unsigned_contract_manifest() -> dict[str, Any]:
    science = build_frozen_science_projection()
    development_gates = copy.deepcopy(science["gates"]["development"])
    if len(development_gates) != 22:
        _reject("v39_contract_development_gate_count")
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "branch": BRANCH_NAME,
        "command": DEVELOPMENT_COMMAND,
        "status": "preregistered_implementation_no_external_effect",
        "git_authority": {
            "base_commit": BASE_COMMIT,
            "base_tree": BASE_TREE,
            "base_parent": BASE_PARENT,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_tree": PREREGISTRATION_TREE,
            "preregistration_path": PREREGISTRATION_PATH,
            "preregistration_git_blob_sha1": PREREGISTRATION_GIT_BLOB_SHA1,
            "preregistration_literal_sha256": PREREGISTRATION_LITERAL_SHA256,
            "preregistration_literal_bytes": PREREGISTRATION_LITERAL_BYTES,
        },
        "source_authority": build_source_authority_pins(),
        "science_authority": {
            "parent_commit": SCIENTIFIC_PARENT_COMMIT,
            "parent_tree": SCIENTIFIC_PARENT_TREE,
            "contract_path": SCIENTIFIC_CONTRACT_PATH,
            "contract_git_blob_sha1": SCIENTIFIC_CONTRACT_GIT_BLOB_SHA1,
            "contract_literal_sha256": SCIENTIFIC_CONTRACT_LITERAL_SHA256,
            "contract_internal_sha256": SCIENTIFIC_CONTRACT_INTERNAL_SHA256,
            "projection_keys": list(SCIENCE_PROJECTION_KEYS),
            "projection_bytes": SCIENCE_PROJECTION_BYTE_COUNT,
            "projection_sha256": SCIENCE_PROJECTION_SHA256,
            "projection": science,
            "development_gates": development_gates,
            "development_gate_count": 22,
        },
        "development_boundary": {
            "document_count": DEVELOPMENT_DOCUMENT_COUNT,
            "market_start": DEVELOPMENT_MARKET_START,
            "market_end_exclusive": DEVELOPMENT_MARKET_END_EXCLUSIVE,
            "corpus_end": DEVELOPMENT_CORPUS_END,
            "warmup_end": DEVELOPMENT_WARMUP_END,
            "blocks": [list(item) for item in DEVELOPMENT_BLOCKS],
            "source_projection_order": list(SOURCE_PROJECTION_ORDER),
            "science_event_order": list(SCIENCE_EVENT_ORDER),
            "confirmation_and_final_unreachable": True,
        },
        "model": {
            "name": MODEL_NAME,
            "manifest_sha256": MODEL_MANIFEST_SHA256,
            "config_digest": MODEL_CONFIG_DIGEST,
            "ordered_layer_digests": list(MODEL_LAYER_DIGESTS),
            "active_from_blob_count": MODEL_ACTIVE_FROM_BLOB_COUNT,
            "ollama_version": OLLAMA_VERSION,
            "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
            "version_response_sha256": RUNTIME_VERSION_RESPONSE_SHA256,
            "show_semantic_sha256": RUNTIME_SHOW_SEMANTIC_SHA256,
            "show_semantic_excluded_keys": list(
                RUNTIME_SHOW_SEMANTIC_EXCLUDED_KEYS
            ),
            "show_raw_sha256_diagnostic_only": (
                RUNTIME_SHOW_RAW_SHA256_DIAGNOSTIC_ONLY
            ),
            "model_info_sha256": RUNTIME_MODEL_INFO_SHA256,
            "prompt_sha256": PROMPT_SHA256,
            "schema_sha256": SCHEMA_SHA256,
            "temperature": MODEL_TEMPERATURE,
            "seed": MODEL_SEED,
            "context_tokens": MODEL_CONTEXT_TOKENS,
            "output_tokens": MODEL_OUTPUT_TOKENS,
            "input_caps": {
                "bytes": MODEL_INPUT_MAX_BYTES,
                "sentences": MODEL_INPUT_MAX_SENTENCES,
                "characters_per_sentence": MODEL_INPUT_MAX_SENTENCE_CHARACTERS,
            },
            "response_max_bytes": MODEL_RESPONSE_MAX_BYTES,
            "runtime_probe_order": [list(item) for item in RUNTIME_PROBE_ORDER],
            "chat_endpoint": OLLAMA_CHAT_ENDPOINT,
            "normal_identity_requests": NORMAL_IDENTITY_REQUEST_COUNT,
            "paused_resumed_identity_requests": (
                PAUSED_RESUMED_IDENTITY_REQUEST_COUNT
            ),
            "generation_count": DEVELOPMENT_DOCUMENT_COUNT,
            "retry_repair_pull_fallback_alternate_model": 0,
        },
        "market": {
            "endpoint": YAHOO_ENDPOINT,
            "symbol_order": list(YAHOO_SYMBOL_ORDER),
            "provider_symbols": dict(YAHOO_PROVIDER_SYMBOLS),
            "query_items": [list(item) for item in YAHOO_QUERY_ITEMS],
            "urls": list(YAHOO_URLS),
            "request_count": YAHOO_REQUEST_COUNT,
            "request_timeout_seconds": YAHOO_REQUEST_TIMEOUT_SECONDS,
            "max_response_bytes": YAHOO_MAX_RESPONSE_BYTES,
            "max_total_response_bytes": YAHOO_MAX_TOTAL_RESPONSE_BYTES,
            "max_batch_seconds": YAHOO_MAX_BATCH_SECONDS,
            "user_agent": YAHOO_USER_AGENT,
            "retry_redirect_proxy_cookie_auth_compression_fallback": 0,
        },
        "pilot": {
            "count": DEVELOPMENT_PILOT_COUNT,
            "remaining_count": DEVELOPMENT_REMAINING_COUNT,
            "sort_order": list(PILOT_SORT_ORDER),
            "projection_formula": "sum(durations_ns)+70*max(durations_ns)",
            "projection_multiplier": PILOT_PROJECTION_MULTIPLIER,
            "strict_pause_threshold_ns": PILOT_PROJECTED_THRESHOLD_NS,
            "pause_comparison": "strictly_greater_than",
        },
        "one_shot": {
            "attempt_id": DEVELOPMENT_ATTEMPT_ID,
            "private_namespace": PRIVATE_NAMESPACE,
            "preflight_namespace": PRIVATE_PREFLIGHT_NAMESPACE,
            "development_namespace": PRIVATE_DEVELOPMENT_NAMESPACE,
        },
        "public_paths": {
            "preflight": PREFLIGHT_ARTIFACT_PATH,
            "pause": PAUSE_ARTIFACT_PATH,
            "continuation": CONTINUATION_PREREGISTRATION_PATH,
            "result": RESULT_ARTIFACT_PATH,
            "comparison": COMPARISON_PATH,
        },
        "implementation": {
            "production_paths": list(IMPLEMENTATION_PRODUCTION_PATHS),
            "test_paths": list(IMPLEMENTATION_TEST_PATHS),
            "all_paths": list(IMPLEMENTATION_ALLOWED_PATHS),
            "all_changes_are_additions": True,
            "predecessor_blobs_unchanged": True,
        },
        "effects": {
            "count_keys": list(EFFECT_COUNT_KEYS),
            "budgets": build_effect_budgets(),
            "high_level_order": list(HIGH_LEVEL_EFFECT_ORDER),
        },
        "privacy": {
            "readable_sec_contact_public": False,
            "accessions_public": False,
            "source_or_response_bodies_public": False,
            "canonical_model_requests_public": False,
            "realized_response_metadata_public": False,
            "pre_release_science_values_public": False,
            "redacted_errors_only": True,
        },
    }


# This literal is filled from the exact unsigned manifest above.  Its import-time
# assertion makes accidental edits a hard failure rather than a new contract.
CONTRACT_MANIFEST_SHA256: Final[str] = (
    "7d8c029679451e4a4b84feb90d497c17f56a5ebd626e54e89366831f70249df1"
)


def build_contract_manifest() -> dict[str, Any]:
    unsigned = _build_unsigned_contract_manifest()
    observed = canonical_sha256(unsigned)
    if observed != CONTRACT_MANIFEST_SHA256:
        _reject("v39_contract_manifest_literal")
    return {**unsigned, "contract_manifest_sha256": observed}


def validate_contract_manifest(value: Any) -> dict[str, Any]:
    """Accept only the exact self-hashed v3.9 manifest."""

    if not isinstance(value, Mapping):
        _reject("v39_contract_manifest_shape")
    candidate = validate_self_sha256(value, field="contract_manifest_sha256")
    expected = build_contract_manifest()
    if canonical_json_bytes(candidate) != canonical_json_bytes(expected):
        _reject("v39_contract_manifest_mismatch")
    return copy.deepcopy(expected)


__all__ = [
    "BASE_COMMIT",
    "BASE_PARENT",
    "BASE_TREE",
    "BRANCH_NAME",
    "COMPARISON_PATH",
    "CONTRACT_MANIFEST_SHA256",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_VERSION",
    "CONTINUATION_PREREGISTRATION_PATH",
    "ContractViolation",
    "DEVELOPMENT_ATTEMPT_ID",
    "DEVELOPMENT_BLOCKS",
    "DEVELOPMENT_COMMAND",
    "DEVELOPMENT_DOCUMENT_COUNT",
    "DEVELOPMENT_PILOT_COUNT",
    "DEVELOPMENT_REMAINING_COUNT",
    "EFFECT_COUNT_KEYS",
    "EXPERIMENT_FAMILY_SEC_REQUEST_COUNT",
    "FROZEN_SCIENCE_PROJECTION_SHA256",
    "HIGH_LEVEL_EFFECT_ORDER",
    "IMPLEMENTATION_ALLOWED_PATHS",
    "IMPLEMENTATION_PRODUCTION_PATHS",
    "IMPLEMENTATION_TEST_PATHS",
    "MODEL_ACTIVE_FROM_BLOB_COUNT",
    "MODEL_CONFIG_DIGEST",
    "MODEL_CONTEXT_TOKENS",
    "MODEL_INPUT_MAX_BYTES",
    "MODEL_INPUT_MAX_SENTENCES",
    "MODEL_INPUT_MAX_SENTENCE_CHARACTERS",
    "MODEL_LAYER_DIGESTS",
    "MODEL_MANIFEST_SHA256",
    "MODEL_NAME",
    "MODEL_OUTPUT_TOKENS",
    "MODEL_RESPONSE_MAX_BYTES",
    "MODEL_SEED",
    "MODEL_TEMPERATURE",
    "NORMAL_EFFECT_BUDGET",
    "NORMAL_IDENTITY_REQUEST_COUNT",
    "OLLAMA_CHAT_ENDPOINT",
    "OLLAMA_SHOW_ENDPOINT",
    "OLLAMA_VERSION",
    "OLLAMA_VERSION_ENDPOINT",
    "PAUSED_RESUMED_EFFECT_BUDGET",
    "PAUSED_RESUMED_IDENTITY_REQUEST_COUNT",
    "PAUSE_ARTIFACT_PATH",
    "PILOT_PAUSE_EFFECT_BUDGET",
    "PILOT_PROJECTED_THRESHOLD_NS",
    "PREFLIGHT_ARTIFACT_PATH",
    "PREREGISTRATION_COMMIT",
    "PREREGISTRATION_GIT_BLOB_SHA1",
    "PREREGISTRATION_LITERAL_BYTES",
    "PREREGISTRATION_LITERAL_SHA256",
    "PREREGISTRATION_PATH",
    "PREREGISTRATION_TREE",
    "PRIVATE_DEVELOPMENT_NAMESPACE",
    "PRIVATE_NAMESPACE",
    "PRIVATE_PREFLIGHT_NAMESPACE",
    "PROMPT_SHA256",
    "RESULT_ARTIFACT_PATH",
    "RUNTIME_FINGERPRINT_SHA256",
    "RUNTIME_MODEL_INFO_SHA256",
    "RUNTIME_PROBE_ORDER",
    "RUNTIME_SHOW_SEMANTIC_EXCLUDED_KEYS",
    "RUNTIME_SHOW_SEMANTIC_SHA256",
    "RUNTIME_VERSION_RESPONSE_SHA256",
    "SCHEMA_SHA256",
    "SCIENCE_EVENT_ORDER",
    "SCIENCE_PROJECTION_BYTE_COUNT",
    "SCIENCE_PROJECTION_KEYS",
    "SCIENCE_PROJECTION_SHA256",
    "SCIENTIFIC_CONTRACT_GIT_BLOB_SHA1",
    "SCIENTIFIC_CONTRACT_INTERNAL_SHA256",
    "SCIENTIFIC_CONTRACT_LITERAL_SHA256",
    "SCIENTIFIC_CONTRACT_PATH",
    "SCIENTIFIC_PARENT_COMMIT",
    "SCIENTIFIC_PARENT_TREE",
    "SOURCE_PROJECTION_ORDER",
    "V38_ACQUISITION_MODULE_GIT_BLOB_SHA1",
    "V38_ACQUISITION_MODULE_LITERAL_SHA256",
    "V38_ACQUISITION_MODULE_PATH",
    "V38_CHECKPOINT_FILE_SHA256",
    "V38_COMPACT_REPLAY_SHA256",
    "V38_INVENTORY_BYTE_COUNT",
    "V38_INVENTORY_FILE_COUNT",
    "V38_INVENTORY_SHA256",
    "V38_LOGICAL_CHECKPOINT_SHA256",
    "V38_ROLE_COUNTS",
    "V38_ROLE_MANIFEST_INVENTORY_SHA256",
    "V38_ROLE_PLAN_SHA256",
    "V38_SEC_REQUEST_COUNT",
    "V38_SOURCE_AUTHORITY_BLOB_SHA1",
    "V38_SOURCE_AUTHORITY_LITERAL_SHA256",
    "V38_SOURCE_AUTHORITY_PATH",
    "V38_SOURCE_MODULE_GIT_BLOB_SHA1",
    "V38_SOURCE_MODULE_LITERAL_SHA256",
    "V38_SOURCE_MODULE_PATH",
    "V38_STAGE_SOURCE_SEAL_SHA256",
    "V38_TERMINAL_BLOB_SHA1",
    "V38_TERMINAL_INTERNAL_SHA256",
    "V38_TERMINAL_LITERAL_SHA256",
    "V38_TERMINAL_PATH",
    "YAHOO_ENDPOINT",
    "YAHOO_MAX_BATCH_SECONDS",
    "YAHOO_MAX_RESPONSE_BYTES",
    "YAHOO_MAX_TOTAL_RESPONSE_BYTES",
    "YAHOO_PROVIDER_SYMBOLS",
    "YAHOO_QUERY_ITEMS",
    "YAHOO_REQUEST_COUNT",
    "YAHOO_REQUEST_TIMEOUT_SECONDS",
    "YAHOO_SYMBOL_ORDER",
    "YAHOO_URLS",
    "YAHOO_USER_AGENT",
    "ZERO_EFFECT_BUDGET",
    "add_self_sha256",
    "build_contract_manifest",
    "build_effect_budgets",
    "build_frozen_science_projection",
    "build_source_authority_pins",
    "canonical_json_bytes",
    "canonical_sha256",
    "pilot_pause_required",
    "project_science_manifest",
    "projected_pilot_ns",
    "sha256_bytes",
    "validate_command",
    "validate_continuation_ancestry",
    "validate_contract_manifest",
    "validate_effect_counts",
    "validate_implementation_ancestry",
    "validate_pause_ancestry",
    "validate_preflight_ancestry",
    "validate_preregistration_ancestry",
    "validate_result_ancestry",
    "validate_self_sha256",
    "verify_frozen_science_projection",
]
