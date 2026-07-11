"""Pure, deliberately non-authorizing SEC/Gemma stage-evidence audit.

This module is the narrow bridge between the evidence components and the
effectful reveal store.  It replays the parts that can currently be proved
from detached bytes and records the remaining trust gaps explicitly.  It does
*not* return ``SemanticPrerequisiteValidation`` and therefore cannot unlock a
holdout stage.  The fail-closed boundary is intentional: several required
production attestations do not yet have authoritative component APIs.

No function in this module performs filesystem, network, model, market, SEC,
or outcome I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
import copy
import hashlib
import hmac
import json
import math
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_artifact_sealer import (
    validate_prediction_artifact_seal_receipt,
)
from agent_benchmark.sec_filing_gemma_contract import (
    CANDIDATE_IDS,
    MAX_FIT_SECONDS,
    MAX_MODEL_SECONDS,
    MAX_RUNTIME_SECONDS,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    REQUIRED_SOURCE_HASHES,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    STAGE_MODEL_CALL_CAPS,
    STAGE_ORDER,
    SecFilingGemmaContractError,
    canonical_sha256,
    validate_calendar_source_evidence_manifest,
    validate_candidate_manifest,
    validate_contract_manifest,
    validate_corpus_universe_manifest,
    validate_stage_content_manifest,
)
from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_filing_gemma_market_source_bytes import (
    validate_market_source_bytes_against_stage,
)
from agent_benchmark.sec_filing_gemma_no_leverage import (
    validate_sec_gemma_no_leverage_proof,
)
from agent_benchmark.sec_filing_gemma_ollama import (
    validate_ollama_model_attempt_receipt,
    validate_runtime_identity_guard,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
    validate_prediction_prefix,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    build_single_candidate_reveal_request,
    candidate_design_sha256,
    validate_reveal_registry,
)
from agent_benchmark.sec_filing_gemma_scoring import (
    validate_development_ranking_receipt,
    validate_score_receipt,
    validate_stage_gate_receipt,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    validate_stage_access_manifest,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


STAGE_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-evidence-audit-v1"
)
STAGE_AUDIT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-evidence-audit-receipt-v1"
)
STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-runtime-receipt-v1"
)
AUTHORITATIVE_VALIDATOR_ID: Final[str] = (
    "aapl-sec-gemma-authoritative-stage-verifier-v1"
)
OWNED_HARDENED_TRANSPORT_MODE: Final[str] = (
    "owned_hardened_loopback_session_requires_stage_attestation"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PREREQUISITE_TRANSITIONS: Final[dict[str, str]] = {
    "development": "intermediate",
    "intermediate": "final",
}
_BLOCKING_GAPS: Final[dict[str, str]] = {
    "artifact_seal_cas": (
        "only the latest seal transition is replayed; the envelope does not yet "
        "chain every artifact and receipt forward from deterministic genesis"
    ),
    "artifact_replay": (
        "the envelope does not yet replay SEC catalog and document bytes, "
        "preprocessing, extraction and feature proofs, labels, and the prelabel ledger"
    ),
    "calendar_source_semantics": (
        "official calendar bytes are hash-bound but no pure parser proves the "
        "frozen session semantics from those bytes"
    ),
    "chronology": (
        "prediction-row chronology is structurally replayed, but authoritative "
        "event bindings and training rows are not yet derived from raw corpus, "
        "feature, and matured-label evidence"
    ),
    "market_source_byte_reconciliation": (
        "snapshot-to-stage values replay, but provider-response-to-snapshot "
        "acquisition provenance has no authoritative receipt"
    ),
    "model_attempt_replay": (
        "exact attempt bytes replay, but the independently expected payload is "
        "not yet derived from authoritative preprocessing and corpus evidence"
    ),
    "prediction_replay": (
        "policy-prefix transitions and learner arithmetic replay, but event "
        "bindings and training matrices are still supplied by the evidence envelope"
    ),
    "prerequisite_evidence_identity": (
        "the directly invoked verifier envelope still omits authoritative raw "
        "corpus, content, feature, label, and full seal-chain pins, and no "
        "downstream API requires the consumed authorization-entry hash"
    ),
    "runtime_budget": (
        "stage-specific measurements are internally reconciled but the owned "
        "transport and monotonic runtime are not independently attested"
    ),
    "source_identity": (
        "source bytes match candidate hashes, but candidate source roles do not "
        "yet bind canonical repository paths and role semantics"
    ),
    "stage_access_identity": (
        "the access plan binds carry-in normalized-text hashes, byte counts, and "
        "content records, but does not replay the actual normalized-text bytes "
        "and downstream APIs do not require the consumed authorization-entry hash"
    ),
    "zero_cost": (
        "loopback and zero-cost receipt fields replay, but independent network "
        "transport attestation is still absent"
    ),
}
_INTERMEDIATE_STAGE_IDENTITY_GAP: Final[str] = (
    "the intermediate evidence envelope supplies only an opaque parent evidence "
    "hash; it does not replay the development winner or bind the parent learner "
    "output state to the intermediate learner input state"
)
_LEARNER_FIT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_sha256",
        "head_variant",
        "fold_id",
        "train_label_maturity_through",
        "training_set_sha256",
        "training_row_count",
        "maximum_training_label_maturity_session",
        "feature_schema_sha256",
    }
)
_LEARNER_FOLD_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fold_train_cutoff_session",
        "training_set_count",
        "training_positive_count",
        "training_set_membership_sha256",
        "semantic_training_feature_matrix_sha256",
        "ablation_training_feature_matrix_sha256",
        "training_binary_target_sha256",
        "training_edge_target_sha256",
        "training_set_max_label_maturity_session",
        "semantic_fold_state_sha256",
        "ablation_fold_state_sha256",
    }
)


class SecFilingGemmaStageVerifierError(ValueError):
    """Detached stage evidence is invalid or cannot authorize a reveal."""


class SecFilingGemmaStageVerifierBlocked(SecFilingGemmaStageVerifierError):
    """All replayable evidence passed, but authoritative promotion is blocked."""


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageVerifierError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise SecFilingGemmaStageVerifierError(
            f"{location} keys changed; missing={missing}, extra={extra}"
        )


def _plain(value: Any, location: str) -> Any:
    """Take one detached finite-JSON snapshot and reject custom containers."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaStageVerifierError(f"{location} is non-finite")
        return value
    if type(value) is list:
        return [_plain(child, f"{location}[{index}]") for index, child in enumerate(value)]
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise SecFilingGemmaStageVerifierError(
                    f"{location} contains a non-string key"
                )
            result[key] = _plain(child, f"{location}.{key}")
        return result
    raise SecFilingGemmaStageVerifierError(
        f"{location} must contain detached plain JSON values"
    )


def _mapping_snapshot(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SecFilingGemmaStageVerifierError(f"{location} must be a mapping")
    # MappingProxyType is used by the reveal store.  Detach it once, then reject
    # all custom nested containers before any security-sensitive replay.
    try:
        detached = _plain(dict(value), location)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            f"{location} could not be detached safely"
        ) from exc
    if type(detached) is not dict:
        raise SecFilingGemmaStageVerifierError(f"{location} must be an object")
    return detached


def _decode_base64(value: Any, location: str, *, maximum: int = 2_000_000_000) -> bytes:
    if not isinstance(value, str) or not value:
        raise SecFilingGemmaStageVerifierError(
            f"{location} must be nonempty canonical Base64"
        )
    try:
        payload = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            f"{location} is not strict Base64"
        ) from exc
    if not payload or len(payload) > maximum or base64.b64encode(payload).decode("ascii") != value:
        raise SecFilingGemmaStageVerifierError(
            f"{location} is empty, oversized, or noncanonical"
        )
    return payload


def _strict_json_bytes(payload: bytes, location: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = json.loads(text)
        canonical = json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            f"{location} is not canonical JSON bytes"
        ) from exc
    if type(parsed) is not dict or canonical != payload:
        raise SecFilingGemmaStageVerifierError(
            f"{location} is not one canonical JSON object"
        )
    return _plain(parsed, location)


def validate_candidate_source_bytes(
    *,
    contract_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    source_bytes_base64_by_role: Mapping[str, str],
) -> dict[str, Any]:
    """Replay the frozen contract/candidate and every candidate source byte pin."""

    contract = _mapping_snapshot(contract_manifest, "contract manifest")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    sources = _mapping_snapshot(source_bytes_base64_by_role, "source bytes")
    try:
        contract_hash = validate_contract_manifest(contract)
        candidate_hash = validate_candidate_manifest(
            candidate, expected_candidate_sha256=expected_candidate_sha256
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Contract or candidate identity replay failed"
        ) from exc
    _expect_keys(sources, set(REQUIRED_SOURCE_HASHES), "source bytes")
    observed: dict[str, str] = {}
    for role in REQUIRED_SOURCE_HASHES:
        payload = _decode_base64(sources[role], f"source bytes {role}", maximum=32 * 1024 * 1024)
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(
            digest, candidate["bindings"]["source_hashes"][role]
        ):
            raise SecFilingGemmaStageVerifierError(
                f"Source bytes for {role} differ from the candidate pin"
            )
        observed[role] = digest
    return {
        "contract_sha256": contract_hash,
        "candidate_sha256": candidate_hash,
        "source_hashes_sha256": canonical_sha256(observed),
        "stage_verifier_source_sha256": observed["stage_verifier"],
    }


def validate_calendar_and_universe_snapshot(
    *,
    calendar_evidence_manifest: Mapping[str, Any],
    calendar_source_bytes_base64_by_name: Mapping[str, str],
    corpus_universe_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact source-byte hashes and the complete frozen universe.

    This proves byte identity, not the still-missing semantic parse from the
    official NYSE pages/PDFs into the session sequence.
    """

    calendar = _mapping_snapshot(calendar_evidence_manifest, "calendar evidence")
    source_bytes = _mapping_snapshot(
        calendar_source_bytes_base64_by_name, "calendar source bytes"
    )
    universe = _mapping_snapshot(corpus_universe_manifest, "corpus universe")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    records = calendar.get("source_records")
    if type(records) is not dict:
        raise SecFilingGemmaStageVerifierError("Calendar source records are invalid")
    _expect_keys(source_bytes, set(records), "calendar source bytes")
    for name, record in records.items():
        if type(record) is not dict:
            raise SecFilingGemmaStageVerifierError("Calendar source record is invalid")
        payload = _decode_base64(
            source_bytes[name], f"calendar source bytes {name}", maximum=64 * 1024 * 1024
        )
        if (
            hashlib.sha256(payload).hexdigest() != record.get("content_sha256")
            or len(payload) != record.get("byte_count")
        ):
            raise SecFilingGemmaStageVerifierError(
                f"Calendar source bytes for {name} do not match the manifest"
            )
    try:
        calendar_hash = validate_calendar_source_evidence_manifest(
            calendar,
            expected_calendar_source_evidence_sha256=candidate["bindings"][
                "calendar_source_evidence_sha256"
            ],
        )
        universe_hash = validate_corpus_universe_manifest(
            universe,
            session_dates=EXPECTED_SESSIONS,
            expected_universe_sha256=candidate["bindings"]["corpus_universe_sha256"],
            require_complete_coverage=True,
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Calendar or complete-universe replay failed"
        ) from exc
    for observed, expected, label in (
        (universe.get("calendar_artifact_sha256"), calendar_hash, "calendar artifact"),
        (
            universe.get("calendar_sessions_sha256"),
            candidate["bindings"]["calendar_sessions_sha256"],
            "calendar sessions",
        ),
        (
            universe.get("catalog_artifact_sha256"),
            candidate["bindings"]["sec_catalog_artifact_sha256"],
            "SEC catalog",
        ),
        (
            universe.get("universe_semantic_sha256"),
            candidate["bindings"]["corpus_universe_semantic_sha256"],
            "universe semantics",
        ),
    ):
        if observed != expected:
            raise SecFilingGemmaStageVerifierError(
                f"Corpus {label} differs from the candidate"
            )
    return {
        "calendar_source_evidence_sha256": calendar_hash,
        "corpus_universe_sha256": universe_hash,
        "calendar_source_bytes_sha256": canonical_sha256(
            {
                name: hashlib.sha256(
                    _decode_base64(source_bytes[name], f"calendar {name}")
                ).hexdigest()
                for name in sorted(source_bytes)
            }
        ),
    }


def validate_model_attempt_batch(
    batch: Mapping[str, Any],
    *,
    candidate_manifest: Mapping[str, Any],
    expected_accessions: Sequence[str],
) -> dict[str, Any]:
    """Replay exact request/response bytes and one before/after runtime guard."""

    value = _mapping_snapshot(batch, "model batch")
    _expect_keys(
        value,
        {
            "stage",
            "before_runtime_evidence",
            "after_runtime_evidence",
            "runtime_guard",
            "attempts",
        },
        "model batch",
    )
    stage = value["stage"]
    if stage not in STAGE_ORDER:
        raise SecFilingGemmaStageVerifierError("Model batch stage is invalid")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    attempts = value["attempts"]
    if type(attempts) is not list or not attempts:
        raise SecFilingGemmaStageVerifierError("Model batch attempts must be nonempty")
    expected = list(expected_accessions)
    if len(attempts) != len(expected) or len(attempts) > STAGE_MODEL_CALL_CAPS[stage]:
        raise SecFilingGemmaStageVerifierError(
            "Model batch count differs from exact stage coverage or exceeds its cap"
        )
    before = value["before_runtime_evidence"]
    after = value["after_runtime_evidence"]
    if type(before) is not dict or type(after) is not dict:
        raise SecFilingGemmaStageVerifierError("Runtime probes must be detached objects")
    runtime_evidence_hash = canonical_sha256(before)
    receipt_hashes: list[str] = []
    observed_accessions: list[str] = []
    elapsed = 0
    for index, item in enumerate(attempts):
        if type(item) is not dict:
            raise SecFilingGemmaStageVerifierError("Model attempt item must be an object")
        _expect_keys(
            item,
            {
                "accession_number",
                "expected_model_payload",
                "expected_sentence_ids",
                "receipt",
            },
            f"model attempts[{index}]",
        )
        accession = item["accession_number"]
        if accession != expected[index]:
            raise SecFilingGemmaStageVerifierError(
                "Model attempt coverage is omitted, duplicated, or reordered"
            )
        payload = item["expected_model_payload"]
        sentence_ids = item["expected_sentence_ids"]
        if type(payload) is not dict or type(sentence_ids) is not list:
            raise SecFilingGemmaStageVerifierError(
                "Model attempt expected payload or sentence IDs are not detached"
            )
        try:
            receipt = validate_ollama_model_attempt_receipt(
                item["receipt"],
                expected_candidate_sha256=candidate["candidate_sha256"],
                expected_model_payload_sha256=canonical_sha256(payload),
                expected_sentence_ids=sentence_ids,
                expected_runtime_evidence_sha256=runtime_evidence_hash,
                expected_model_digest=candidate["model"]["digest"],
                expected_runtime_fingerprint_sha256=candidate["model"][
                    "runtime_fingerprint_sha256"
                ],
                expected_transport_mode=OWNED_HARDENED_TRANSPORT_MODE,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SecFilingGemmaStageVerifierError(
                f"Model attempt {accession} did not replay from exact bytes"
            ) from exc
        if json.loads(receipt.request_bytes.decode("utf-8")) != payload:
            raise SecFilingGemmaStageVerifierError(
                "Model receipt request differs from the independent payload"
            )
        receipt_hashes.append(receipt.receipt_sha256)
        observed_accessions.append(accession)
        elapsed += receipt.elapsed_nanoseconds
    guard = value["runtime_guard"]
    if type(guard) is not dict:
        raise SecFilingGemmaStageVerifierError("Runtime guard must be detached")
    try:
        guard_hash = validate_runtime_identity_guard(
            guard,
            before_evidence=before,
            after_evidence=after,
            expected_model_digest=candidate["model"]["digest"],
            expected_runtime_fingerprint_sha256=candidate["model"][
                "runtime_fingerprint_sha256"
            ],
            stage=stage,
            model_call_receipt_sha256s=receipt_hashes,
            expected_runtime_guard_sha256=guard.get("runtime_guard_sha256"),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Before/after runtime guard or exact receipt order did not replay"
        ) from exc
    return {
        "stage": stage,
        "accessions_sha256": canonical_sha256(observed_accessions),
        "model_attempt_receipt_sha256s": receipt_hashes,
        "runtime_guard_sha256": guard_hash,
        "model_elapsed_nanoseconds": elapsed,
    }


def validate_market_snapshot_stage_replay(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replay canonical snapshot bytes into one structural market stage."""

    replay = _mapping_snapshot(value, "market replay")
    _expect_keys(
        replay,
        {
            "stage",
            "source_manifest",
            "market_stage_manifest",
            "artifact_bytes_base64_by_symbol",
            "window_bytes_base64_by_symbol",
        },
        "market replay",
    )
    artifacts = {
        symbol: _decode_base64(encoded, f"market artifact {symbol}")
        for symbol, encoded in _mapping_snapshot(
            replay["artifact_bytes_base64_by_symbol"], "market artifact bytes"
        ).items()
    }
    windows = {
        symbol: _decode_base64(encoded, f"market window {symbol}")
        for symbol, encoded in _mapping_snapshot(
            replay["window_bytes_base64_by_symbol"], "market window bytes"
        ).items()
    }
    source = replay["source_manifest"]
    stage_manifest = replay["market_stage_manifest"]
    if type(source) is not dict or type(stage_manifest) is not dict:
        raise SecFilingGemmaStageVerifierError("Market manifests must be detached")
    try:
        return validate_market_source_bytes_against_stage(
            artifact_bytes_by_symbol=artifacts,
            window_bytes_by_symbol=windows,
            source_manifest=source,
            stage_manifest=stage_manifest,
            expected_artifact_stage=replay["stage"],
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_market_stage_manifest_sha256=stage_manifest[
                "market_stage_manifest_sha256"
            ],
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Market snapshot-to-stage replay failed"
        ) from exc


def validate_prediction_artifact_replay(
    value: Mapping[str, Any],
    *,
    candidate_sha256: str,
    corpus_universe_sha256: str,
) -> dict[str, Any]:
    """Validate exact seal CAS bytes and replay every prediction transition."""

    replay = _mapping_snapshot(value, "prediction replay")
    _expect_keys(
        replay,
        {
            "artifact_bytes_base64",
            "seal_receipt_bytes_base64",
            "external_prior_pin_bytes_base64",
            "external_next_pin_bytes_base64",
            "event_bindings",
        },
        "prediction replay",
    )
    artifact = _decode_base64(replay["artifact_bytes_base64"], "prediction artifact")
    seal_receipt = _decode_base64(
        replay["seal_receipt_bytes_base64"], "prediction seal receipt", maximum=256 * 1024
    )
    prior_pin = _decode_base64(
        replay["external_prior_pin_bytes_base64"], "prediction prior pin", maximum=256 * 1024
    )
    next_pin = _decode_base64(
        replay["external_next_pin_bytes_base64"], "prediction next pin", maximum=256 * 1024
    )
    try:
        seal = validate_prediction_artifact_seal_receipt(
            artifact_bytes=artifact,
            receipt_bytes=seal_receipt,
            external_prior_pin_bytes=prior_pin,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Prediction artifact seal CAS replay failed"
        ) from exc
    if seal.next_external_pin_bytes != next_pin:
        raise SecFilingGemmaStageVerifierError(
            "Prediction seal does not match the externally pinned next state"
        )
    prefix = _strict_json_bytes(artifact, "prediction artifact")
    events = replay["event_bindings"]
    if type(events) is not list:
        raise SecFilingGemmaStageVerifierError("Prediction event bindings must be a list")
    try:
        summary = validate_prediction_prefix(
            prefix,
            session_dates=EXPECTED_SESSIONS,
            expected_calendar_sessions_sha256=prefix["calendar_sessions_sha256"],
            expected_candidate_sha256=candidate_sha256,
            expected_corpus_universe_sha256=corpus_universe_sha256,
            expected_event_bindings=events,
            expected_prediction_prefix_sha256=seal.prediction_prefix_sha256,
            expected_tip_sha256=seal.prediction_tip_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Prediction chronology, policy state, or prefix ancestry failed replay"
        ) from exc
    if seal.candidate_sha256 != candidate_sha256:
        raise SecFilingGemmaStageVerifierError("Prediction artifact belongs to another candidate")
    return {
        **summary,
        "artifact_sha256": seal.artifact_sha256,
        "seal_sha256": seal.seal_sha256,
        "seal_receipt_sha256": seal.receipt_sha256,
        "stage": seal.stage,
        "prefix": prefix,
    }


def _decode_matrix(value: Any, location: str) -> list[list[float]]:
    if type(value) is not list or not value:
        raise SecFilingGemmaStageVerifierError(f"{location} must be a nonempty matrix")
    result: list[list[float]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if type(row) is not list or not row:
            raise SecFilingGemmaStageVerifierError(f"{location}[{row_index}] is invalid")
        decoded: list[float] = []
        for column_index, item in enumerate(row):
            if not isinstance(item, str):
                raise SecFilingGemmaStageVerifierError(f"{location} must use float.hex")
            try:
                number = float.fromhex(item)
            except ValueError as exc:
                raise SecFilingGemmaStageVerifierError(f"{location} has invalid float.hex") from exc
            if not math.isfinite(number) or number.hex() != item:
                raise SecFilingGemmaStageVerifierError(f"{location} has noncanonical float.hex")
            decoded.append(number)
        if width is None:
            width = len(decoded)
        elif len(decoded) != width:
            raise SecFilingGemmaStageVerifierError(f"{location} is ragged")
        result.append(decoded)
    return result


def validate_learner_refit_replays(
    replays: Sequence[Mapping[str, Any]],
    *,
    candidate_sha256: str,
    prediction_prefix: Mapping[str, Any],
) -> dict[str, Any]:
    """Refit every supplied fold/variant and replay its row predictions exactly."""

    if isinstance(replays, (str, bytes)) or not isinstance(replays, Sequence) or not replays:
        raise SecFilingGemmaStageVerifierError("Learner replays must be a nonempty sequence")
    prefix_rows = prediction_prefix.get("rows")
    if type(prefix_rows) is not list:
        raise SecFilingGemmaStageVerifierError("Prediction prefix rows are invalid")
    rows_by_hash = {row["prediction_row_sha256"]: row for row in prefix_rows}
    available = [
        row for row in prefix_rows if row.get("prediction_status") == AVAILABLE_PREDICTION_STATUS
    ]
    contexts_by_fold: dict[str, dict[str, Any]] = {}
    for row in available:
        fold_id = row.get("fold_id")
        context = row.get("fold_context")
        if not isinstance(fold_id, str) or type(context) is not dict:
            raise SecFilingGemmaStageVerifierError(
                "Available prediction row lacks a detached fold context"
            )
        _expect_keys(context, set(_LEARNER_FOLD_CONTEXT_KEYS), "prediction fold context")
        prior = contexts_by_fold.setdefault(fold_id, context)
        if prior != context:
            raise SecFilingGemmaStageVerifierError(
                "Available predictions disagree about their fixed fold context"
            )
    seen_cases: set[tuple[str, str]] = set()
    seen_fits: set[tuple[str, str]] = set()
    state_hashes: dict[str, dict[str, str]] = {}
    input_identity_hashes: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(replays):
        item = _mapping_snapshot(raw, f"learner replays[{index}]")
        _expect_keys(
            item,
            {
                "fold_id",
                "variant",
                "feature_names",
                "training_features_hex",
                "binary_targets",
                "edge_targets_hex",
                "fit_metadata",
                "expected_state",
                "prediction_cases",
            },
            f"learner replays[{index}]",
        )
        variant = item["variant"]
        if variant not in {"semantic", "ablation"}:
            raise SecFilingGemmaStageVerifierError("Learner variant is invalid")
        fit_key = (item["fold_id"], variant)
        if fit_key in seen_fits:
            raise SecFilingGemmaStageVerifierError(
                "Learner fold/variant refit is duplicated"
            )
        seen_fits.add(fit_key)
        features = _decode_matrix(item["training_features_hex"], "training features")
        edges_matrix = _decode_matrix(
            [[edge] for edge in item["edge_targets_hex"]], "edge targets"
        )
        edges = [row[0] for row in edges_matrix]
        targets = item["binary_targets"]
        names = item["feature_names"]
        metadata = item["fit_metadata"]
        if type(targets) is not list or type(names) is not list or type(metadata) is not dict:
            raise SecFilingGemmaStageVerifierError("Learner inputs are not detached")
        _expect_keys(metadata, set(_LEARNER_FIT_METADATA_KEYS), "learner fit metadata")
        if any(type(target) is not int or target not in {0, 1} for target in targets):
            raise SecFilingGemmaStageVerifierError(
                "Learner binary targets must be canonical integer zero or one"
            )
        if metadata.get("candidate_sha256") != candidate_sha256 or metadata.get(
            "head_variant"
        ) != variant or metadata.get("fold_id") != item["fold_id"]:
            raise SecFilingGemmaStageVerifierError("Learner metadata crossed a binding")
        context = contexts_by_fold.get(item["fold_id"])
        if context is None:
            raise SecFilingGemmaStageVerifierError(
                "Learner refit has no available prediction fold context"
            )
        feature_matrix_hash = canonical_sha256(item["training_features_hex"])
        binary_target_hash = canonical_sha256(targets)
        edge_target_hash = canonical_sha256(item["edge_targets_hex"])
        if metadata.get("training_set_sha256") != context.get(
            "training_set_membership_sha256"
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training membership differs from the prediction fold context"
            )
        if feature_matrix_hash != context.get(
            f"{variant}_training_feature_matrix_sha256"
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training feature matrix differs from the prediction fold context"
            )
        if (
            binary_target_hash != context.get("training_binary_target_sha256")
            or edge_target_hash != context.get("training_edge_target_sha256")
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training targets differ from the prediction fold context"
            )
        if (
            metadata.get("training_row_count") != len(features)
            or context.get("training_set_count") != len(features)
            or context.get("training_positive_count") != sum(targets)
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training counts differ from the prediction fold context"
            )
        if (
            metadata.get("train_label_maturity_through")
            != context.get("fold_train_cutoff_session")
            or metadata.get("maximum_training_label_maturity_session")
            != context.get("training_set_max_label_maturity_session")
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training maturity differs from the prediction fold context"
            )
        input_identity = {
            "training_set_membership_sha256": metadata["training_set_sha256"],
            "training_feature_matrix_sha256": feature_matrix_hash,
            "training_binary_target_sha256": binary_target_hash,
            "training_edge_target_sha256": edge_target_hash,
            "feature_schema_sha256": metadata["feature_schema_sha256"],
            "training_row_count": len(features),
            "training_positive_count": sum(targets),
            "fold_train_cutoff_session": metadata["train_label_maturity_through"],
            "training_set_max_label_maturity_session": metadata[
                "maximum_training_label_maturity_session"
            ],
        }
        input_identity_hashes.setdefault(item["fold_id"], {})[variant] = canonical_sha256(
            input_identity
        )
        try:
            learner = SecFilingGemmaTwoHeadLearner().fit(
                features,
                targets,
                edges,
                feature_names=names,
                fit_metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            raise SecFilingGemmaStageVerifierError("Deterministic learner refit failed") from exc
        state = item["expected_state"]
        if type(state) is not dict or learner.to_state() != state:
            raise SecFilingGemmaStageVerifierError(
                "Persisted learner state differs from deterministic refit"
            )
        state_hashes.setdefault(item["fold_id"], {})[variant] = learner.model_sha256
        cases = item["prediction_cases"]
        if type(cases) is not list:
            raise SecFilingGemmaStageVerifierError("Learner prediction cases must be a list")
        for case in cases:
            if type(case) is not dict:
                raise SecFilingGemmaStageVerifierError("Learner prediction case is invalid")
            _expect_keys(case, {"prediction_row_sha256", "features_hex"}, "learner case")
            row_hash = case["prediction_row_sha256"]
            row = rows_by_hash.get(row_hash)
            if row is None or row.get("fold_id") != item["fold_id"]:
                raise SecFilingGemmaStageVerifierError("Learner case crossed a prediction fold")
            case_key = (row_hash, variant)
            if case_key in seen_cases:
                raise SecFilingGemmaStageVerifierError("Learner prediction case is duplicated")
            seen_cases.add(case_key)
            matrix = _decode_matrix([case["features_hex"]], "prediction features")
            output = learner.predict_components(matrix)
            probability = float(output["cash_win_probability_10bps"][0]).hex()
            edge = float(output["expected_active_log_edge_10bps"][0]).hex()
            if (
                row[f"{variant}_cash_probability_hex"] != probability
                or row[f"{variant}_expected_edge_hex"] != edge
            ):
                raise SecFilingGemmaStageVerifierError(
                    "Prediction row differs from deterministic learner output"
                )
    expected_cases = {
        (row["prediction_row_sha256"], variant)
        for row in available
        for variant in ("semantic", "ablation")
    }
    if seen_cases != expected_cases:
        raise SecFilingGemmaStageVerifierError(
            "Learner replays omit or add an available prediction"
        )
    expected_fits = {
        (row["fold_id"], variant)
        for row in available
        for variant in ("semantic", "ablation")
    }
    if seen_fits != expected_fits:
        raise SecFilingGemmaStageVerifierError(
            "Learner replays omit or add a fold/variant fit"
        )
    for row in available:
        fold_states = state_hashes.get(row["fold_id"], {})
        context = row["fold_context"]
        if (
            fold_states.get("semantic") != context["semantic_fold_state_sha256"]
            or fold_states.get("ablation") != context["ablation_fold_state_sha256"]
        ):
            raise SecFilingGemmaStageVerifierError(
                "Prediction fold context differs from deterministic learner states"
            )
    return {
        "learner_state_sha256s_by_fold": state_hashes,
        "learner_input_identity_sha256s_by_fold": input_identity_hashes,
        "prediction_case_count": len(seen_cases),
        "learner_replay_sha256": canonical_sha256(
            {
                "states": state_hashes,
                "inputs": input_identity_hashes,
                "cases": sorted(seen_cases),
            }
        ),
    }


def validate_raw_scores_gates_and_ranking(
    value: Mapping[str, Any],
    *,
    prediction_prefix: Mapping[str, Any],
    market_stage: Mapping[str, Any],
    prerequisite_stage: str,
) -> dict[str, Any]:
    """Replay every raw score before any gate or ranking receipt is inspected."""

    replay = _mapping_snapshot(value, "score replay")
    _expect_keys(
        replay,
        {
            "label_release_evidence",
            "evaluations",
            "ranking_receipt",
            "selected_candidate_id",
        },
        "score replay",
    )
    labels = replay["label_release_evidence"]
    evaluations = replay["evaluations"]
    selected = replay["selected_candidate_id"]
    if type(labels) is not dict or type(evaluations) is not list or selected not in CANDIDATE_IDS:
        raise SecFilingGemmaStageVerifierError("Score replay inputs are invalid")
    prefix_hash = prediction_prefix["prediction_prefix_sha256"]
    market_hash = market_stage["market_stage_manifest_sha256"]
    label_hash = labels.get("label_release_ledger_sha256")
    _sha256(label_hash, "label release ledger hash")
    raw_score_hashes: list[str] = []
    gate_receipts: list[dict[str, Any]] = []
    gate_hashes: list[str] = []
    evaluation_ids: list[str] = []
    no_leverage_hashes: list[str] = []
    for evaluation in evaluations:
        if type(evaluation) is not dict:
            raise SecFilingGemmaStageVerifierError("Score evaluation must be detached")
        _expect_keys(evaluation, {"candidate_id", "score_receipts", "gate_receipt"}, "score evaluation")
        candidate_id = evaluation["candidate_id"]
        if candidate_id not in CANDIDATE_IDS or candidate_id in evaluation_ids:
            raise SecFilingGemmaStageVerifierError("Score candidate is invalid or duplicated")
        receipts = evaluation["score_receipts"]
        if type(receipts) is not list:
            raise SecFilingGemmaStageVerifierError("Raw score receipts must be a list")
        candidate_hashes: list[str] = []
        for receipt in receipts:
            if type(receipt) is not dict:
                raise SecFilingGemmaStageVerifierError("Raw score receipt is invalid")
            config = receipt.get("configuration")
            if type(config) is not dict or config.get("selected_candidate_id") != candidate_id:
                raise SecFilingGemmaStageVerifierError("Raw score crossed a candidate binding")
            score_hash = _sha256(receipt.get("score_receipt_sha256"), "score receipt hash")
            try:
                validate_score_receipt(
                    receipt,
                    expected_score_receipt_sha256=score_hash,
                    prediction_prefix=prediction_prefix,
                    expected_prediction_prefix_sha256=prefix_hash,
                    label_release_evidence=labels,
                    expected_label_release_ledger_sha256=label_hash,
                    market_stage=market_stage,
                    expected_market_stage_manifest_sha256=market_hash,
                    selected_candidate_id=candidate_id,
                    selected_variant=config["selected_variant"],
                    cost_bps=config["cost_bps"],
                    stage=prerequisite_stage,
                    score_cutoff_session=config["score_cutoff_session"],
                    terminal_convention=config["terminal_convention"],
                )
                proof = validate_sec_gemma_no_leverage_proof(
                    receipt, expected_score_receipt_sha256=score_hash
                )
            except (TypeError, ValueError) as exc:
                raise SecFilingGemmaStageVerifierError(
                    "Raw score or independent no-leverage replay failed"
                ) from exc
            candidate_hashes.append(score_hash)
            raw_score_hashes.append(score_hash)
            no_leverage_hashes.append(proof["proof_sha256"])
        gate = evaluation["gate_receipt"]
        if type(gate) is not dict:
            raise SecFilingGemmaStageVerifierError("Gate receipt is invalid")
        gate_hash = _sha256(gate.get("gate_receipt_sha256"), "gate receipt hash")
        try:
            validate_stage_gate_receipt(
                gate,
                expected_gate_receipt_sha256=gate_hash,
                score_receipts=receipts,
                expected_score_receipt_sha256s=candidate_hashes,
                stage=prerequisite_stage,
            )
        except (TypeError, ValueError) as exc:
            raise SecFilingGemmaStageVerifierError(
                "Gate receipt differs from the already-replayed raw scores"
            ) from exc
        evaluation_ids.append(candidate_id)
        gate_receipts.append(gate)
        gate_hashes.append(gate_hash)
    ranking_hash: str | None = None
    if prerequisite_stage == "development":
        if evaluation_ids != list(CANDIDATE_IDS) or type(replay["ranking_receipt"]) is not dict:
            raise SecFilingGemmaStageVerifierError(
                "Development requires the complete frozen candidate grid and ranking"
            )
        ranking = replay["ranking_receipt"]
        ranking_hash = _sha256(ranking.get("ranking_receipt_sha256"), "ranking hash")
        try:
            validate_development_ranking_receipt(
                ranking,
                expected_ranking_receipt_sha256=ranking_hash,
                gate_receipts=gate_receipts,
                expected_gate_receipt_sha256s=gate_hashes,
            )
        except (TypeError, ValueError) as exc:
            raise SecFilingGemmaStageVerifierError(
                "Development ranking differs from the replayed gates"
            ) from exc
        if ranking.get("selected_candidate_id") != selected:
            raise SecFilingGemmaStageVerifierError("Selected candidate differs from ranking")
    else:
        if evaluation_ids != [selected] or replay["ranking_receipt"] is not None:
            raise SecFilingGemmaStageVerifierError(
                "Intermediate stage must evaluate only the frozen selected candidate"
            )
    selected_gate = gate_receipts[evaluation_ids.index(selected)]
    if selected_gate.get("passed") is not True:
        raise SecFilingGemmaStageVerifierError("Selected prerequisite gate did not pass")
    return {
        "selected_candidate_id": selected,
        "raw_score_receipt_sha256s": raw_score_hashes,
        "gate_receipt_sha256s": gate_hashes,
        "ranking_receipt_sha256": ranking_hash,
        "no_leverage_proof_sha256s": no_leverage_hashes,
    }


def validate_stage_runtime_receipt(
    receipt: Mapping[str, Any],
    *,
    prerequisite_stage: str,
    candidate_sha256: str,
    model_batch_summary: Mapping[str, Any],
) -> str:
    """Validate one stage-local diagnostic receipt; it is not an attestation."""

    value = _mapping_snapshot(receipt, "stage runtime receipt")
    _expect_keys(
        value,
        {
            "schema_version",
            "stage",
            "candidate_sha256",
            "phase_elapsed_seconds",
            "sec_request_count",
            "sec_response_bytes",
            "model_attempt_receipt_sha256s",
            "runtime_guard_sha256",
            "paid_api_calls",
            "estimated_cost_usd_hex",
            "pull_attempts",
            "retries",
            "repair_attempts",
            "runtime_receipt_sha256",
        },
        "stage runtime receipt",
    )
    body = {key: value[key] for key in value if key != "runtime_receipt_sha256"}
    receipt_hash = _sha256(value["runtime_receipt_sha256"], "runtime receipt hash")
    if canonical_sha256(body) != receipt_hash:
        raise SecFilingGemmaStageVerifierError("Runtime receipt is not canonical")
    if (
        value["schema_version"] != STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION
        or value["stage"] != prerequisite_stage
        or value["candidate_sha256"] != candidate_sha256
        or value["model_attempt_receipt_sha256s"]
        != model_batch_summary["model_attempt_receipt_sha256s"]
        or value["runtime_guard_sha256"] != model_batch_summary["runtime_guard_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError("Runtime receipt crossed a stage binding")
    phases = value["phase_elapsed_seconds"]
    if type(phases) is not dict or set(phases) != {"sec", "model", "fit_simulation_sealing", "total"}:
        raise SecFilingGemmaStageVerifierError("Runtime phases are invalid")
    decoded: dict[str, float] = {}
    for name, encoded in phases.items():
        if not isinstance(encoded, str):
            raise SecFilingGemmaStageVerifierError("Runtime seconds must use float.hex")
        try:
            number = float.fromhex(encoded)
        except ValueError as exc:
            raise SecFilingGemmaStageVerifierError("Runtime seconds are invalid") from exc
        if not math.isfinite(number) or number < 0.0 or number.hex() != encoded:
            raise SecFilingGemmaStageVerifierError("Runtime seconds are noncanonical")
        decoded[name] = number
    if (
        decoded["sec"] > MAX_SEC_SECONDS
        or decoded["model"] > MAX_MODEL_SECONDS
        or decoded["fit_simulation_sealing"] > MAX_FIT_SECONDS
        or decoded["total"] > MAX_RUNTIME_SECONDS
        or decoded["total"] + 1e-9
        < decoded["sec"] + decoded["model"] + decoded["fit_simulation_sealing"]
    ):
        raise SecFilingGemmaStageVerifierError("Runtime budget was exceeded")
    model_elapsed_nanoseconds = model_batch_summary.get("model_elapsed_nanoseconds")
    if (
        type(model_elapsed_nanoseconds) is not int
        or model_elapsed_nanoseconds < 0
        or decoded["model"] + 1e-9 < model_elapsed_nanoseconds / 1_000_000_000.0
    ):
        raise SecFilingGemmaStageVerifierError(
            "Runtime model phase is shorter than exact attempt diagnostics"
        )
    if type(value["sec_request_count"]) is not int or not 0 <= value["sec_request_count"] <= MAX_SEC_REQUESTS:
        raise SecFilingGemmaStageVerifierError("SEC request count is invalid")
    if type(value["sec_response_bytes"]) is not int or not 0 <= value["sec_response_bytes"] <= MAX_SEC_BYTES:
        raise SecFilingGemmaStageVerifierError("SEC response bytes are invalid")
    for field in ("paid_api_calls", "pull_attempts", "retries", "repair_attempts"):
        if type(value[field]) is not int or value[field] != 0:
            raise SecFilingGemmaStageVerifierError(f"{field} must be exactly zero")
    if value["estimated_cost_usd_hex"] != 0.0.hex():
        raise SecFilingGemmaStageVerifierError("Estimated cost must be exactly zero")
    return receipt_hash


def validate_registry_request_and_stage_access(
    *,
    registry: Mapping[str, Any],
    registry_external_pin: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
    expected_context: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    corpus_universe_manifest: Mapping[str, Any],
    prerequisite_content_manifest: Mapping[str, Any],
    expected_stage_evidence_sha256: str,
) -> dict[str, str]:
    """Replay registry/request/access identities without opening requested bytes."""

    registry_value = _mapping_snapshot(registry, "reveal registry")
    pin = _mapping_snapshot(registry_external_pin, "registry external pin")
    access = _mapping_snapshot(stage_access_manifest, "stage access manifest")
    context = _mapping_snapshot(expected_context, "expected reveal context")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    universe = _mapping_snapshot(corpus_universe_manifest, "corpus universe")
    prerequisite_content = _mapping_snapshot(
        prerequisite_content_manifest, "prerequisite content manifest"
    )
    required_context = {
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_sha256",
        "stage",
        "stage_access_manifest_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    }
    _expect_keys(context, required_context, "expected reveal context")
    prerequisite = context["prerequisite_stage"]
    requested = context["stage"]
    if _PREREQUISITE_TRANSITIONS.get(prerequisite) != requested:
        raise SecFilingGemmaStageVerifierError("Reveal transition is invalid")
    if context["prerequisite_stage_evidence_sha256"] != expected_stage_evidence_sha256:
        raise SecFilingGemmaStageVerifierError("Context binds another stage evidence artifact")
    content_hash = _sha256(
        prerequisite_content.get("content_manifest_sha256"),
        "prerequisite content manifest hash",
    )
    if prerequisite_content.get("artifact_stage") != prerequisite:
        raise SecFilingGemmaStageVerifierError(
            "Prerequisite content manifest belongs to another stage"
        )
    try:
        validate_stage_content_manifest(
            prerequisite_content,
            universe_manifest=universe,
            expected_content_manifest_sha256=content_hash,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Prerequisite content manifest failed exact universe replay"
        ) from exc
    try:
        registry_hash = validate_reveal_registry(registry_value, external_pin=pin)
        request = build_single_candidate_reveal_request(
            registry_value,
            external_pin=pin,
            candidate_manifest=candidate,
            stage=requested,
            stage_access_manifest_sha256=context["stage_access_manifest_sha256"],
            prerequisite_stage_evidence_sha256=expected_stage_evidence_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Registry or reveal request identity failed replay"
        ) from exc
    if request["request_sha256"] != context["request_sha256"]:
        raise SecFilingGemmaStageVerifierError("Context request hash is stale or substituted")
    for key in (
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "stage_access_manifest_sha256",
    ):
        if request[key] != context[key]:
            raise SecFilingGemmaStageVerifierError(f"Context field {key} changed")
    market_access = access.get("market_access")
    budgets = access.get("budgets")
    output = access.get("output")
    sec_plan = access.get("sec_access_plan")
    registry_section = access.get("registry")
    if not all(type(item) is dict for item in (market_access, budgets, output, sec_plan, registry_section)):
        raise SecFilingGemmaStageVerifierError("Stage access sections are invalid")
    sources = market_access["sources"]
    if type(sources) is not list:
        raise SecFilingGemmaStageVerifierError("Stage market sources are invalid")
    artifact_hashes = {item["symbol"]: item["artifact_sha256"] for item in sources}
    window_hashes = {item["symbol"]: item["window_sha256"] for item in sources}
    try:
        access_hash = validate_stage_access_manifest(
            access,
            expected_stage_access_manifest_sha256=context[
                "stage_access_manifest_sha256"
            ],
            prerequisite_stage=prerequisite,
            requested_stage=requested,
            candidate_manifest=candidate,
            expected_candidate_sha256=context["candidate_sha256"],
            expected_candidate_design_sha256=context["candidate_design_sha256"],
            expected_attempt_id=context["attempt_id"],
            expected_stage_verifier_source_sha256=candidate["bindings"][
                "source_hashes"
            ]["stage_verifier"],
            reveal_request=request,
            expected_reveal_request_sha256=context["request_sha256"],
            registry_sha256=context["registry_sha256"],
            registry_tip_sha256=context["registry_tip_sha256"],
            registered_entry_count=registry_section["registered_entry_count"],
            registry_entry_sha256=context["registry_entry_sha256"],
            base_corpus_universe_sha256=candidate["bindings"][
                "corpus_universe_sha256"
            ],
            corpus_universe_manifest=universe,
            prerequisite_content_manifest=prerequisite_content,
            expected_prerequisite_content_manifest_sha256=content_hash,
            session_calendar_sha256=candidate["bindings"][
                "calendar_sessions_sha256"
            ],
            authorized_documents=sec_plan["documents"],
            market_source_manifest_sha256=market_access[
                "source_manifest_sha256"
            ],
            market_source_artifact_sha256s=artifact_hashes,
            market_source_window_sha256s=window_hashes,
            max_sec_response_bytes=budgets["max_sec_response_bytes"],
            output_namespace=output["namespace"],
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Requested-stage access plan differs from exact non-authorizing replay"
        ) from exc
    return {
        "registry_sha256": registry_hash,
        "request_sha256": request["request_sha256"],
        "stage_access_manifest_sha256": access_hash,
    }


def _stage_evidence_hash(evidence: Mapping[str, Any]) -> str:
    observed = _sha256(evidence.get("stage_evidence_sha256"), "stage evidence hash")
    body = {key: evidence[key] for key in evidence if key != "stage_evidence_sha256"}
    calculated = canonical_sha256(body)
    if not hmac.compare_digest(observed, calculated):
        raise SecFilingGemmaStageVerifierError("Stage evidence is not canonical")
    return observed


def audit_stage_evidence(
    evidence: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
    expected_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay all currently provable evidence and return a blocked audit receipt."""

    value = _mapping_snapshot(evidence, "stage evidence")
    _expect_keys(
        value,
        {
            "schema_version",
            "prerequisite_stage",
            "parent_stage_evidence_sha256",
            "contract_manifest",
            "candidate_manifest",
            "source_bytes_base64_by_role",
            "calendar_evidence_manifest",
            "calendar_source_bytes_base64_by_name",
            "corpus_universe_manifest",
            "prerequisite_content_manifest",
            "model_batches_by_stage",
            "market_replays_by_stage",
            "prediction_replay",
            "learner_replays",
            "score_replay",
            "stage_runtime_receipt",
            "reveal_registry",
            "registry_external_pin",
            "stage_evidence_sha256",
        },
        "stage evidence",
    )
    if value["schema_version"] != STAGE_EVIDENCE_SCHEMA_VERSION:
        raise SecFilingGemmaStageVerifierError("Stage evidence schema changed")
    prerequisite = value["prerequisite_stage"]
    if prerequisite not in _PREREQUISITE_TRANSITIONS:
        raise SecFilingGemmaStageVerifierError("Only prerequisite stages are auditable")
    if prerequisite == "development":
        if value["parent_stage_evidence_sha256"] is not None:
            raise SecFilingGemmaStageVerifierError("Development cannot claim a parent stage")
    else:
        _sha256(value["parent_stage_evidence_sha256"], "parent stage evidence hash")
    evidence_hash = _stage_evidence_hash(value)
    candidate = value["candidate_manifest"]
    candidate_hash = candidate.get("candidate_sha256") if type(candidate) is dict else None
    _sha256(candidate_hash, "candidate hash")
    identities = validate_candidate_source_bytes(
        contract_manifest=value["contract_manifest"],
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate_hash,
        source_bytes_base64_by_role=value["source_bytes_base64_by_role"],
    )
    calendar = validate_calendar_and_universe_snapshot(
        calendar_evidence_manifest=value["calendar_evidence_manifest"],
        calendar_source_bytes_base64_by_name=value[
            "calendar_source_bytes_base64_by_name"
        ],
        corpus_universe_manifest=value["corpus_universe_manifest"],
        candidate_manifest=candidate,
    )
    stages = STAGE_ORDER[: STAGE_ORDER.index(prerequisite) + 1]
    batches = value["model_batches_by_stage"]
    markets = value["market_replays_by_stage"]
    if type(batches) is not dict or set(batches) != set(stages):
        raise SecFilingGemmaStageVerifierError("Model batch stage coverage is incomplete")
    if type(markets) is not dict or set(markets) != set(stages):
        raise SecFilingGemmaStageVerifierError("Market replay stage coverage is incomplete")
    universe_records = value["corpus_universe_manifest"]["records"]
    model_summaries: dict[str, Any] = {}
    market_summaries: dict[str, Any] = {}
    for stage in stages:
        expected_accessions = [
            record["accession_number"]
            for record in universe_records
            if record["artifact_stage"] == stage
        ]
        expected_accessions.sort(
            key=lambda accession: next(
                (
                    record["availability_session"],
                    record["accession_number"],
                )
                for record in universe_records
                if record["accession_number"] == accession
            )
        )
        model_summaries[stage] = validate_model_attempt_batch(
            batches[stage],
            candidate_manifest=candidate,
            expected_accessions=expected_accessions,
        )
        if model_summaries[stage].get("stage") != stage:
            raise SecFilingGemmaStageVerifierError(
                "Model batch replay crossed a stage slot"
            )
        market_summaries[stage] = validate_market_snapshot_stage_replay(markets[stage])
        if market_summaries[stage]["artifact_stage"] != stage:
            raise SecFilingGemmaStageVerifierError("Market replay crossed a stage")
    prediction = validate_prediction_artifact_replay(
        value["prediction_replay"],
        candidate_sha256=candidate_hash,
        corpus_universe_sha256=calendar["corpus_universe_sha256"],
    )
    if prediction["stage"] != prerequisite:
        raise SecFilingGemmaStageVerifierError("Prediction artifact stage is not prerequisite")
    learner = validate_learner_refit_replays(
        value["learner_replays"],
        candidate_sha256=candidate_hash,
        prediction_prefix=prediction["prefix"],
    )
    prerequisite_market = markets[prerequisite]["market_stage_manifest"]
    scores = validate_raw_scores_gates_and_ranking(
        value["score_replay"],
        prediction_prefix=prediction["prefix"],
        market_stage=prerequisite_market,
        prerequisite_stage=prerequisite,
    )
    runtime_hash = validate_stage_runtime_receipt(
        value["stage_runtime_receipt"],
        prerequisite_stage=prerequisite,
        candidate_sha256=candidate_hash,
        model_batch_summary=model_summaries[prerequisite],
    )
    access = validate_registry_request_and_stage_access(
        registry=value["reveal_registry"],
        registry_external_pin=value["registry_external_pin"],
        stage_access_manifest=stage_access_manifest,
        expected_context=expected_context,
        candidate_manifest=candidate,
        corpus_universe_manifest=value["corpus_universe_manifest"],
        prerequisite_content_manifest=value["prerequisite_content_manifest"],
        expected_stage_evidence_sha256=evidence_hash,
    )

    replayed = {
        "candidate_identity",
        "gate_replay",
        "ledger_replay",
        "no_leverage",
        "registry_identity",
        "request_identity",
        "stage_access_manifest_replay",
        "stage_identity",
    }
    status = {
        check: (
            {"status": "blocked", "reason": _BLOCKING_GAPS[check]}
            if check in _BLOCKING_GAPS
            else {"status": "replayed" if check in replayed else "blocked", "reason": None}
        )
        for check in REQUIRED_STAGE_VERIFIER_CHECKS
    }
    if prerequisite == "intermediate":
        status["stage_identity"] = {
            "status": "blocked",
            "reason": _INTERMEDIATE_STAGE_IDENTITY_GAP,
        }
    # Checks not backed by one authoritative component remain blocked even if
    # their lower-level ingredients were individually inspected.
    for check in REQUIRED_STAGE_VERIFIER_CHECKS:
        if status[check]["status"] == "blocked" and status[check]["reason"] is None:
            status[check]["reason"] = (
                "no complete authoritative end-to-end component currently proves this check"
            )
    component_hashes = {
        "candidate_and_sources": canonical_sha256(identities),
        "calendar_and_universe": canonical_sha256(calendar),
        "model_batches": canonical_sha256(model_summaries),
        "market_replays": canonical_sha256(market_summaries),
        "prediction": canonical_sha256(
            {key: prediction[key] for key in prediction if key != "prefix"}
        ),
        "learner": canonical_sha256(learner),
        "scores": canonical_sha256(scores),
        "runtime": runtime_hash,
        "registry_request_access": canonical_sha256(access),
    }
    body = {
        "schema_version": STAGE_AUDIT_RECEIPT_SCHEMA_VERSION,
        "validator_id": AUTHORITATIVE_VALIDATOR_ID,
        "validator_source_sha256": identities["stage_verifier_source_sha256"],
        "prerequisite_stage": prerequisite,
        "requested_stage": _PREREQUISITE_TRANSITIONS[prerequisite],
        "stage_evidence_sha256": evidence_hash,
        "candidate_sha256": candidate_hash,
        "candidate_design_sha256": candidate_design_sha256(candidate),
        "selected_candidate_id": scores["selected_candidate_id"],
        "semantic_checks": list(REQUIRED_STAGE_VERIFIER_CHECKS),
        "check_status": status,
        "component_receipt_sha256s": component_hashes,
        "blocking_check_count": sum(
            item["status"] == "blocked" for item in status.values()
        ),
        "all_checks_completed": False,
        "authorizes_outcome_access": False,
    }
    return {**body, "audit_receipt_sha256": canonical_sha256(body)}


def authoritative_prerequisite_validator(
    evidence: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
    expected_context: Mapping[str, Any],
) -> None:
    """Fail closed instead of manufacturing an authorizing semantic result."""

    receipt = audit_stage_evidence(evidence, stage_access_manifest, expected_context)
    blocked = [
        check
        for check, result in receipt["check_status"].items()
        if result["status"] == "blocked"
    ]
    raise SecFilingGemmaStageVerifierBlocked(
        "Stage promotion remains disabled; unresolved authoritative checks: "
        + ", ".join(blocked)
    )


__all__ = [
    "AUTHORITATIVE_VALIDATOR_ID",
    "OWNED_HARDENED_TRANSPORT_MODE",
    "STAGE_AUDIT_RECEIPT_SCHEMA_VERSION",
    "STAGE_EVIDENCE_SCHEMA_VERSION",
    "STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION",
    "SecFilingGemmaStageVerifierBlocked",
    "SecFilingGemmaStageVerifierError",
    "audit_stage_evidence",
    "authoritative_prerequisite_validator",
    "validate_calendar_and_universe_snapshot",
    "validate_candidate_source_bytes",
    "validate_learner_refit_replays",
    "validate_market_snapshot_stage_replay",
    "validate_model_attempt_batch",
    "validate_prediction_artifact_replay",
    "validate_raw_scores_gates_and_ranking",
    "validate_registry_request_and_stage_access",
    "validate_stage_runtime_receipt",
]
