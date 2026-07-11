"""Atomic exact-byte seals for pre-label SEC/Gemma prediction prefixes.

The prediction-evidence module is deliberately pure.  This module is its
narrow durable boundary: it accepts one *canonical byte string* containing a
prediction prefix, derives every seal identity from those bytes, and advances
an externally pinned append-only state with compare-and-swap semantics.

The store never accepts caller-supplied artifact checksums or identity maps.
It also never stores labels or realized outcomes.  A receipt can be checked by
the stage verifier using only the exact artifact bytes, the exact receipt
bytes, and the previously externalized pin bytes; filesystem trust is not part
of that verification operation.  A seal proves structural exact-byte ancestry,
not semantic validity by itself.  Before a seal can authorize scoring or a
stage transition, the fixed stage verifier must also replay the authoritative
prediction-prefix validator against these same bytes and the external
candidate, calendar, event, and market pins.
"""

from __future__ import annotations

from collections.abc import Mapping
import base64
import copy
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import time
from typing import Any, Final
import uuid

from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    PREDICTION_PREFIX_SCHEMA_VERSION,
    PREDICTION_ROW_SCHEMA_VERSION,
    _PREDICTION_ROW_KEYS as _AUTHORITATIVE_PREDICTION_ROW_KEYS,
)


STORE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-prediction-artifact-store-v1"
)
EXTERNAL_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-prediction-artifact-external-pin-v1"
)
SEAL_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-prediction-artifact-seal-receipt-v1"
)
STATE_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-prediction-artifact-state-entry-v1"
)
ARTIFACT_ENCODING: Final[str] = "canonical-json-utf8-sorted-keys-v1"
STATE_FILENAME: Final[str] = "sec_gemma_prediction_artifact_store.json"
LOCK_FILENAME: Final[str] = ".sec_gemma_prediction_artifact_store.lock"

_STAGES: Final[frozenset[str]] = frozenset(
    {"development", "intermediate", "final"}
)
_STAGE_SEQUENCE: Final[tuple[str, ...]] = (
    "development",
    "intermediate",
    "final",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TEMP_RE = re.compile(
    rf"\.{re.escape(STATE_FILENAME)}\.[0-9a-f]{{32}}\.tmp\Z"
)
_WINDOWS_REPARSE_POINT = 0x400
_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_MAX_ARTIFACT_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_STATE_BYTES: Final[int] = 512 * 1024 * 1024

_PREFIX_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "initial_event_sequence_sha256",
        "event_sequence_sha256",
        "row_count",
        "genesis_sha256",
        "parent_prefix_sha256",
        "parent_tip_sha256",
        "appended_row_sha256",
        "tip_sha256",
        "rows_sha256",
        "rows",
        "prediction_prefix_sha256",
    }
)
_PIN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "candidate_sha256",
        "stage",
        "sealed_artifact_count",
        "last_prediction_sequence_number",
        "prediction_prefix_sha256",
        "prediction_tip_sha256",
        "prediction_rows_sha256",
        "artifact_sha256",
        "seal_tip_sha256",
        "external_pin_sha256",
    }
)
_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_encoding",
        "candidate_sha256",
        "stage",
        "prediction_sequence_number",
        "prediction_prefix_sha256",
        "prediction_tip_sha256",
        "prediction_rows_sha256",
        "parent_prediction_prefix_sha256",
        "parent_prediction_tip_sha256",
        "parent_prediction_rows_sha256",
        "parent_artifact_sha256",
        "artifact_sha256",
        "artifact_size_bytes",
        "prior_external_pin_sha256",
        "parent_seal_sha256",
        "seal_sha256",
        "next_external_pin_sha256",
        "receipt_sha256",
    }
)
_STATE_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "artifact_bytes_base64", "receipt"}
)
_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "candidate_sha256",
        "stage",
        "entries",
        "current_external_pin",
        "state_sha256",
    }
)

# ``label_maturity_session`` is scheduling metadata already present before a
# prediction is made and is intentionally allowed.  The keys below represent
# released targets, realized ledgers, or future outcomes and can never enter a
# pre-label artifact store, even if a caller recomputes every surrounding hash.
_FORBIDDEN_OUTCOME_KEYS: Final[frozenset[str]] = frozenset(
    {
        "label",
        "labels",
        "label_release",
        "label_release_ledger",
        "label_release_ledger_sha256",
        "cash_active_log_edge_10bps",
        "cash_active_log_edge_10bps_hex",
        "cash_beats_long_10bps",
        "strategy_ledger_slice_sha256",
        "benchmark_ledger_slice_sha256",
        "outcome_ledger_row_sha256",
        "release_session",
        "first_eligible_prediction_session",
        "as_of_decision_session",
        "release_sha256",
        "realized_return",
        "future_return",
        "future_price",
        "return",
        "returns",
        "profit",
        "loss",
        "profit_and_loss",
        "pnl",
    }
)


class SecFilingGemmaArtifactSealerError(ValueError):
    """The prediction artifact or its durable seal failed closed."""


@dataclass(frozen=True, slots=True)
class ValidatedPredictionArtifactSeal:
    """Immutable result of exact-byte receipt validation."""

    candidate_sha256: str
    stage: str
    prediction_sequence_number: int
    prediction_prefix_sha256: str
    prediction_tip_sha256: str
    artifact_sha256: str
    parent_seal_sha256: str
    seal_sha256: str
    receipt_sha256: str
    next_external_pin_bytes: bytes


@dataclass(frozen=True, slots=True)
class PredictionArtifactSealResult:
    """Canonical receipt plus the newly externalizable exact pin bytes."""

    artifact_bytes: bytes
    receipt_bytes: bytes
    next_external_pin_bytes: bytes
    validation: ValidatedPredictionArtifactSeal


@dataclass(frozen=True, slots=True)
class _ArtifactIdentity:
    candidate_sha256: str
    stage: str
    sequence_number: int
    prediction_prefix_sha256: str
    prediction_tip_sha256: str
    prediction_rows_sha256: str
    prior_prediction_rows_sha256: str | None
    parent_prediction_prefix_sha256: str | None
    parent_prediction_tip_sha256: str
    artifact_sha256: str
    artifact_size_bytes: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any, location: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must be a finite JSON value"
        ) from exc


def _canonical_sha256(value: Any, location: str) -> str:
    return _sha256_bytes(_canonical_json_bytes(value, location))


def _strict_json_bytes(payload: bytes, location: str, *, maximum: int) -> Any:
    if type(payload) is not bytes:
        raise TypeError(f"{location} must be exact immutable bytes")
    if not payload or len(payload) > maximum:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must contain between 1 and {maximum} bytes"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecFilingGemmaArtifactSealerError(
                    f"{location} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} contains non-finite JSON constant {value}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except SecFilingGemmaArtifactSealerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} is not strict UTF-8 JSON"
        ) from exc
    if not hmac.compare_digest(_canonical_json_bytes(value, location), payload):
        raise SecFilingGemmaArtifactSealerError(
            f"{location} is not the exact canonical JSON encoding"
        )
    return value


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must be a string-keyed JSON object"
        )
    return value


def _expect_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaArtifactSealerError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _optional_sha256(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, location)


def _stage(value: Any, location: str = "stage") -> str:
    if not isinstance(value, str) or value not in _STAGES:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must be development, intermediate, or final"
        )
    return value


def _reject_outcomes(value: Any, location: str = "prediction artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SecFilingGemmaArtifactSealerError(
                    f"{location} contains a non-string JSON key"
                )
            camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
            lowered = camel_split.casefold()
            tokens = {
                token
                for token in re.split(r"[^a-z0-9]+", lowered)
                if token
            }
            allowed_label_metadata = lowered.endswith("label_maturity_session")
            if (
                "outcome" in lowered
                or ("label" in lowered and not allowed_label_metadata)
                or lowered.endswith("_return")
                or lowered.startswith("return_")
                or lowered in _FORBIDDEN_OUTCOME_KEYS
                or (
                    not allowed_label_metadata
                    and tokens
                    & {
                        "future",
                        "label",
                        "labels",
                        "loss",
                        "outcome",
                        "outcomes",
                        "pass",
                        "passed",
                        "pnl",
                        "profit",
                        "realized",
                        "result",
                        "results",
                        "return",
                        "returns",
                        "score",
                        "scores",
                    }
                )
            ):
                raise SecFilingGemmaArtifactSealerError(
                    f"{location} contains forbidden label/outcome field {key!r}"
                )
            _reject_outcomes(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_outcomes(child, f"{location}[{index}]")


def _artifact_identity(artifact_bytes: bytes) -> _ArtifactIdentity:
    artifact = _expect_mapping(
        _strict_json_bytes(
            artifact_bytes,
            "prediction artifact bytes",
            maximum=_MAX_ARTIFACT_BYTES,
        ),
        "prediction artifact",
    )
    _expect_keys(artifact, _PREFIX_KEYS, "prediction artifact")
    if artifact["schema_version"] != PREDICTION_PREFIX_SCHEMA_VERSION:
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact is not a pre-label prediction prefix"
        )
    _reject_outcomes(artifact)

    candidate = _sha256(artifact["candidate_sha256"], "candidate_sha256")
    for key in (
        "contract_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "initial_event_sequence_sha256",
        "event_sequence_sha256",
        "genesis_sha256",
        "parent_tip_sha256",
        "appended_row_sha256",
        "tip_sha256",
        "rows_sha256",
        "prediction_prefix_sha256",
    ):
        _sha256(artifact[key], key)
    parent_prefix = _optional_sha256(
        artifact["parent_prefix_sha256"], "parent_prefix_sha256"
    )
    rows = artifact["rows"]
    row_count = _strict_int(artifact["row_count"], "row_count", minimum=1)
    if not isinstance(rows, list) or len(rows) != row_count:
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact row_count does not match its exact rows"
        )
    if artifact["rows_sha256"] != _canonical_sha256(rows, "prediction rows"):
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact rows hash is inconsistent"
        )

    observed_stage: str | None = None
    previous_stage_index: int | None = None
    row_hashes: set[str] = set()
    for index, raw_row in enumerate(rows, start=1):
        row = _expect_mapping(raw_row, f"prediction row {index}")
        _expect_keys(
            row,
            frozenset(_AUTHORITATIVE_PREDICTION_ROW_KEYS),
            f"prediction row {index}",
        )
        if row["schema_version"] != PREDICTION_ROW_SCHEMA_VERSION:
            raise SecFilingGemmaArtifactSealerError(
                f"Prediction row {index} has the wrong schema"
            )
        if _strict_int(row["sequence_number"], "row sequence", minimum=1) != index:
            raise SecFilingGemmaArtifactSealerError(
                "Prediction rows are duplicated, reordered, or skipped"
            )
        if row["candidate_sha256"] != candidate:
            raise SecFilingGemmaArtifactSealerError(
                "Prediction row belongs to another candidate"
            )
        row_stage = _stage(row["stage"], "prediction row stage")
        stage_index = _STAGE_SEQUENCE.index(row_stage)
        if index == 1 and row_stage != "development":
            raise SecFilingGemmaArtifactSealerError(
                "The cumulative prediction chain must begin in development"
            )
        if previous_stage_index is not None and not (
            previous_stage_index <= stage_index <= previous_stage_index + 1
        ):
            raise SecFilingGemmaArtifactSealerError(
                "Prediction stages regress, skip a stage, or are reordered"
            )
        previous_stage_index = stage_index
        observed_stage = row_stage
        row_hash = _sha256(
            row["prediction_row_sha256"], "prediction_row_sha256"
        )
        if row_hash in row_hashes:
            raise SecFilingGemmaArtifactSealerError(
                "Prediction artifact duplicates a prediction row"
            )
        row_hashes.add(row_hash)
        row_body = {
            key: row[key] for key in row if key != "prediction_row_sha256"
        }
        if row_hash != _canonical_sha256(row_body, f"prediction row {index}"):
            raise SecFilingGemmaArtifactSealerError(
                f"Prediction row {index} hash is inconsistent"
            )

    assert observed_stage is not None
    last = _expect_mapping(rows[-1], "last prediction row")
    last_hash = last["prediction_row_sha256"]
    if (
        artifact["appended_row_sha256"] != last_hash
        or artifact["tip_sha256"] != last_hash
        or last["prior_prediction_prefix_sha256"] != parent_prefix
        or last["parent_prediction_sha256"] != artifact["parent_tip_sha256"]
    ):
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact tip or parent identity is inconsistent"
        )
    if row_count == 1 and parent_prefix is not None:
        raise SecFilingGemmaArtifactSealerError(
            "First prediction artifact cannot claim a parent prefix"
        )
    prefix_body = {
        key: artifact[key]
        for key in artifact
        if key != "prediction_prefix_sha256"
    }
    prefix_hash = _canonical_sha256(prefix_body, "prediction prefix body")
    if artifact["prediction_prefix_sha256"] != prefix_hash:
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact prefix hash is inconsistent"
        )
    return _ArtifactIdentity(
        candidate_sha256=candidate,
        stage=observed_stage,
        sequence_number=row_count,
        prediction_prefix_sha256=prefix_hash,
        prediction_tip_sha256=last_hash,
        prediction_rows_sha256=artifact["rows_sha256"],
        prior_prediction_rows_sha256=(
            None
            if row_count == 1
            else _canonical_sha256(rows[:-1], "prior prediction rows")
        ),
        parent_prediction_prefix_sha256=parent_prefix,
        parent_prediction_tip_sha256=artifact["parent_tip_sha256"],
        artifact_sha256=_sha256_bytes(artifact_bytes),
        artifact_size_bytes=len(artifact_bytes),
    )


def build_prediction_artifact_bytes(
    prediction_prefix: Mapping[str, Any],
) -> bytes:
    """Detach one mapping into canonical bytes and validate its pre-label shape."""

    if not isinstance(prediction_prefix, Mapping):
        raise TypeError("prediction_prefix must be a mapping")
    # Serialize once, then parse the detached result.  Subsequent validation
    # never consults the caller's possibly mutable object.
    detached = _canonical_json_bytes(prediction_prefix, "prediction_prefix")
    _artifact_identity(detached)
    return detached


def _seal_genesis(candidate_sha256: str, stage: str) -> str:
    return _canonical_sha256(
        {
            "domain": "aapl-sec-gemma-prediction-artifact-seal-genesis-v1",
            "candidate_sha256": candidate_sha256,
            "stage": stage,
        },
        "seal genesis",
    )


def _build_pin(
    *,
    candidate_sha256: str,
    stage: str,
    count: int,
    sequence_number: int,
    prediction_prefix_sha256: str | None,
    prediction_tip_sha256: str | None,
    prediction_rows_sha256: str | None,
    artifact_sha256: str | None,
    seal_tip_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_version": EXTERNAL_PIN_SCHEMA_VERSION,
        "candidate_sha256": candidate_sha256,
        "stage": stage,
        "sealed_artifact_count": count,
        "last_prediction_sequence_number": sequence_number,
        "prediction_prefix_sha256": prediction_prefix_sha256,
        "prediction_tip_sha256": prediction_tip_sha256,
        "prediction_rows_sha256": prediction_rows_sha256,
        "artifact_sha256": artifact_sha256,
        "seal_tip_sha256": seal_tip_sha256,
    }
    return {**body, "external_pin_sha256": _canonical_sha256(body, "pin body")}


def _initial_pin(candidate_sha256: str, stage: str) -> dict[str, Any]:
    if stage != "development":
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact sealing must begin in development"
        )
    return _build_pin(
        candidate_sha256=candidate_sha256,
        stage=stage,
        count=0,
        sequence_number=0,
        prediction_prefix_sha256=None,
        prediction_tip_sha256=None,
        prediction_rows_sha256=None,
        artifact_sha256=None,
        seal_tip_sha256=_seal_genesis(candidate_sha256, stage),
    )


def _validate_pin(value: Any, location: str) -> dict[str, Any]:
    pin = _expect_mapping(value, location)
    _expect_keys(pin, _PIN_KEYS, location)
    if pin["schema_version"] != EXTERNAL_PIN_SCHEMA_VERSION:
        raise SecFilingGemmaArtifactSealerError(f"Unknown {location} schema")
    candidate = _sha256(pin["candidate_sha256"], f"{location} candidate")
    stage = _stage(pin["stage"], f"{location} stage")
    count = _strict_int(pin["sealed_artifact_count"], f"{location} count")
    sequence = _strict_int(
        pin["last_prediction_sequence_number"], f"{location} sequence"
    )
    if sequence != count:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} count and sequence disagree"
        )
    prefix = _optional_sha256(
        pin["prediction_prefix_sha256"], f"{location} prefix"
    )
    tip = _optional_sha256(pin["prediction_tip_sha256"], f"{location} tip")
    rows_hash = _optional_sha256(
        pin["prediction_rows_sha256"], f"{location} rows"
    )
    artifact_hash = _optional_sha256(
        pin["artifact_sha256"], f"{location} artifact"
    )
    seal_tip = _sha256(pin["seal_tip_sha256"], f"{location} seal tip")
    if count == 0:
        if (
            stage != "development"
            or prefix is not None
            or tip is not None
            or rows_hash is not None
            or artifact_hash is not None
        ):
            raise SecFilingGemmaArtifactSealerError(
                f"{location} genesis cannot bind a prediction"
            )
        if seal_tip != _seal_genesis(candidate, stage):
            raise SecFilingGemmaArtifactSealerError(
                f"{location} genesis seal changed"
            )
    elif (
        prefix is None
        or tip is None
        or rows_hash is None
        or artifact_hash is None
    ):
        raise SecFilingGemmaArtifactSealerError(
            f"{location} omits its latest prediction identity"
        )
    body = {key: pin[key] for key in pin if key != "external_pin_sha256"}
    expected = _canonical_sha256(body, f"{location} body")
    if not hmac.compare_digest(
        expected, _sha256(pin["external_pin_sha256"], f"{location} hash")
    ):
        raise SecFilingGemmaArtifactSealerError(f"{location} hash is inconsistent")
    return copy.deepcopy(dict(pin))


def _pin_from_exact_bytes(payload: bytes, location: str) -> dict[str, Any]:
    return _validate_pin(
        _strict_json_bytes(payload, location, maximum=64 * 1024), location
    )


def _seal_body(
    identity: _ArtifactIdentity, prior_pin: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "domain": "aapl-sec-gemma-prediction-artifact-seal-v1",
        "artifact_encoding": ARTIFACT_ENCODING,
        "candidate_sha256": identity.candidate_sha256,
        "stage": identity.stage,
        "prediction_sequence_number": identity.sequence_number,
        "prediction_prefix_sha256": identity.prediction_prefix_sha256,
        "prediction_tip_sha256": identity.prediction_tip_sha256,
        "prediction_rows_sha256": identity.prediction_rows_sha256,
        "parent_prediction_prefix_sha256": (
            identity.parent_prediction_prefix_sha256
        ),
        "parent_prediction_tip_sha256": identity.parent_prediction_tip_sha256,
        "parent_prediction_rows_sha256": prior_pin[
            "prediction_rows_sha256"
        ],
        "parent_artifact_sha256": prior_pin["artifact_sha256"],
        "artifact_sha256": identity.artifact_sha256,
        "artifact_size_bytes": identity.artifact_size_bytes,
        "prior_external_pin_sha256": prior_pin["external_pin_sha256"],
        "parent_seal_sha256": prior_pin["seal_tip_sha256"],
    }


def _build_receipt(
    identity: _ArtifactIdentity, prior_pin: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    seal_body = _seal_body(identity, prior_pin)
    seal_hash = _canonical_sha256(seal_body, "seal body")
    next_pin = _build_pin(
        candidate_sha256=identity.candidate_sha256,
        stage=identity.stage,
        count=prior_pin["sealed_artifact_count"] + 1,
        sequence_number=identity.sequence_number,
        prediction_prefix_sha256=identity.prediction_prefix_sha256,
        prediction_tip_sha256=identity.prediction_tip_sha256,
        prediction_rows_sha256=identity.prediction_rows_sha256,
        artifact_sha256=identity.artifact_sha256,
        seal_tip_sha256=seal_hash,
    )
    body = {
        "schema_version": SEAL_RECEIPT_SCHEMA_VERSION,
        **{key: value for key, value in seal_body.items() if key != "domain"},
        "seal_sha256": seal_hash,
        "next_external_pin_sha256": next_pin["external_pin_sha256"],
    }
    receipt = {
        **body,
        "receipt_sha256": _canonical_sha256(body, "seal receipt body"),
    }
    return receipt, next_pin


def validate_prediction_artifact_seal_receipt(
    *,
    artifact_bytes: bytes,
    receipt_bytes: bytes,
    external_prior_pin_bytes: bytes,
) -> ValidatedPredictionArtifactSeal:
    """Purely validate one seal from exact bytes and an external prior pin."""

    identity = _artifact_identity(artifact_bytes)
    prior = _pin_from_exact_bytes(
        external_prior_pin_bytes, "external prior pin bytes"
    )
    receipt = _expect_mapping(
        _strict_json_bytes(
            receipt_bytes, "seal receipt bytes", maximum=256 * 1024
        ),
        "seal receipt",
    )
    _expect_keys(receipt, _RECEIPT_KEYS, "seal receipt")
    if receipt["schema_version"] != SEAL_RECEIPT_SCHEMA_VERSION:
        raise SecFilingGemmaArtifactSealerError("Unknown seal receipt schema")

    if identity.candidate_sha256 != prior["candidate_sha256"]:
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact belongs to another candidate"
        )
    prior_stage_index = _STAGE_SEQUENCE.index(prior["stage"])
    identity_stage_index = _STAGE_SEQUENCE.index(identity.stage)
    if prior["sealed_artifact_count"] == 0:
        valid_stage_transition = identity.stage == "development"
    else:
        valid_stage_transition = (
            prior_stage_index
            <= identity_stage_index
            <= prior_stage_index + 1
        )
    if not valid_stage_transition:
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact stage regresses or skips a stage"
        )
    expected_sequence = prior["last_prediction_sequence_number"] + 1
    if identity.sequence_number != expected_sequence:
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact is duplicated, reordered, skipped, or rolled back"
        )
    if prior["sealed_artifact_count"] == 0:
        if identity.parent_prediction_prefix_sha256 is not None:
            raise SecFilingGemmaArtifactSealerError(
                "First sealed artifact cannot claim a parent prefix"
            )
    elif (
        identity.parent_prediction_prefix_sha256
        != prior["prediction_prefix_sha256"]
        or identity.parent_prediction_tip_sha256
        != prior["prediction_tip_sha256"]
        or identity.prior_prediction_rows_sha256
        != prior["prediction_rows_sha256"]
    ):
        raise SecFilingGemmaArtifactSealerError(
            "Prediction artifact is not the exact child of the external prior pin"
        )

    expected_receipt, next_pin = _build_receipt(identity, prior)
    if dict(receipt) != expected_receipt:
        raise SecFilingGemmaArtifactSealerError(
            "Seal receipt is altered, cross-artifact, or not prior-pin bound"
        )
    next_pin_bytes = _canonical_json_bytes(next_pin, "next external pin")
    return ValidatedPredictionArtifactSeal(
        candidate_sha256=identity.candidate_sha256,
        stage=identity.stage,
        prediction_sequence_number=identity.sequence_number,
        prediction_prefix_sha256=identity.prediction_prefix_sha256,
        prediction_tip_sha256=identity.prediction_tip_sha256,
        artifact_sha256=identity.artifact_sha256,
        parent_seal_sha256=prior["seal_tip_sha256"],
        seal_sha256=expected_receipt["seal_sha256"],
        receipt_sha256=expected_receipt["receipt_sha256"],
        next_external_pin_bytes=next_pin_bytes,
    )


def _state_snapshot(
    *,
    candidate_sha256: str,
    stage: str,
    entries: list[dict[str, Any]],
    current_pin: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": STORE_SCHEMA_VERSION,
        "candidate_sha256": candidate_sha256,
        "stage": stage,
        "entries": copy.deepcopy(entries),
        "current_external_pin": copy.deepcopy(dict(current_pin)),
    }
    return {**body, "state_sha256": _canonical_sha256(body, "state body")}


def _decode_artifact(value: Any, location: str) -> bytes:
    if not isinstance(value, str):
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must be canonical Base64 text"
        )
    try:
        payload = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} is not strict Base64"
        ) from exc
    if base64.b64encode(payload).decode("ascii") != value:
        raise SecFilingGemmaArtifactSealerError(
            f"{location} is not canonical Base64"
        )
    return payload


def _validate_state(value: Any) -> dict[str, Any]:
    state = _expect_mapping(value, "artifact-sealer state")
    _expect_keys(state, _STATE_KEYS, "artifact-sealer state")
    if state["schema_version"] != STORE_SCHEMA_VERSION:
        raise SecFilingGemmaArtifactSealerError("Unknown artifact-sealer state schema")
    candidate = _sha256(state["candidate_sha256"], "state candidate")
    stage = _stage(state["stage"], "state stage")
    entries = state["entries"]
    if not isinstance(entries, list):
        raise SecFilingGemmaArtifactSealerError("State entries must be a JSON list")
    prior = _initial_pin(candidate, "development")
    seen_artifacts: set[str] = set()
    seen_prefixes: set[str] = set()
    seen_seals: set[str] = set()
    seen_receipts: set[str] = set()
    normalized_entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries, start=1):
        entry = _expect_mapping(raw_entry, f"state entry {index}")
        _expect_keys(entry, _STATE_ENTRY_KEYS, f"state entry {index}")
        if entry["schema_version"] != STATE_ENTRY_SCHEMA_VERSION:
            raise SecFilingGemmaArtifactSealerError(
                f"State entry {index} has an unknown schema"
            )
        artifact_bytes = _decode_artifact(
            entry["artifact_bytes_base64"], f"state entry {index} artifact"
        )
        receipt = _expect_mapping(entry["receipt"], f"state entry {index} receipt")
        receipt_bytes = _canonical_json_bytes(
            receipt, f"state entry {index} receipt"
        )
        validation = validate_prediction_artifact_seal_receipt(
            artifact_bytes=artifact_bytes,
            receipt_bytes=receipt_bytes,
            external_prior_pin_bytes=_canonical_json_bytes(
                prior, f"state entry {index} prior pin"
            ),
        )
        if validation.prediction_sequence_number != index:
            raise SecFilingGemmaArtifactSealerError(
                "State entries are duplicated, reordered, skipped, or rolled back"
            )
        identities = (
            (validation.artifact_sha256, seen_artifacts, "artifact"),
            (validation.prediction_prefix_sha256, seen_prefixes, "prefix"),
            (validation.seal_sha256, seen_seals, "seal"),
            (validation.receipt_sha256, seen_receipts, "receipt"),
        )
        for identity, observed, name in identities:
            if identity in observed:
                raise SecFilingGemmaArtifactSealerError(
                    f"State duplicates a prediction {name}"
                )
            observed.add(identity)
        prior = _pin_from_exact_bytes(
            validation.next_external_pin_bytes,
            f"state entry {index} next pin",
        )
        normalized_entries.append(copy.deepcopy(dict(entry)))

    current = _validate_pin(state["current_external_pin"], "current external pin")
    if current != prior or current["stage"] != stage:
        raise SecFilingGemmaArtifactSealerError(
            "Current external pin is stale, forked, or rolled back"
        )
    expected = _state_snapshot(
        candidate_sha256=candidate,
        stage=stage,
        entries=normalized_entries,
        current_pin=current,
    )
    if dict(state) != expected:
        raise SecFilingGemmaArtifactSealerError(
            "Artifact-sealer state hash or canonical content is inconsistent"
        )
    return expected


def _is_reparse(details: os.stat_result) -> bool:
    return bool(
        getattr(details, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _validate_real_directory(path: Path, location: str) -> os.stat_result:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise SecFilingGemmaArtifactSealerError(f"{location} does not exist") from exc
    if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
        raise SecFilingGemmaArtifactSealerError(
            f"{location} cannot be a link or reparse point"
        )
    if not stat.S_ISDIR(details.st_mode):
        raise SecFilingGemmaArtifactSealerError(f"{location} must be a directory")
    return details


def _secure_directory(path: Path, *, create: bool) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    windows = PureWindowsPath(str(absolute))
    if str(absolute).startswith(("\\\\", "//")) or windows.drive.startswith("\\"):
        raise SecFilingGemmaArtifactSealerError(
            "Artifact-sealer directory must be on a local filesystem"
        )
    if not absolute.anchor:
        raise SecFilingGemmaArtifactSealerError(
            "Artifact-sealer directory must be absolute"
        )
    anchor = Path(absolute.anchor)
    _validate_real_directory(anchor, "artifact-sealer filesystem root")
    current = anchor
    for component in absolute.relative_to(anchor).parts:
        if component in {"", ".", ".."}:
            raise SecFilingGemmaArtifactSealerError(
                "Artifact-sealer directory contains an unsafe component"
            )
        current = current / component
        try:
            _validate_real_directory(current, "artifact-sealer directory")
        except SecFilingGemmaArtifactSealerError:
            if not create or current.exists() or current.is_symlink():
                raise
            try:
                current.mkdir()
            except FileExistsError:
                pass
            _validate_real_directory(current, "artifact-sealer directory")
    return absolute


def _validate_regular_details(details: os.stat_result, location: str) -> None:
    if (
        stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
    ):
        raise SecFilingGemmaArtifactSealerError(
            f"{location} must be a single-link regular file, never a link or "
            "reparse point"
        )


def _read_regular_bytes(path: Path, location: str, *, maximum: int) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise SecFilingGemmaArtifactSealerError(f"{location} does not exist") from exc
    _validate_regular_details(before, location)
    if before.st_size <= 0 or before.st_size > maximum:
        raise SecFilingGemmaArtifactSealerError(f"{location} has an unsafe size")
    try:
        descriptor = os.open(path, os.O_RDONLY | _BINARY | _NOFOLLOW)
    except OSError as exc:
        raise SecFilingGemmaArtifactSealerError(
            f"Could not securely open {location}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        _validate_regular_details(opened, location)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SecFilingGemmaArtifactSealerError(
                f"{location} identity changed while opening"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise SecFilingGemmaArtifactSealerError(
                    f"{location} exceeded its size limit while reading"
                )
        after = path.lstat()
        _validate_regular_details(after, location)
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            raise SecFilingGemmaArtifactSealerError(
                f"{location} identity changed while reading"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


class _ExclusiveFileLock:
    def __init__(self, path: Path, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.descriptor: int | None = None

    def __enter__(self) -> "_ExclusiveFileLock":
        flags = os.O_RDWR | os.O_CREAT | _BINARY | _NOFOLLOW
        before: os.stat_result | None = None
        try:
            before = self.path.lstat()
            _validate_regular_details(before, "artifact-sealer lock")
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise SecFilingGemmaArtifactSealerError(
                "Could not securely open artifact-sealer lock"
            ) from exc
        try:
            details = os.fstat(descriptor)
            _validate_regular_details(details, "artifact-sealer lock")
            after = self.path.lstat()
            _validate_regular_details(after, "artifact-sealer lock")
            if (
                (details.st_dev, details.st_ino) != (after.st_dev, after.st_ino)
                or before is not None
                and (before.st_dev, before.st_ino)
                != (details.st_dev, details.st_ino)
            ):
                raise SecFilingGemmaArtifactSealerError(
                    "Artifact-sealer lock identity changed while opening"
                )
            if details.st_size == 0:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise SecFilingGemmaArtifactSealerError(
                            "Timed out acquiring exclusive artifact-sealer lock"
                        ) from exc
                    time.sleep(0.025)
            self.descriptor = descriptor
            return self
        except Exception:
            os.close(descriptor)
            raise

    def __exit__(self, *args: Any) -> None:
        descriptor, self.descriptor = self.descriptor, None
        if descriptor is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | _BINARY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW,
            0o600,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise SecFilingGemmaArtifactSealerError(
                    "Could not finish artifact-sealer state write"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if path.exists() or path.is_symlink():
            _validate_regular_details(path.lstat(), "artifact-sealer state")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            details = temporary.lstat()
        except FileNotFoundError:
            return
        _validate_regular_details(details, "artifact-sealer temporary state")
        temporary.unlink()
        _fsync_directory(path.parent)


def _cleanup_interrupted_temporaries(directory: Path) -> None:
    for entry in os.scandir(directory):
        if _TEMP_RE.fullmatch(entry.name) is None:
            continue
        path = Path(entry.path)
        _validate_regular_details(
            path.lstat(), "interrupted artifact-sealer temporary state"
        )
        path.unlink()
    _fsync_directory(directory)


class SecFilingGemmaPredictionArtifactSealer:
    """Single-candidate, single-stage append-only exact-byte seal store."""

    __slots__ = ("_store_directory", "_lock_timeout_seconds")

    def __init__(
        self, *, store_directory: Path, lock_timeout_seconds: float = 5.0
    ) -> None:
        if isinstance(lock_timeout_seconds, bool) or not isinstance(
            lock_timeout_seconds, (int, float)
        ):
            raise TypeError("lock_timeout_seconds must be a positive number")
        if not 0.0 < float(lock_timeout_seconds) <= 60.0:
            raise SecFilingGemmaArtifactSealerError(
                "lock_timeout_seconds must be in (0, 60]"
            )
        object.__setattr__(
            self,
            "_store_directory",
            Path(os.path.abspath(os.fspath(store_directory))),
        )
        object.__setattr__(self, "_lock_timeout_seconds", float(lock_timeout_seconds))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError("Artifact-sealer configuration is immutable")
        object.__setattr__(self, name, value)

    @property
    def store_directory(self) -> Path:
        return self._store_directory

    @property
    def state_path(self) -> Path:
        return self._store_directory / STATE_FILENAME

    @property
    def lock_path(self) -> Path:
        return self._store_directory / LOCK_FILENAME

    def _locked(self) -> _ExclusiveFileLock:
        secured = _secure_directory(self._store_directory, create=True)
        if secured != self._store_directory:
            raise SecFilingGemmaArtifactSealerError(
                "Artifact-sealer directory identity changed"
            )
        return _ExclusiveFileLock(self.lock_path, self._lock_timeout_seconds)

    def _read_state_locked(self) -> dict[str, Any]:
        payload = _read_regular_bytes(
            self.state_path,
            "authoritative artifact-sealer state",
            maximum=_MAX_STATE_BYTES,
        )
        parsed = _strict_json_bytes(
            payload,
            "authoritative artifact-sealer state",
            maximum=_MAX_STATE_BYTES,
        )
        return _validate_state(parsed)

    def initialize(self, *, candidate_sha256: str, stage: str) -> bytes:
        """Create deterministic genesis once and return its exact pin bytes.

        An already-advanced store cannot be re-trusted through ``initialize``;
        callers must use :meth:`load` with their external pin instead.
        """

        candidate = _sha256(candidate_sha256, "candidate_sha256")
        stage_value = _stage(stage)
        if stage_value != "development":
            raise SecFilingGemmaArtifactSealerError(
                "Prediction artifact sealing must initialize in development"
            )
        with self._locked():
            locked_directory = self._store_directory
            _cleanup_interrupted_temporaries(locked_directory)
            if self.state_path.exists() or self.state_path.is_symlink():
                self._read_state_locked()
                raise SecFilingGemmaArtifactSealerError(
                    "Existing store always requires its external pin; use load"
                )
            pin = _initial_pin(candidate, stage_value)
            state = _state_snapshot(
                candidate_sha256=candidate,
                stage=stage_value,
                entries=[],
                current_pin=pin,
            )
            _atomic_replace(
                locked_directory / STATE_FILENAME,
                _canonical_json_bytes(state, "initial state"),
            )
            if self._store_directory != locked_directory:
                raise SecFilingGemmaArtifactSealerError(
                    "Artifact-sealer path changed during initialization"
                )
            loaded = self._read_state_locked()
            return _canonical_json_bytes(
                loaded["current_external_pin"], "initial external pin"
            )

    def load(self, *, external_pin_bytes: bytes) -> bytes:
        """Reload every exact artifact/receipt and reject rollback or a fork."""

        external = _pin_from_exact_bytes(
            external_pin_bytes, "expected external pin bytes"
        )
        with self._locked():
            locked_directory = self._store_directory
            _cleanup_interrupted_temporaries(locked_directory)
            state = self._read_state_locked()
            current_bytes = _canonical_json_bytes(
                state["current_external_pin"], "current external pin"
            )
            if not hmac.compare_digest(current_bytes, external_pin_bytes):
                raise SecFilingGemmaArtifactSealerError(
                    "Artifact-sealer state is stale, forked, or rolled back against the external pin"
                )
            if external != state["current_external_pin"]:
                raise SecFilingGemmaArtifactSealerError(
                    "External pin does not match authoritative state"
                )
            if self._store_directory != locked_directory:
                raise SecFilingGemmaArtifactSealerError(
                    "Artifact-sealer path changed while loading"
                )
            return current_bytes

    def compare_and_swap_seal(
        self,
        *,
        artifact_bytes: bytes,
        external_prior_pin_bytes: bytes,
    ) -> PredictionArtifactSealResult:
        """Atomically seal exactly one child prefix against an external pin."""

        # Validate caller bytes before taking an OS lock, then replay the same
        # pure validation under the authoritative state immediately before CAS.
        identity = _artifact_identity(artifact_bytes)
        supplied_prior = _pin_from_exact_bytes(
            external_prior_pin_bytes, "external prior pin bytes"
        )
        with self._locked():
            locked_directory = self._store_directory
            _cleanup_interrupted_temporaries(locked_directory)
            state = self._read_state_locked()
            current_pin_bytes = _canonical_json_bytes(
                state["current_external_pin"], "current external pin"
            )
            if not hmac.compare_digest(
                current_pin_bytes, external_prior_pin_bytes
            ) or supplied_prior != state["current_external_pin"]:
                raise SecFilingGemmaArtifactSealerError(
                    "Seal compare-and-swap rejected a stale, forked, or rollback pin"
                )
            if identity.candidate_sha256 != state["candidate_sha256"]:
                raise SecFilingGemmaArtifactSealerError(
                    "Prediction artifact belongs to another candidate"
                )
            receipt, _ = _build_receipt(identity, supplied_prior)
            receipt_bytes = _canonical_json_bytes(receipt, "seal receipt")
            validation = validate_prediction_artifact_seal_receipt(
                artifact_bytes=artifact_bytes,
                receipt_bytes=receipt_bytes,
                external_prior_pin_bytes=external_prior_pin_bytes,
            )
            entries = copy.deepcopy(state["entries"])
            entries.append(
                {
                    "schema_version": STATE_ENTRY_SCHEMA_VERSION,
                    "artifact_bytes_base64": base64.b64encode(
                        artifact_bytes
                    ).decode("ascii"),
                    "receipt": receipt,
                }
            )
            next_pin = _pin_from_exact_bytes(
                validation.next_external_pin_bytes, "next external pin"
            )
            next_state = _state_snapshot(
                candidate_sha256=state["candidate_sha256"],
                stage=identity.stage,
                entries=entries,
                current_pin=next_pin,
            )
            _atomic_replace(
                locked_directory / STATE_FILENAME,
                _canonical_json_bytes(next_state, "next state"),
            )
            if self._store_directory != locked_directory:
                raise SecFilingGemmaArtifactSealerError(
                    "Artifact-sealer path changed during compare-and-swap"
                )
            reloaded = self._read_state_locked()
            if reloaded != next_state:
                raise SecFilingGemmaArtifactSealerError(
                    "Persisted artifact-sealer state does not match the CAS result"
                )
            return PredictionArtifactSealResult(
                artifact_bytes=artifact_bytes,
                receipt_bytes=receipt_bytes,
                next_external_pin_bytes=validation.next_external_pin_bytes,
                validation=validation,
            )


__all__ = [
    "ARTIFACT_ENCODING",
    "EXTERNAL_PIN_SCHEMA_VERSION",
    "LOCK_FILENAME",
    "SEAL_RECEIPT_SCHEMA_VERSION",
    "STATE_FILENAME",
    "STORE_SCHEMA_VERSION",
    "PredictionArtifactSealResult",
    "SecFilingGemmaArtifactSealerError",
    "SecFilingGemmaPredictionArtifactSealer",
    "ValidatedPredictionArtifactSeal",
    "build_prediction_artifact_bytes",
    "validate_prediction_artifact_seal_receipt",
]
