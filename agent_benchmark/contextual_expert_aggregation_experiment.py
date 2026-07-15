"""Fail-closed staged-runner foundations for contextual expert aggregation.

This module deliberately contains no model replay, account simulation, score,
or gate.  It owns only the frozen local-input contract, reproducibility checks,
stage deadline, parent-bundle verification, and generic atomic sealing helpers
that the experiment runner will use after the model API is frozen.

Nothing in this module downloads data or calls an API.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contextual_expert_aggregation_artifacts import (
    ARM_ORDER,
    COMPOSITE_CHECKPOINT_SCHEMA_VERSION,
    COST_ORDER,
    DEVELOPMENT_PAYLOAD_NAMES,
    DEVELOPMENT_STAGE,
    FIXED_POLICY_ORDER,
    POLICY_ORDER,
    ContextualExpertAggregationArtifactError,
    parse_composite_stage_checkpoint,
)


CONTRACT_VERSION = "aapl-causal-contextual-expert-aggregation-v2"
EXPECTED_BRANCH = "codex/aapl-causal-contextual-expert-aggregation-v2"
EXPECTED_ORIGIN_REPOSITORY = (
    "github.com/AntonioDomenech/LLM-memory-trading-agent"
)
EXPECTED_ORIGIN_URLS = frozenset(
    {
        "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git",
        "git@github.com:AntonioDomenech/LLM-memory-trading-agent.git",
        "ssh://git@github.com/AntonioDomenech/LLM-memory-trading-agent.git",
    }
)
RUN_TIME_LIMIT_SECONDS = 3600.0
ALLOWED_STAGES = frozenset({"development", "confirmation"})

DEVELOPMENT_END = pd.Timestamp("2018-12-31")
CONFIRMATION_END = pd.Timestamp("2023-12-31")

PHYSICAL_PRICE_COLUMNS = (
    "date",
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "spy_adj_close",
    "qqq_adj_close",
)
CANONICAL_PRICE_COLUMNS = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "aapl_adj_open",
    "spy_adj_close",
    "qqq_adj_close",
)

ROOT_GIT_ATTRIBUTES_PATH = Path(".gitattributes")
ROOT_GIT_IGNORE_PATH = Path(".gitignore")
REQUIREMENTS_PATH = Path("requirements.txt")
CONTRACT_PATH = Path("docs/aapl_causal_contextual_expert_aggregation_v2.md")
PREDECESSOR_CONTRACT_PATH = Path(
    "docs/aapl_causal_contextual_expert_aggregation_v1.md"
)
PREDECESSOR_REJECTION_PATH = Path(
    "e/aapl_causal_contextual_expert_aggregation_v1/PREFLIGHT_REJECTED.md"
)
FIXED_EXPERT_CONTRACT_PATH = Path(
    "docs/aapl_chronological_exhaustion_expert_v1.md"
)
MODEL_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation.py"
)
PACKAGE_INITIALIZER_PATH = Path("agent_benchmark/__init__.py")
BOOTSTRAP_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_bootstrap.py"
)
FIXED_EXPERT_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/chronological_exhaustion_expert.py"
)
REPLAY_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_replay.py"
)
LEDGER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_ledger.py"
)
EVALUATION_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_evaluation.py"
)
ARTIFACTS_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_artifacts.py"
)
STAGE_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_stage.py"
)
VERIFIER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_verifier.py"
)
RUNNER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_experiment.py"
)
MODEL_TEST_PATH = Path("tests/test_contextual_expert_aggregation.py")
PYTEST_CONFIGURATION_PATH = Path("tests/conftest.py")
BOOTSTRAP_TEST_PATH = Path(
    "tests/test_contextual_expert_aggregation_bootstrap.py"
)
FIXED_EXPERT_TEST_PATH = Path("tests/test_chronological_exhaustion_expert.py")
REPLAY_TEST_PATH = Path("tests/test_contextual_expert_aggregation_replay.py")
LEDGER_TEST_PATH = Path("tests/test_contextual_expert_aggregation_ledger.py")
EVALUATION_TEST_PATH = Path(
    "tests/test_contextual_expert_aggregation_evaluation.py"
)
ARTIFACTS_TEST_PATH = Path(
    "tests/test_contextual_expert_aggregation_artifacts.py"
)
STAGE_TEST_PATH = Path("tests/test_contextual_expert_aggregation_stage.py")
VERIFIER_TEST_PATH = Path(
    "tests/test_contextual_expert_aggregation_verifier.py"
)
RUNNER_TEST_PATH = Path(
    "tests/test_contextual_expert_aggregation_experiment.py"
)
README_PATH = Path("README.md")
FROZEN_DEPENDENCY_PATHS = (
    ROOT_GIT_ATTRIBUTES_PATH,
    ROOT_GIT_IGNORE_PATH,
    REQUIREMENTS_PATH,
    CONTRACT_PATH,
    PREDECESSOR_CONTRACT_PATH,
    PREDECESSOR_REJECTION_PATH,
    FIXED_EXPERT_CONTRACT_PATH,
    PACKAGE_INITIALIZER_PATH,
    BOOTSTRAP_IMPLEMENTATION_PATH,
    MODEL_IMPLEMENTATION_PATH,
    FIXED_EXPERT_IMPLEMENTATION_PATH,
    REPLAY_IMPLEMENTATION_PATH,
    LEDGER_IMPLEMENTATION_PATH,
    EVALUATION_IMPLEMENTATION_PATH,
    ARTIFACTS_IMPLEMENTATION_PATH,
    STAGE_IMPLEMENTATION_PATH,
    VERIFIER_IMPLEMENTATION_PATH,
    RUNNER_IMPLEMENTATION_PATH,
    PYTEST_CONFIGURATION_PATH,
    BOOTSTRAP_TEST_PATH,
    MODEL_TEST_PATH,
    FIXED_EXPERT_TEST_PATH,
    REPLAY_TEST_PATH,
    LEDGER_TEST_PATH,
    EVALUATION_TEST_PATH,
    ARTIFACTS_TEST_PATH,
    STAGE_TEST_PATH,
    VERIFIER_TEST_PATH,
    RUNNER_TEST_PATH,
    README_PATH,
)
_CRLF_EQUIVALENT_FROZEN_SUFFIXES = frozenset({".md", ".py", ".pyw"})
_CRLF_EQUIVALENT_FROZEN_CONTROLS = frozenset(
    {
        ROOT_GIT_ATTRIBUTES_PATH.as_posix(),
        ROOT_GIT_IGNORE_PATH.as_posix(),
        REQUIREMENTS_PATH.as_posix(),
    }
)

DEVELOPMENT_CHECKPOINT_FILENAME = "development_checkpoint_through_2018.json"
DEVELOPMENT_GATE_REPORT_FILENAME = "development_gate_report.json"
DEVELOPMENT_REPORT_FILENAME = "report.json"
DEVELOPMENT_PRICE_FILENAME = "development_prices_through_2018.csv"
CONFIRMATION_REGISTRY_DIRECTORY = Path("codex-evidence") / CONTRACT_VERSION
CONFIRMATION_ATTEMPT_STAGE = "confirmation"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SAFE_STAGE_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

DEVELOPMENT_AUTHORIZATION_REQUIRED_PAYLOADS = DEVELOPMENT_PAYLOAD_NAMES

_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_AUTHORIZED_LOADER_ATTESTATION = object()


class ContextualExpertAggregationExperimentError(RuntimeError):
    """Raised when a frozen runner or artifact invariant is violated."""


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following a link or reparse point."""

    return Path(os.path.abspath(os.fspath(path)))


def _path_is_reparse_point(path: Path) -> bool:
    """Detect POSIX links and Windows junction/reparse-point redirections."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ContextualExpertAggregationExperimentError(
            f"Could not inspect path safety: {path}"
        ) from exc
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0)
        & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _harden_path(
    path: Path,
    *,
    label: str,
    require_exists: bool = False,
) -> Path:
    """Reject redirection in every existing lexical component before use."""

    absolute = _lexical_absolute(path)
    anchor = Path(absolute.anchor)
    cursor = anchor
    for component in absolute.parts[1:]:
        cursor = cursor / component
        if _path_is_reparse_point(cursor):
            raise ContextualExpertAggregationExperimentError(
                f"{label} must not use a symlink, junction, or reparse point"
            )
        if not cursor.exists():
            break
    if require_exists and not absolute.exists():
        raise ContextualExpertAggregationExperimentError(f"{label} does not exist")
    return absolute


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    hardened = _harden_path(path, label=label, require_exists=True)
    if not hardened.is_dir():
        raise ContextualExpertAggregationExperimentError(
            f"{label} must be an existing directory"
        )
    try:
        metadata = os.stat(hardened, follow_symlinks=False)
    except OSError as exc:
        raise ContextualExpertAggregationExperimentError(
            f"Could not bind {label} identity"
        ) from exc
    return int(metadata.st_dev), int(metadata.st_ino)


def _require_same_directory_identity(
    path: Path, expected: tuple[int, int], *, label: str
) -> None:
    if _directory_identity(path, label=label) != expected:
        raise ContextualExpertAggregationExperimentError(
            f"{label} changed identity during the operation"
        )


@dataclass(frozen=True)
class BundleIdentity:
    """Frozen identity of an already sealed local bundle."""

    manifest_path: Path
    contract_version: str
    stage: str
    manifest_file_sha256: str
    manifest_self_sha256: str
    checksums_file_sha256: str


@dataclass(frozen=True)
class AuthorizedPriceSpec:
    """Frozen identity and coverage of one physically bounded price file."""

    stage: str
    relative_path: Path
    raw_sha256: str
    git_blob: str | None
    first_session: str
    last_session: str
    rows: int
    date_sequence_sha256: str
    canonical_sha256: str
    source_bundle: BundleIdentity | None = None
    source_provenance_filename: str = "input_provenance.json"
    source_provenance_sha256: str | None = None
    source_parent_bundle: BundleIdentity | None = None
    embedded_parent_filename: str | None = None


DEVELOPMENT_SOURCE_BUNDLE = BundleIdentity(
    manifest_path=Path(
        "e/chronological_exhaustion_expert_v1/"
        "exhaustion-expert-development-v1/stage_manifest.json"
    ),
    contract_version="aapl-chronological-exhaustion-expert-v1",
    stage="development",
    manifest_file_sha256=(
        "sha256:0a375ca457b61b197ca49dbd557ffa679f97d7534e2628c7110a4cb1e36d0cac"
    ),
    manifest_self_sha256=(
        "sha256:d6352a96629008a97b54d56462d9549c5b59cf0d43bbbe621d602c2d9a7b4d94"
    ),
    checksums_file_sha256=(
        "sha256:1241a282b9408b4669e276a3179e9a491d3724b884deaa54abadd872d7313348"
    ),
)

CONFIRMATION_SOURCE_PARENT_BUNDLE = BundleIdentity(
    manifest_path=Path(
        "e/binary_regime_union_selector_v1/"
        "binary-regime-union-selector-development-v1/stage_manifest.json"
    ),
    contract_version="aapl-binary-regime-union-selector-v1",
    stage="development",
    manifest_file_sha256=(
        "sha256:348891ec02f6631de959b01989aea968c7274ddddfcdf687c317496572ea0116"
    ),
    manifest_self_sha256=(
        "sha256:3dfd21609d8851157d7062e5f27cad9e50119fb45a2cd5e8cff8cba9a3d28cd6"
    ),
    checksums_file_sha256=(
        "sha256:bdd6eca8d9b120e593dd07490e883fb7620ce2e7f5a152eb366e180a4a64109c"
    ),
)

CONFIRMATION_SOURCE_BUNDLE = BundleIdentity(
    manifest_path=Path(
        "e/binary_regime_union_selector_v1/"
        "binary-regime-union-selector-validation-v1/stage_manifest.json"
    ),
    contract_version="aapl-binary-regime-union-selector-v1",
    stage="validation",
    manifest_file_sha256=(
        "sha256:95d7a2570f720987480f6343524d95f9242bedbc182c646035999bc6cd0bb5fe"
    ),
    manifest_self_sha256=(
        "sha256:0354355ac460042f96663d1a45cf5e9a8cf4fe873ddf5cc87afefd9c4b5d81dc"
    ),
    checksums_file_sha256=(
        "sha256:496bd0ca94f30f48d90662a8d344eb34d2ab48d64920b7aa270bcae4f56ac115"
    ),
)

DEVELOPMENT_PRICE_SPEC = AuthorizedPriceSpec(
    stage="development",
    relative_path=Path(
        "e/chronological_exhaustion_expert_v1/authorized_inputs/"
        "aapl_spy_qqq_through_2018.csv"
    ),
    raw_sha256=(
        "sha256:9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1"
    ),
    git_blob="71b9aa69478a9a591200833131e930064859721f",
    first_session="1999-03-10",
    last_session="2018-12-31",
    rows=4986,
    date_sequence_sha256=(
        "3b8844f939269760624816e166ebf15705c27549aa89dd578ddb66bd50d87df9"
    ),
    canonical_sha256=(
        "sha256:31b56551b8d1b837f2e69178ab7f206b6bf3c19d11d9d47be0e33cf501db3f45"
    ),
    source_bundle=DEVELOPMENT_SOURCE_BUNDLE,
    source_provenance_sha256=(
        "sha256:122ddeb1a90fa32666e9b7f49cccb10adc3b2de71b56ad0824f9b4f1e137a070"
    ),
)

CONFIRMATION_PRICE_SPEC = AuthorizedPriceSpec(
    stage="confirmation",
    relative_path=Path(
        "e/binary_regime_union_selector_v1/authorized_inputs/"
        "aapl_spy_qqq_through_2023.csv"
    ),
    raw_sha256=(
        "sha256:c5189db9796f25ae69d14814b22ac4a852449aef8b86d3615a289b9fcb8029e9"
    ),
    git_blob="9b47e6596294025bb22ec872051dcc5f6320962a",
    first_session="1999-03-10",
    last_session="2023-12-29",
    rows=6244,
    date_sequence_sha256=(
        "77098a2d35b6cee78ccb100514e599ee4ef6dac55738e7084b02d0e0dd0b63c1"
    ),
    canonical_sha256=(
        "sha256:3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535"
    ),
    source_bundle=CONFIRMATION_SOURCE_BUNDLE,
    source_provenance_sha256=(
        "sha256:2a13e0bc4a5b5a9dc7ecff2fbbc0c02c1b37e7383bc9752e608ec7affadfaf6c"
    ),
    source_parent_bundle=CONFIRMATION_SOURCE_PARENT_BUNDLE,
    embedded_parent_filename="authorized_development_manifest.json",
)

AUTHORIZED_PRICE_SPECS: Mapping[str, AuthorizedPriceSpec] = {
    "development": DEVELOPMENT_PRICE_SPEC,
    "confirmation": CONFIRMATION_PRICE_SPEC,
}


@dataclass(frozen=True)
class LoadedPriceSnapshot:
    """A validated canonical price frame and its deterministic evidence."""

    spec: AuthorizedPriceSpec
    frame: pd.DataFrame
    raw_sha256: str
    canonical_csv_bytes: bytes
    provenance: Mapping[str, Any]
    _loader_attestation: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _authorized_lineage: bool = field(
        default=False, init=False, repr=False, compare=False
    )


@dataclass(frozen=True)
class TrackedFileIdentity:
    path: str
    sha256: str
    git_blob: str


@dataclass(frozen=True)
class VerifiedBundle:
    directory: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    payload_sha256: Mapping[str, str]
    checksums: Mapping[str, str]


@dataclass(frozen=True)
class SealedBundle:
    directory: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    checksums: Mapping[str, str]


@dataclass(frozen=True)
class ConfirmationAuthorization:
    """Durable authority consumed before any confirmation byte is read."""

    development_manifest_path: Path
    development_manifest_sha256: str
    development_checkpoint_sha256: str
    development_gate_report_sha256: str
    git_identity: Mapping[str, Any]
    prelock_git_identity: Mapping[str, Any]
    development_verification: Mapping[str, Any]
    attempt_identity_sha256: str
    attempt_lock_path: Path
    attempt_lock_sha256: str
    attempt_lock_parent_fsync_supported: bool


@dataclass(frozen=True)
class DevelopmentVerificationEvidence:
    """Hash-bound result returned by the one registered exact verifier."""

    verifier_id: str
    verifier_dependency_path: str
    passed: bool
    exact_payload_names: tuple[str, ...]
    report_sha256: str
    checkpoint_sha256: str
    gate_report_sha256: str
    semantic_evidence_sha256: str


@dataclass(frozen=True)
class DevelopmentVerifierRegistration:
    """Source-frozen exact verifier registration; never supplied by a caller."""

    verifier_id: str
    dependency_path: Path
    expected_payload_names: frozenset[str]
    verify: Callable[[VerifiedBundle, AuthorizedPriceSpec], DevelopmentVerificationEvidence]


# The verifier imports these frozen protocol dataclasses, so registration is
# resolved only after this module has finished importing.  Once resolved, the
# exact source-owned object is cached and cannot be replaced through any public
# setter or authorize-time callback.  Explicit ``None`` remains a test-only,
# fail-closed state.
_UNRESOLVED_DEVELOPMENT_VERIFIER = object()
_REGISTERED_DEVELOPMENT_VERIFIER: (
    DevelopmentVerifierRegistration | None | object
) = _UNRESOLVED_DEVELOPMENT_VERIFIER


def _development_verifier_registration() -> DevelopmentVerifierRegistration | None:
    global _REGISTERED_DEVELOPMENT_VERIFIER

    registration = _REGISTERED_DEVELOPMENT_VERIFIER
    if registration is _UNRESOLVED_DEVELOPMENT_VERIFIER:
        try:
            from .contextual_expert_aggregation_verifier import (
                DEVELOPMENT_VERIFIER_REGISTRATION,
            )
        except (ImportError, AttributeError) as exc:
            raise ContextualExpertAggregationExperimentError(
                "Exact development authorization verifier is not registered"
            ) from exc
        registration = DEVELOPMENT_VERIFIER_REGISTRATION
        _REGISTERED_DEVELOPMENT_VERIFIER = registration
    if registration is None:
        return None
    if not isinstance(registration, DevelopmentVerifierRegistration):
        raise ContextualExpertAggregationExperimentError(
            "Exact development authorization verifier registration is invalid"
        )
    return registration


class StageDeadline:
    """Strict monotonic deadline starting at construction (runner entry)."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        *,
        limit_seconds: float = RUN_TIME_LIMIT_SECONDS,
    ) -> None:
        if (
            not math.isfinite(limit_seconds)
            or limit_seconds <= 0.0
            or limit_seconds > RUN_TIME_LIMIT_SECONDS
        ):
            raise ContextualExpertAggregationExperimentError(
                "Deadline limit must be positive, finite, and at most 3600 seconds"
            )
        started = float(clock())
        if not math.isfinite(started):
            raise ContextualExpertAggregationExperimentError(
                "Monotonic clock returned a nonfinite start"
            )
        self._clock = clock
        self._started = started
        self._last = started
        self.limit_seconds = float(limit_seconds)

    def elapsed(self) -> float:
        current = float(self._clock())
        if not math.isfinite(current) or current < self._last:
            raise ContextualExpertAggregationExperimentError(
                "Monotonic clock moved backwards or became nonfinite"
            )
        self._last = current
        return current - self._started

    def check(self, location: str) -> float:
        elapsed = self.elapsed()
        if elapsed >= self.limit_seconds:
            raise ContextualExpertAggregationExperimentError(
                f"Stage reached its {self.limit_seconds:.0f}s deadline at {location}"
            )
        return elapsed


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContextualExpertAggregationExperimentError(
            f"{field} must be a lowercase sha256 digest"
        )
    return value


def self_hashed_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "manifest_sha256" in payload:
        raise ContextualExpertAggregationExperimentError(
            "Manifest payload may not predeclare its self-hash"
        )
    clean = dict(payload)
    return {
        **clean,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(clean)),
    }


def _safe_flat_filename(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or "\x00" in value
    ):
        raise ContextualExpertAggregationExperimentError(
            f"{field} contains an unsafe artifact filename"
        )
    return value


def _safe_run_id(value: Any) -> str:
    if not isinstance(value, str) or _SAFE_RUN_ID_RE.fullmatch(value) is None:
        raise ContextualExpertAggregationExperimentError(
            "run_id must be a safe nonempty name"
        )
    return value


def _validated_hash_inventory(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ContextualExpertAggregationExperimentError(
            f"{field} must be a JSON object"
        )
    result: dict[str, str] = {}
    for filename, digest in value.items():
        name = _safe_flat_filename(filename, field=field)
        result[name] = _require_sha256(digest, field=f"{field}.{name}")
    return result


def canonical_price_csv_bytes(frame: pd.DataFrame) -> bytes:
    if tuple(frame.columns) != CANONICAL_PRICE_COLUMNS:
        raise ContextualExpertAggregationExperimentError(
            "Canonical price frame has unexpected columns"
        )
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.name != "date":
        raise ContextualExpertAggregationExperimentError(
            "Canonical price frame requires a named DatetimeIndex"
        )
    buffer = io.StringIO(newline="")
    frame.reset_index(names="date").to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def _validate_price_spec(spec: AuthorizedPriceSpec) -> None:
    if _SAFE_STAGE_RE.fullmatch(spec.stage) is None:
        raise ContextualExpertAggregationExperimentError("Unsafe price stage")
    if spec.relative_path.is_absolute() or ".." in spec.relative_path.parts:
        raise ContextualExpertAggregationExperimentError(
            "Authorized price path must be a safe relative path"
        )
    _require_sha256(spec.raw_sha256, field="raw_sha256")
    _require_sha256(spec.canonical_sha256, field="canonical_sha256")
    if spec.git_blob is not None and _GIT_OBJECT_RE.fullmatch(spec.git_blob) is None:
        raise ContextualExpertAggregationExperimentError("Invalid frozen Git blob")
    if spec.rows <= 0 or not re.fullmatch(r"[0-9a-f]{64}", spec.date_sequence_sha256):
        raise ContextualExpertAggregationExperimentError(
            "Invalid frozen session coverage"
        )


def _attest_loaded_snapshot(
    snapshot: LoadedPriceSnapshot, *, authorized_lineage: bool
) -> LoadedPriceSnapshot:
    object.__setattr__(
        snapshot, "_loader_attestation", _AUTHORIZED_LOADER_ATTESTATION
    )
    object.__setattr__(snapshot, "_authorized_lineage", authorized_lineage)
    return snapshot


def load_bounded_price_snapshot(
    path: Path, *, spec: AuthorizedPriceSpec
) -> LoadedPriceSnapshot:
    """Load one exact physically bounded six-column local price snapshot."""

    _validate_price_spec(spec)
    source = _harden_path(
        path, label="Authorized price snapshot", require_exists=True
    )
    try:
        raw_bytes = source.read_bytes()
    except OSError as exc:
        raise ContextualExpertAggregationExperimentError(
            f"Authorized price snapshot is unreadable: {source}"
        ) from exc
    observed_raw_sha = sha256_bytes(raw_bytes)
    if observed_raw_sha != spec.raw_sha256:
        raise ContextualExpertAggregationExperimentError(
            "Authorized price snapshot raw hash changed"
        )

    dates: list[datetime] = []
    values: dict[str, list[float]] = {
        name: [] for name in PHYSICAL_PRICE_COLUMNS[1:]
    }
    try:
        decoded = raw_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        if tuple(reader.fieldnames or ()) != PHYSICAL_PRICE_COLUMNS:
            raise ContextualExpertAggregationExperimentError(
                "Authorized price snapshot has unexpected columns"
            )
        previous: datetime | None = None
        for raw_row in reader:
            if (
                None in raw_row
                or set(raw_row) != set(PHYSICAL_PRICE_COLUMNS)
                or any(raw_row[name] in (None, "") for name in PHYSICAL_PRICE_COLUMNS)
            ):
                raise ContextualExpertAggregationExperimentError(
                    "Authorized price snapshot contains an incomplete raw row"
                )
            raw_date = str(raw_row["date"])
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date) is None:
                raise ContextualExpertAggregationExperimentError(
                    "Authorized price dates must be exact ISO dates"
                )
            current = datetime.strptime(raw_date, "%Y-%m-%d")
            if current.strftime("%Y-%m-%d") != raw_date or (
                previous is not None and current <= previous
            ):
                raise ContextualExpertAggregationExperimentError(
                    "Authorized price dates are duplicated or not increasing"
                )
            for name in PHYSICAL_PRICE_COLUMNS[1:]:
                numeric = float(str(raw_row[name]))
                if not math.isfinite(numeric) or numeric <= 0.0:
                    raise ContextualExpertAggregationExperimentError(
                        "Authorized price snapshot contains a nonpositive price"
                    )
                values[name].append(numeric)
            dates.append(current)
            previous = current
    except ContextualExpertAggregationExperimentError:
        raise
    except (UnicodeDecodeError, csv.Error, TypeError, ValueError, OverflowError) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Authorized price snapshot raw parsing failed"
        ) from exc

    if not dates or len(dates) != spec.rows:
        raise ContextualExpertAggregationExperimentError(
            "Authorized price snapshot row count changed"
        )
    first = dates[0].strftime("%Y-%m-%d")
    last = dates[-1].strftime("%Y-%m-%d")
    if first != spec.first_session or last != spec.last_session:
        raise ContextualExpertAggregationExperimentError(
            "Authorized price snapshot physical date bound changed"
        )
    date_payload = "".join(
        f"{value.strftime('%Y-%m-%d')}\n" for value in dates
    ).encode("ascii")
    date_hash = hashlib.sha256(date_payload).hexdigest()
    if date_hash != spec.date_sequence_sha256:
        raise ContextualExpertAggregationExperimentError(
            "Authorized price snapshot session sequence changed"
        )

    frame = pd.DataFrame(
        values,
        index=pd.DatetimeIndex(dates, name="date"),
        dtype=float,
    )
    frame["aapl_adj_open"] = (
        frame["aapl_open"]
        * frame["aapl_adj_close"]
        / frame["aapl_close"]
    )
    frame = frame.loc[:, list(CANONICAL_PRICE_COLUMNS)]
    if (
        not np.isfinite(frame.to_numpy(dtype=float)).all()
        or (frame.to_numpy(dtype=float) <= 0.0).any()
    ):
        raise ContextualExpertAggregationExperimentError(
            "Canonical adjusted prices are invalid"
        )
    canonical_bytes = canonical_price_csv_bytes(frame)
    canonical_sha = sha256_bytes(canonical_bytes)
    if canonical_sha != spec.canonical_sha256:
        raise ContextualExpertAggregationExperimentError(
            "Canonical adjusted-open reconstruction hash changed"
        )
    provenance = {
        "source_type": "physically_bounded_local_csv",
        "source_path": str(source),
        "network_access": False,
        "physical_snapshot_has_later_rows": False,
        "rows_after_bound_returned": False,
        "raw_sha256": observed_raw_sha,
        "bounded_first_date": first,
        "bounded_last_date": last,
        "bounded_rows": len(frame),
        "bounded_result_sha256": canonical_sha,
        "session_coverage": {
            "first_session": first,
            "last_session": last,
            "observations": len(frame),
            "date_sequence_sha256": date_hash,
        },
        "adjusted_open_formula": (
            "aapl_open * aapl_adj_close / aapl_close"
        ),
    }
    return _attest_loaded_snapshot(
        LoadedPriceSnapshot(
            spec=spec,
            frame=frame,
            raw_sha256=observed_raw_sha,
            canonical_csv_bytes=canonical_bytes,
            provenance=provenance,
        ),
        authorized_lineage=False,
    )


def require_snapshot_prefix(
    development: LoadedPriceSnapshot,
    confirmation: LoadedPriceSnapshot,
    *,
    development_path: Path | None = None,
    confirmation_path: Path | None = None,
) -> None:
    """Require exact raw and canonical through-development prefix identity."""

    if development.frame.empty or len(development.frame) >= len(confirmation.frame):
        raise ContextualExpertAggregationExperimentError(
            "Confirmation must strictly extend development"
        )
    expected_index = confirmation.frame.index[: len(development.frame)]
    if not development.frame.index.equals(expected_index):
        raise ContextualExpertAggregationExperimentError(
            "Confirmation session prefix changed"
        )
    if not confirmation.canonical_csv_bytes.startswith(
        development.canonical_csv_bytes
    ):
        raise ContextualExpertAggregationExperimentError(
            "Confirmation canonical price prefix changed"
        )
    if (development_path is None) != (confirmation_path is None):
        raise ContextualExpertAggregationExperimentError(
            "Both raw prefix paths must be supplied together"
        )
    if development_path is not None and confirmation_path is not None:
        try:
            shorter_path = _harden_path(
                development_path,
                label="Development raw prefix",
                require_exists=True,
            )
            longer_path = _harden_path(
                confirmation_path,
                label="Confirmation raw prefix",
                require_exists=True,
            )
            shorter = shorter_path.read_bytes()
            longer = longer_path.read_bytes()
        except OSError as exc:
            raise ContextualExpertAggregationExperimentError(
                "Raw prefix files are unreadable"
            ) from exc
        if not shorter.endswith(b"\n") or not longer.startswith(shorter):
            raise ContextualExpertAggregationExperimentError(
                "Confirmation raw CSV is not the exact development byte prefix"
            )


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _git_text(root: Path, *args: str) -> str:
    return _git_bytes(root, *args).decode("utf-8", errors="strict").strip()


def tracked_file_identity(
    repo_root: Path,
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_git_blob: str | None = None,
    require_literal_local_bytes: bool = False,
    allow_crlf_equivalent: bool = False,
) -> TrackedFileIdentity:
    """Verify HEAD, index, and explicit working-file byte identity."""

    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    source = _harden_path(path, label="Tracked file", require_exists=True)
    try:
        relative = source.relative_to(root).as_posix()
        if allow_crlf_equivalent and not _is_crlf_equivalent_frozen_dependency(
            relative
        ):
            raise ContextualExpertAggregationExperimentError(
                "CRLF equivalence is restricted to the explicit frozen text dependencies"
            )
        _git_bytes(root, "ls-files", "--error-unmatch", "--", relative)
        committed = _git_bytes(root, "show", f"HEAD:{relative}")
        head_blob = _git_text(root, "rev-parse", f"HEAD:{relative}")
        index_line = _git_text(root, "ls-files", "--stage", "--", relative)
        index_blob = index_line.split(maxsplit=2)[1]
        local = source.read_bytes()
    except (
        OSError,
        ValueError,
        IndexError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Required file must be an exact tracked file at HEAD and in the index"
        ) from exc
    if (
        _GIT_OBJECT_RE.fullmatch(head_blob) is None
        or head_blob != index_blob
    ):
        raise ContextualExpertAggregationExperimentError(
            "Required file differs between HEAD and the index"
        )
    if require_literal_local_bytes:
        local_matches = local == committed
    elif allow_crlf_equivalent:
        local_matches = local.replace(b"\r\n", b"\n") == committed
    else:
        local_matches = local == committed
    if require_literal_local_bytes and not local_matches:
        raise ContextualExpertAggregationExperimentError(
            "Required binary input differs byte-for-byte from HEAD"
        )
    if not local_matches:
        raise ContextualExpertAggregationExperimentError(
            "Required working file differs from its permitted HEAD byte identity"
        )
    digest = sha256_bytes(committed)
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, field="expected_sha256"
    ):
        raise ContextualExpertAggregationExperimentError(
            "Required tracked file SHA-256 changed"
        )
    if expected_git_blob is not None and head_blob != expected_git_blob:
        raise ContextualExpertAggregationExperimentError(
            "Required tracked file Git blob changed"
        )
    return TrackedFileIdentity(relative, digest, head_blob)


def _frozen_dependency_paths() -> tuple[Path, ...]:
    paths = tuple(FROZEN_DEPENDENCY_PATHS)
    if not paths:
        raise ContextualExpertAggregationExperimentError(
            "The frozen dependency set must be nonempty"
        )
    normalized: list[Path] = []
    for raw in paths:
        value = Path(raw)
        if value.is_absolute() or ".." in value.parts or value in normalized:
            raise ContextualExpertAggregationExperimentError(
                "The frozen dependency set contains an unsafe or duplicate path"
            )
        normalized.append(value)
    return tuple(normalized)


def _is_crlf_equivalent_frozen_dependency(relative: str) -> bool:
    path = Path(relative)
    return path in _frozen_dependency_paths() and (
        relative in _CRLF_EQUIVALENT_FROZEN_CONTROLS
        or path.suffix.lower() in _CRLF_EQUIVALENT_FROZEN_SUFFIXES
    )


def _git_metadata_identity(
    repo_root: Path, *, expected_branch: str | None = EXPECTED_BRANCH
) -> dict[str, Any]:
    """Read Git metadata only; never scan or hash the full working tree."""

    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    try:
        actual = _harden_path(
            Path(_git_text(root, "rev-parse", "--show-toplevel")),
            label="Git top-level directory",
            require_exists=True,
        )
        branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
        commit = _git_text(root, "rev-parse", "HEAD")
        upstream = _git_text(
            root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        upstream_commit = _git_text(root, "rev-parse", "@{upstream}")
        origin_url = _git_text(root, "remote", "get-url", "origin")
    except (
        OSError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Stage requires a valid Git repository with an origin upstream"
        ) from exc
    if actual != root:
        raise ContextualExpertAggregationExperimentError(
            "repo_root must be the actual non-redirected Git root"
        )
    if expected_branch is not None and branch != expected_branch:
        raise ContextualExpertAggregationExperimentError(
            "Stage is running on the wrong frozen branch"
        )
    expected_upstream = f"origin/{branch}"
    if (
        not branch
        or _GIT_OBJECT_RE.fullmatch(commit) is None
        or upstream != expected_upstream
        or upstream_commit != commit
        or origin_url not in EXPECTED_ORIGIN_URLS
    ):
        raise ContextualExpertAggregationExperimentError(
            "Stage requires attached HEAD, the exact frozen origin repository, "
            "and the origin/current-branch upstream"
        )
    return {
        "branch": branch,
        "commit": commit,
        "upstream": upstream,
        "upstream_remote": "origin",
        "upstream_branch": branch,
        "upstream_commit": upstream_commit,
        "origin_url": origin_url,
        "origin_repository": EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
    }


def _tracked_dependency_identity(repo_root: Path) -> dict[str, dict[str, str]]:
    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    result: dict[str, dict[str, str]] = {}
    for relative in _frozen_dependency_paths():
        identity = tracked_file_identity(
            root,
            root / relative,
            allow_crlf_equivalent=True,
        )
        result[identity.path] = {
            "sha256": identity.sha256,
            "git_blob": identity.git_blob,
        }
    return result


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _path_scoped_prelock_git_identity(repo_root: Path) -> dict[str, Any]:
    """Bind Git and frozen dependencies without a broad status operation."""

    return {
        **_git_metadata_identity(repo_root),
        "dirty": None,
        "cleanliness_scope": "git_metadata_and_frozen_dependency_paths_only",
        "confirmation_input_bytes_opened": False,
        "tracked_dependency_identity": _tracked_dependency_identity(repo_root),
        "runtime_versions": _runtime_versions(),
    }


def _content_aware_git_deltas(repo_root: Path) -> tuple[bytes, bytes, bytes]:
    """Return worktree, index, and untracked paths without stat-only dirt."""

    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    common = (
        "--name-only",
        "-z",
        "--no-ext-diff",
        "--no-textconv",
        "--ignore-submodules=none",
    )
    try:
        worktree = _git_bytes(root, "diff", *common, "--")
        index = _git_bytes(root, "diff", "--cached", *common, "--")
        untracked = _git_bytes(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Stage could not inspect content-aware Git deltas"
        ) from exc
    return worktree, index, untracked


def clean_git_identity(
    repo_root: Path,
    *,
    expected_branch: str | None = EXPECTED_BRANCH,
) -> dict[str, Any]:
    """Require a Git-visible clean tree after the attempt lock exists."""

    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    metadata = _git_metadata_identity(root, expected_branch=expected_branch)
    worktree_delta, index_delta, untracked = _content_aware_git_deltas(root)
    if worktree_delta or index_delta or untracked:
        raise ContextualExpertAggregationExperimentError(
            "Stage requires a Git-visible clean worktree and index"
        )
    return {
        **metadata,
        "dirty": False,
        "cleanliness_scope": (
            "git_visible_index_and_worktree_after_attempt_lock"
        ),
        "tracked_dependency_identity": _tracked_dependency_identity(root),
        "runtime_versions": _runtime_versions(),
    }


def _verify_bundle_identity(
    directory: Path,
    *,
    expected_contract_version: str,
    expected_stage: str,
    expected_manifest_sha256: str | None = None,
    expected_payload_names: set[str] | frozenset[str] | None = None,
    require_stage_pass: bool | None = None,
    repo_root: Path | None = None,
) -> VerifiedBundle:
    """Verify bytes against one already-authorized exact bundle identity."""

    root = _harden_path(
        directory, label="Sealed bundle directory", require_exists=True
    )
    manifest_path = root / "stage_manifest.json"
    checksums_path = root / "checksums.json"
    if not root.is_dir():
        raise ContextualExpertAggregationExperimentError(
            "Sealed bundle directory does not exist"
        )
    try:
        entries = list(root.iterdir())
        if any(_path_is_reparse_point(entry) or not entry.is_file() for entry in entries):
            raise ContextualExpertAggregationExperimentError(
                "Sealed bundle inventory must contain regular flat files only"
            )
        manifest_bytes = manifest_path.read_bytes()
        checksums_bytes = checksums_path.read_bytes()
        manifest_value = json.loads(manifest_bytes.decode("utf-8"))
        checksums_value = json.loads(checksums_bytes.decode("utf-8"))
    except ContextualExpertAggregationExperimentError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Sealed bundle metadata is unreadable"
        ) from exc
    try:
        canonical_manifest_bytes = pretty_json_bytes(manifest_value)
        canonical_checksums_bytes = pretty_json_bytes(checksums_value)
    except (TypeError, ValueError) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Stage seal metadata is not canonical finite JSON"
        ) from exc
    if manifest_bytes != canonical_manifest_bytes:
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest must use exact canonical pretty JSON bytes"
        )
    if checksums_bytes != canonical_checksums_bytes:
        raise ContextualExpertAggregationExperimentError(
            "Stage checksums must use exact canonical pretty JSON bytes"
        )
    if not isinstance(manifest_value, dict):
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest must be a JSON object"
        )
    contract_version = manifest_value.get("contract_version")
    stage = manifest_value.get("stage")
    stage_pass = manifest_value.get("stage_pass")
    run_id = manifest_value.get("run_id")
    if (
        not isinstance(contract_version, str)
        or not contract_version
        or not isinstance(stage, str)
        or _SAFE_STAGE_RE.fullmatch(stage) is None
        or type(stage_pass) is not bool
    ):
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest has invalid contract, stage, or pass-status types"
        )
    _safe_run_id(run_id)
    if run_id != root.name:
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest run_id must exactly equal its directory name"
        )
    if require_stage_pass is not None and type(require_stage_pass) is not bool:
        raise ContextualExpertAggregationExperimentError(
            "require_stage_pass must be an exact boolean when supplied"
        )
    unsigned = dict(manifest_value)
    recorded_self_hash = unsigned.pop("manifest_sha256", None)
    _require_sha256(recorded_self_hash, field="manifest_sha256")
    computed_self_hash = sha256_bytes(canonical_json_bytes(unsigned))
    if recorded_self_hash != computed_self_hash:
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest self-hash is invalid"
        )
    if (
        expected_manifest_sha256 is not None
        and recorded_self_hash
        != _require_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256"
        )
    ):
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest identity changed"
        )
    if manifest_value.get("contract_version") != expected_contract_version:
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest contract version changed"
        )
    if manifest_value.get("stage") != expected_stage:
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest stage changed"
        )
    if (
        require_stage_pass is not None
        and manifest_value.get("stage_pass") is not require_stage_pass
    ):
        raise ContextualExpertAggregationExperimentError(
            "Stage manifest pass status is not authorized"
        )

    payload_hashes = _validated_hash_inventory(
        manifest_value.get("payload_sha256"), field="payload_sha256"
    )
    if "stage_manifest.json" in payload_hashes or "checksums.json" in payload_hashes:
        raise ContextualExpertAggregationExperimentError(
            "Reserved seal metadata may not be a payload"
        )
    if expected_payload_names is not None and set(payload_hashes) != set(
        expected_payload_names
    ):
        raise ContextualExpertAggregationExperimentError(
            "Stage payload inventory differs from the frozen exact inventory"
        )
    checksums = _validated_hash_inventory(checksums_value, field="checksums")
    expected_checksums = {
        **payload_hashes,
        "stage_manifest.json": sha256_bytes(manifest_bytes),
    }
    if checksums != expected_checksums:
        raise ContextualExpertAggregationExperimentError(
            "Checksum inventory does not exactly bind the stage manifest and payloads"
        )
    expected_files = set(expected_checksums) | {"checksums.json"}
    actual_files = {entry.name for entry in entries}
    if actual_files != expected_files:
        raise ContextualExpertAggregationExperimentError(
            "Sealed bundle contains a missing or extra file"
        )
    for filename, expected in checksums.items():
        if sha256_bytes((root / filename).read_bytes()) != expected:
            raise ContextualExpertAggregationExperimentError(
                f"Sealed payload checksum changed: {filename}"
            )
    if repo_root is not None:
        repository = _harden_path(
            repo_root, label="Repository root", require_exists=True
        )
        for filename in sorted(actual_files):
            tracked_file_identity(
                repository,
                root / filename,
                require_literal_local_bytes=True,
            )
    return VerifiedBundle(
        directory=root,
        manifest_path=manifest_path,
        manifest=dict(manifest_value),
        payload_sha256=payload_hashes,
        checksums=checksums,
    )


def verify_exact_bundle(
    directory: Path,
    *,
    expected_contract_version: str,
    expected_stage: str,
    expected_manifest_sha256: str | None = None,
    expected_payload_names: set[str] | frozenset[str] | None = None,
    require_stage_pass: bool | None = None,
    repo_root: Path | None = None,
) -> VerifiedBundle:
    """Verify one explicitly named current-contract stage bundle."""

    if (
        expected_contract_version != CONTRACT_VERSION
        or expected_stage not in ALLOWED_STAGES
    ):
        raise ContextualExpertAggregationExperimentError(
            "Exact bundle verification requires an allowed current-stage identity"
        )
    return _verify_bundle_identity(
        directory,
        expected_contract_version=expected_contract_version,
        expected_stage=expected_stage,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_payload_names=expected_payload_names,
        require_stage_pass=require_stage_pass,
        repo_root=repo_root,
    )


def verify_frozen_bundle_identity(
    repo_root: Path, identity: BundleIdentity
) -> VerifiedBundle:
    """Verify a frozen bundle, including tracked manifest/checksum bytes."""

    repository = _harden_path(
        repo_root, label="Repository root", require_exists=True
    )
    manifest_path = _harden_path(
        repository / identity.manifest_path,
        label="Frozen bundle manifest",
        require_exists=True,
    )
    if identity not in {
        DEVELOPMENT_SOURCE_BUNDLE,
        CONFIRMATION_SOURCE_PARENT_BUNDLE,
        CONFIRMATION_SOURCE_BUNDLE,
    }:
        raise ContextualExpertAggregationExperimentError(
            "Frozen source bundle identity is not preregistered"
        )
    verified = _verify_bundle_identity(
        manifest_path.parent,
        expected_contract_version=identity.contract_version,
        expected_stage=identity.stage,
        expected_manifest_sha256=identity.manifest_self_sha256,
        repo_root=repository,
    )
    if sha256_bytes(manifest_path.read_bytes()) != identity.manifest_file_sha256:
        raise ContextualExpertAggregationExperimentError(
            "Frozen source manifest file hash changed"
        )
    checksums_path = manifest_path.parent / "checksums.json"
    if sha256_bytes(checksums_path.read_bytes()) != identity.checksums_file_sha256:
        raise ContextualExpertAggregationExperimentError(
            "Frozen source checksum file hash changed"
        )
    return verified


def require_parent_link(
    child: VerifiedBundle,
    parent: VerifiedBundle,
    *,
    embedded_parent_filename: str | None = None,
) -> None:
    """Require an exact canonical and optional embedded parent-manifest link."""

    parent_self_hash = parent.manifest.get("manifest_sha256")
    if child.manifest.get("parent_manifest_sha256") != parent_self_hash:
        raise ContextualExpertAggregationExperimentError(
            "Child bundle does not bind the authorized parent manifest"
        )
    if embedded_parent_filename is not None:
        filename = _safe_flat_filename(
            embedded_parent_filename, field="embedded_parent_filename"
        )
        if filename not in child.payload_sha256:
            raise ContextualExpertAggregationExperimentError(
                "Child bundle omits its embedded parent manifest"
            )
        if (
            (child.directory / filename).read_bytes()
            != parent.manifest_path.read_bytes()
        ):
            raise ContextualExpertAggregationExperimentError(
                "Embedded parent manifest differs from the authorized parent"
            )


def require_dependency_continuity(
    current_git_identity: Mapping[str, Any], parent_manifest: Mapping[str, Any]
) -> None:
    current_dependencies = current_git_identity.get(
        "tracked_dependency_identity"
    )
    parent_git = parent_manifest.get("git_identity")
    if not isinstance(parent_git, Mapping):
        raise ContextualExpertAggregationExperimentError(
            "Parent manifest omits Git identity"
        )
    parent_branch = parent_git.get("branch")
    parent_commit = parent_git.get("commit")
    parent_git_is_frozen = (
        parent_branch == current_git_identity.get("branch")
        and parent_git.get("upstream") == f"origin/{parent_branch}"
        and parent_git.get("upstream_remote") == "origin"
        and parent_git.get("upstream_branch") == parent_branch
        and parent_git.get("upstream_commit") == parent_commit
        and isinstance(parent_commit, str)
        and _GIT_OBJECT_RE.fullmatch(parent_commit) is not None
        and parent_git.get("origin_url") in EXPECTED_ORIGIN_URLS
        and parent_git.get("origin_repository") == EXPECTED_ORIGIN_REPOSITORY
        and parent_git.get("head_equals_upstream") is True
    )
    if (
        not isinstance(current_dependencies, Mapping)
        or current_dependencies != parent_git.get("tracked_dependency_identity")
        or current_git_identity.get("runtime_versions")
        != parent_git.get("runtime_versions")
        or not parent_git_is_frozen
    ):
        raise ContextualExpertAggregationExperimentError(
            "Frozen Git origin, code, tests, or runtime changed after the parent stage"
        )


def require_loaded_snapshot_integrity(
    snapshot: LoadedPriceSnapshot, *, expected_spec: AuthorizedPriceSpec
) -> None:
    """Require an attested authorized-loader result, not a constructible value."""

    if (
        not isinstance(snapshot, LoadedPriceSnapshot)
        or snapshot.spec != expected_spec
        or snapshot._loader_attestation is not _AUTHORIZED_LOADER_ATTESTATION
        or snapshot._authorized_lineage is not True
    ):
        raise ContextualExpertAggregationExperimentError(
            "Loaded price snapshot lacks authorized-loader lineage attestation"
        )
    if (
        snapshot.raw_sha256 != expected_spec.raw_sha256
        or not isinstance(snapshot.canonical_csv_bytes, bytes)
        or sha256_bytes(snapshot.canonical_csv_bytes)
        != expected_spec.canonical_sha256
    ):
        raise ContextualExpertAggregationExperimentError(
            "Loaded price snapshot hashes do not match the frozen stage"
        )
    recomputed = canonical_price_csv_bytes(snapshot.frame)
    if recomputed != snapshot.canonical_csv_bytes:
        raise ContextualExpertAggregationExperimentError(
            "Loaded price frame differs from its locked canonical bytes"
        )
    if (
        len(snapshot.frame) != expected_spec.rows
        or snapshot.frame.index.empty
        or snapshot.frame.index[0].date().isoformat()
        != expected_spec.first_session
        or snapshot.frame.index[-1].date().isoformat()
        != expected_spec.last_session
    ):
        raise ContextualExpertAggregationExperimentError(
            "Loaded price frame coverage differs from the frozen stage"
        )
    tracked = snapshot.provenance.get("tracked_input")
    expected_tracked = {
        "path": expected_spec.relative_path.as_posix(),
        "sha256": expected_spec.raw_sha256,
        "git_blob": expected_spec.git_blob,
    }
    if tracked != expected_tracked:
        raise ContextualExpertAggregationExperimentError(
            "Loaded price snapshot lacks exact tracked-input provenance"
        )
    provenance = snapshot.provenance
    session_coverage = provenance.get("session_coverage")
    expected_coverage = {
        "first_session": expected_spec.first_session,
        "last_session": expected_spec.last_session,
        "observations": expected_spec.rows,
        "date_sequence_sha256": expected_spec.date_sequence_sha256,
    }
    expected_lineage = {
        "source_manifest_sha256": (
            None
            if expected_spec.source_bundle is None
            else expected_spec.source_bundle.manifest_self_sha256
        ),
        "source_provenance_sha256": expected_spec.source_provenance_sha256,
        "source_parent_manifest_sha256": (
            None
            if expected_spec.source_parent_bundle is None
            else expected_spec.source_parent_bundle.manifest_self_sha256
        ),
    }
    if (
        provenance.get("source_type") != "physically_bounded_local_csv"
        or provenance.get("network_access") is not False
        or provenance.get("physical_snapshot_has_later_rows") is not False
        or provenance.get("rows_after_bound_returned") is not False
        or provenance.get("raw_sha256") != expected_spec.raw_sha256
        or provenance.get("bounded_first_date") != expected_spec.first_session
        or provenance.get("bounded_last_date") != expected_spec.last_session
        or provenance.get("bounded_rows") != expected_spec.rows
        or provenance.get("bounded_result_sha256")
        != expected_spec.canonical_sha256
        or provenance.get("adjusted_open_formula")
        != "aapl_open * aapl_adj_close / aapl_close"
        or session_coverage != expected_coverage
        or provenance.get("authorized_loader_lineage") != expected_lineage
    ):
        raise ContextualExpertAggregationExperimentError(
            "Loaded price snapshot provenance flags or authorized lineage are invalid"
        )


def _bundle_json_object(bundle: VerifiedBundle, filename: str) -> dict[str, Any]:
    try:
        value = json.loads((bundle.directory / filename).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationExperimentError(
            f"Development authorization payload is unreadable: {filename}"
        ) from exc
    if not isinstance(value, dict):
        raise ContextualExpertAggregationExperimentError(
            f"Development authorization payload must be an object: {filename}"
        )
    return value


def _validate_development_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Require the artifact-owned composite model plus account checkpoint."""

    try:
        parsed = parse_composite_stage_checkpoint(checkpoint)
    except (
        ContextualExpertAggregationArtifactError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Development checkpoint is not the exact composite artifact checkpoint"
        ) from exc
    if (
        parsed.stage != DEVELOPMENT_STAGE
        or parsed.checkpoint_schema_version
        != COMPOSITE_CHECKPOINT_SCHEMA_VERSION
        or tuple(parsed.replay_checkpoints) != ARM_ORDER
        or tuple(parsed.administrative_accounts) != COST_ORDER
        or any(
            tuple(parsed.administrative_accounts[cost]) != POLICY_ORDER
            for cost in COST_ORDER
        )
    ):
        raise ContextualExpertAggregationExperimentError(
            "Development checkpoint differs from the frozen composite ordering"
        )


def _registered_development_verification(
    bundle: VerifiedBundle,
    *,
    current_git_identity: Mapping[str, Any],
) -> DevelopmentVerificationEvidence:
    registration = _development_verifier_registration()
    if registration is None:
        raise ContextualExpertAggregationExperimentError(
            "Exact development authorization verifier is not registered"
        )
    dependency_path = Path(registration.dependency_path)
    if (
        dependency_path != VERIFIER_IMPLEMENTATION_PATH
        or dependency_path not in _frozen_dependency_paths()
    ):
        raise ContextualExpertAggregationExperimentError(
            "Exact development verifier dependency is not frozen"
        )
    expected_names = frozenset(registration.expected_payload_names)
    if (
        not registration.verifier_id
        or expected_names != DEVELOPMENT_AUTHORIZATION_REQUIRED_PAYLOADS
        or set(bundle.payload_sha256) != set(expected_names)
    ):
        raise ContextualExpertAggregationExperimentError(
            "Exact development verifier has an invalid frozen payload inventory"
        )
    dependency_identity = current_git_identity.get("tracked_dependency_identity")
    dependency_key = dependency_path.as_posix()
    if (
        not isinstance(dependency_identity, Mapping)
        or dependency_key not in dependency_identity
    ):
        raise ContextualExpertAggregationExperimentError(
            "Exact development verifier dependency identity is not bound"
        )
    evidence = registration.verify(bundle, DEVELOPMENT_PRICE_SPEC)
    if not isinstance(evidence, DevelopmentVerificationEvidence):
        raise ContextualExpertAggregationExperimentError(
            "Exact development verifier returned an invalid evidence type"
        )
    expected_sorted_names = tuple(sorted(expected_names))
    if (
        evidence.verifier_id != registration.verifier_id
        or evidence.verifier_dependency_path != dependency_key
        or evidence.passed is not True
        or evidence.exact_payload_names != expected_sorted_names
        or evidence.report_sha256
        != bundle.payload_sha256[DEVELOPMENT_REPORT_FILENAME]
        or evidence.checkpoint_sha256
        != bundle.payload_sha256[DEVELOPMENT_CHECKPOINT_FILENAME]
        or evidence.gate_report_sha256
        != bundle.payload_sha256[DEVELOPMENT_GATE_REPORT_FILENAME]
    ):
        raise ContextualExpertAggregationExperimentError(
            "Exact development verifier evidence does not bind the sealed bundle"
        )
    _require_sha256(
        evidence.semantic_evidence_sha256, field="semantic_evidence_sha256"
    )
    return evidence


def _development_verification_mapping(
    evidence: DevelopmentVerificationEvidence,
    *,
    current_git_identity: Mapping[str, Any],
) -> dict[str, Any]:
    dependency = current_git_identity["tracked_dependency_identity"][
        evidence.verifier_dependency_path
    ]
    return {
        "verifier_id": evidence.verifier_id,
        "verifier_dependency_path": evidence.verifier_dependency_path,
        "verifier_dependency_identity": dict(dependency),
        "exact_payload_names": list(evidence.exact_payload_names),
        "report_sha256": evidence.report_sha256,
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "gate_report_sha256": evidence.gate_report_sha256,
        "semantic_evidence_sha256": evidence.semantic_evidence_sha256,
        "passed": True,
    }


def _validate_development_authorization_bundle(
    bundle: VerifiedBundle, *, current_git_identity: Mapping[str, Any]
) -> DevelopmentVerificationEvidence:
    manifest = bundle.manifest
    if (
        manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("stage") != "development"
        or manifest.get("stage_pass") is not True
        or set(bundle.payload_sha256)
        != set(DEVELOPMENT_AUTHORIZATION_REQUIRED_PAYLOADS)
    ):
        raise ContextualExpertAggregationExperimentError(
            "Confirmation requires the exact passing development bundle"
        )
    _safe_run_id(manifest.get("run_id"))
    require_dependency_continuity(current_git_identity, manifest)
    if (
        manifest.get("bounded_result_sha256")
        != DEVELOPMENT_PRICE_SPEC.canonical_sha256
        or bundle.payload_sha256.get(DEVELOPMENT_PRICE_FILENAME)
        != DEVELOPMENT_PRICE_SPEC.canonical_sha256
    ):
        raise ContextualExpertAggregationExperimentError(
            "Development bundle does not bind the frozen through-2018 prices"
        )
    source = manifest.get("source_provenance")
    expected_tracked_input = {
        "path": DEVELOPMENT_PRICE_SPEC.relative_path.as_posix(),
        "sha256": DEVELOPMENT_PRICE_SPEC.raw_sha256,
        "git_blob": DEVELOPMENT_PRICE_SPEC.git_blob,
    }
    if (
        not isinstance(source, Mapping)
        or source.get("bounded_first_date")
        != DEVELOPMENT_PRICE_SPEC.first_session
        or source.get("bounded_last_date")
        != DEVELOPMENT_PRICE_SPEC.last_session
        or source.get("bounded_rows") != DEVELOPMENT_PRICE_SPEC.rows
        or source.get("bounded_result_sha256")
        != DEVELOPMENT_PRICE_SPEC.canonical_sha256
        or source.get("network_access") is not False
        or source.get("physical_snapshot_has_later_rows") is not False
        or source.get("rows_after_bound_returned") is not False
        or source.get("tracked_input") != expected_tracked_input
    ):
        raise ContextualExpertAggregationExperimentError(
            "Development source provenance is not the frozen physical snapshot"
        )

    input_provenance = _bundle_json_object(bundle, "input_provenance.json")
    if input_provenance != dict(source):
        raise ContextualExpertAggregationExperimentError(
            "Development input provenance payload differs from its manifest"
        )
    gate = _bundle_json_object(bundle, DEVELOPMENT_GATE_REPORT_FILENAME)
    report = _bundle_json_object(bundle, DEVELOPMENT_REPORT_FILENAME)
    checkpoint = _bundle_json_object(bundle, DEVELOPMENT_CHECKPOINT_FILENAME)
    if gate.get("passed") is not True:
        raise ContextualExpertAggregationExperimentError(
            "Development gate report is not an exact pass"
        )
    if (
        report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != "development"
        or report.get("run_id") != manifest.get("run_id")
        or report.get("gate_report") != gate
    ):
        raise ContextualExpertAggregationExperimentError(
            "Development manifest, report, and gates disagree"
        )
    _validate_development_checkpoint(checkpoint)
    return _registered_development_verification(
        bundle,
        current_git_identity=current_git_identity,
    )


def _git_common_directory(repo_root: Path) -> Path:
    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    try:
        raw = Path(_git_text(root, "rev-parse", "--git-common-dir"))
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Could not locate the shared Git evidence registry"
        ) from exc
    candidate = raw if raw.is_absolute() else root / raw
    common = _harden_path(
        candidate, label="Git common directory", require_exists=True
    )
    if not common.is_dir():
        raise ContextualExpertAggregationExperimentError(
            "Git common directory is not a directory"
        )
    return common


def _fsync_directory(directory: Path) -> bool:
    """Durably flush one directory on POSIX or Windows."""

    path = _harden_path(
        directory, label="Durability directory", require_exists=True
    )
    if not path.is_dir():
        raise ContextualExpertAggregationExperimentError(
            "Durability path must be an existing directory"
        )
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            flush_file_buffers = kernel32.FlushFileBuffers
            flush_file_buffers.argtypes = (wintypes.HANDLE,)
            flush_file_buffers.restype = wintypes.BOOL
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL

            generic_write = 0x40000000
            share_read_write_delete = 0x00000001 | 0x00000002 | 0x00000004
            open_existing = 3
            file_flag_backup_semantics = 0x02000000
            invalid_handle = ctypes.c_void_p(-1).value
            handle = create_file(
                str(path),
                generic_write,
                share_read_write_delete,
                None,
                open_existing,
                file_flag_backup_semantics,
                None,
            )
            if handle == invalid_handle:
                raise OSError(
                    ctypes.get_last_error(),
                    "CreateFileW could not open the directory for flushing",
                )
            try:
                if not flush_file_buffers(handle):
                    raise OSError(
                        ctypes.get_last_error(),
                        "FlushFileBuffers could not flush the directory",
                    )
            finally:
                close_handle(handle)
            return True
        except (AttributeError, OSError) as exc:
            raise ContextualExpertAggregationExperimentError(
                "Could not durably flush the Windows evidence directory"
            ) from exc

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
        return True
    except OSError as exc:
        raise ContextualExpertAggregationExperimentError(
            "Could not fsync the durable evidence directory"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _ensure_confirmation_registry_parent(repo_root: Path) -> Path:
    common = _git_common_directory(repo_root)
    parent_identity = _directory_identity(common, label="Git common directory")
    cursor = common
    for component in CONFIRMATION_REGISTRY_DIRECTORY.parts:
        child = cursor / component
        if child.exists() or _path_is_reparse_point(child):
            hardened = _harden_path(
                child,
                label="Confirmation attempt registry directory",
                require_exists=True,
            )
            if not hardened.is_dir():
                raise ContextualExpertAggregationExperimentError(
                    "Confirmation attempt registry path is not a directory"
                )
        else:
            child.mkdir()
            _fsync_directory(cursor)
        cursor = _harden_path(
            child,
            label="Confirmation attempt registry directory",
            require_exists=True,
        )
    _require_same_directory_identity(
        common, parent_identity, label="Git common directory"
    )
    return cursor


def _attempt_identity_sha256(development_manifest_sha256: str) -> str:
    identity = {
        "contract_version": CONTRACT_VERSION,
        "stage": CONFIRMATION_ATTEMPT_STAGE,
        "development_manifest_sha256": _require_sha256(
            development_manifest_sha256,
            field="development_manifest_sha256",
        ),
    }
    return sha256_bytes(canonical_json_bytes(identity))


def _canonical_confirmation_lock_path(
    repo_root: Path, *, development_manifest_sha256: str
) -> Path:
    parent = _ensure_confirmation_registry_parent(repo_root)
    attempt_identity = _attempt_identity_sha256(development_manifest_sha256)
    return parent / f"confirmation-attempt-{attempt_identity.removeprefix('sha256:')}.json"


def _attempt_lock_value(
    *,
    repo_root: Path,
    development_manifest_path: Path,
    development_manifest_sha256: str,
    development_checkpoint_sha256: str,
    development_gate_report_sha256: str,
    prelock_git_identity: Mapping[str, Any],
    development_verification: Mapping[str, Any],
) -> dict[str, Any]:
    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    manifest = _harden_path(
        development_manifest_path,
        label="Development manifest",
        require_exists=True,
    )
    try:
        relative_manifest = manifest.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContextualExpertAggregationExperimentError(
            "Development manifest must be inside the repository"
        ) from exc
    attempt_identity = _attempt_identity_sha256(development_manifest_sha256)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": CONFIRMATION_ATTEMPT_STAGE,
        "status": "attempt_consumed_before_confirmation_price_read",
        "attempt_identity_sha256": attempt_identity,
        "identity_basis": [
            "contract_version",
            "stage",
            "development_manifest_sha256",
        ],
        "development_manifest_path": relative_manifest,
        "development_manifest_sha256": _require_sha256(
            development_manifest_sha256,
            field="development_manifest_sha256",
        ),
        "development_checkpoint_sha256": _require_sha256(
            development_checkpoint_sha256,
            field="development_checkpoint_sha256",
        ),
        "development_gate_report_sha256": _require_sha256(
            development_gate_report_sha256,
            field="development_gate_report_sha256",
        ),
        "prelock_git_identity_sha256": sha256_bytes(
            canonical_json_bytes(prelock_git_identity)
        ),
        "development_verification": dict(development_verification),
        "confirmation_input_bytes_opened_before_lock": False,
        "postlock_complete_cleanliness_required": True,
    }
    return {
        **payload,
        "attempt_lock_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _require_prelock_postlock_identity(
    prelock: Mapping[str, Any], postlock: Mapping[str, Any]
) -> None:
    exact_fields = (
        "branch",
        "commit",
        "upstream",
        "upstream_remote",
        "upstream_branch",
        "upstream_commit",
        "origin_url",
        "origin_repository",
        "head_equals_upstream",
        "tracked_dependency_identity",
        "runtime_versions",
    )
    if any(prelock.get(name) != postlock.get(name) for name in exact_fields):
        raise ContextualExpertAggregationExperimentError(
            "Git or frozen dependencies changed across confirmation locking"
        )


def authorize_confirmation_attempt(
    *,
    repo_root: Path,
    development_manifest_path: Path,
) -> ConfirmationAuthorization:
    """Consume the one canonical parent-bound attempt before confirmation access."""

    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    prelock_git_identity = _path_scoped_prelock_git_identity(root)
    manifest_path = _harden_path(
        development_manifest_path,
        label="Development manifest",
        require_exists=True,
    )
    if manifest_path.name != "stage_manifest.json":
        raise ContextualExpertAggregationExperimentError(
            "Development authorization must name stage_manifest.json"
        )
    parent = verify_exact_bundle(
        manifest_path.parent,
        expected_contract_version=CONTRACT_VERSION,
        expected_stage="development",
        expected_payload_names=DEVELOPMENT_AUTHORIZATION_REQUIRED_PAYLOADS,
        require_stage_pass=True,
        repo_root=root,
    )
    evidence = _validate_development_authorization_bundle(
        parent,
        current_git_identity=prelock_git_identity,
    )
    verification = _development_verification_mapping(
        evidence, current_git_identity=prelock_git_identity
    )
    parent_manifest_sha = str(parent.manifest["manifest_sha256"])
    lock = _canonical_confirmation_lock_path(
        root, development_manifest_sha256=parent_manifest_sha
    )
    lock_parent_identity = _directory_identity(
        lock.parent, label="Confirmation attempt lock parent"
    )
    if lock.exists() or _path_is_reparse_point(lock):
        raise ContextualExpertAggregationExperimentError(
            "Confirmation attempt has already been consumed"
        )
    lock_value = _attempt_lock_value(
        repo_root=root,
        development_manifest_path=manifest_path,
        development_manifest_sha256=parent_manifest_sha,
        development_checkpoint_sha256=evidence.checkpoint_sha256,
        development_gate_report_sha256=evidence.gate_report_sha256,
        prelock_git_identity=prelock_git_identity,
        development_verification=verification,
    )
    lock_bytes = pretty_json_bytes(lock_value)
    _exclusive_write(lock, lock_bytes)
    parent_fsync_supported = _fsync_directory(lock.parent)
    if parent_fsync_supported is not True:
        raise ContextualExpertAggregationExperimentError(
            "Confirmation attempt lock parent directory could not be fsynced"
        )
    _require_same_directory_identity(
        lock.parent,
        lock_parent_identity,
        label="Confirmation attempt lock parent",
    )
    try:
        persisted = _harden_path(
            lock, label="Confirmation attempt lock", require_exists=True
        ).read_bytes()
    except OSError as exc:
        raise ContextualExpertAggregationExperimentError(
            "Durable confirmation attempt lock cannot be read back"
        ) from exc
    if persisted != lock_bytes:
        raise ContextualExpertAggregationExperimentError(
            "Durable confirmation attempt lock changed during creation"
        )

    # A broad clean-tree proof is intentionally delayed until the canonical
    # lock is durable.  If it fails, the consumed lock remains as evidence.
    git_identity = clean_git_identity(root, expected_branch=EXPECTED_BRANCH)
    _require_prelock_postlock_identity(prelock_git_identity, git_identity)
    return ConfirmationAuthorization(
        development_manifest_path=manifest_path,
        development_manifest_sha256=parent_manifest_sha,
        development_checkpoint_sha256=evidence.checkpoint_sha256,
        development_gate_report_sha256=evidence.gate_report_sha256,
        git_identity=dict(git_identity),
        prelock_git_identity=dict(prelock_git_identity),
        development_verification=verification,
        attempt_identity_sha256=_attempt_identity_sha256(parent_manifest_sha),
        attempt_lock_path=lock,
        attempt_lock_sha256=sha256_bytes(lock_bytes),
        attempt_lock_parent_fsync_supported=parent_fsync_supported,
    )


def verify_confirmation_authorization(
    repo_root: Path, authorization: ConfirmationAuthorization
) -> VerifiedBundle:
    """Revalidate canonical lock, exact parent, dependencies, and cleanliness."""

    if not isinstance(authorization, ConfirmationAuthorization):
        raise ContextualExpertAggregationExperimentError(
            "Confirmation requires a verified authorization object"
        )
    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    expected_lock = _canonical_confirmation_lock_path(
        root,
        development_manifest_sha256=authorization.development_manifest_sha256,
    )
    actual_lock = _harden_path(
        authorization.attempt_lock_path,
        label="Confirmation attempt lock",
        require_exists=True,
    )
    if actual_lock != expected_lock or not actual_lock.is_file():
        raise ContextualExpertAggregationExperimentError(
            "Confirmation authorization does not use the canonical attempt lock"
        )
    current_git_identity = clean_git_identity(
        root, expected_branch=EXPECTED_BRANCH
    )
    if current_git_identity != dict(authorization.git_identity):
        raise ContextualExpertAggregationExperimentError(
            "Confirmation Git identity or clean state changed after authorization"
        )
    parent = verify_exact_bundle(
        authorization.development_manifest_path.parent,
        expected_contract_version=CONTRACT_VERSION,
        expected_stage="development",
        expected_manifest_sha256=authorization.development_manifest_sha256,
        expected_payload_names=DEVELOPMENT_AUTHORIZATION_REQUIRED_PAYLOADS,
        require_stage_pass=True,
        repo_root=root,
    )
    evidence = _validate_development_authorization_bundle(
        parent,
        current_git_identity=current_git_identity,
    )
    verification = _development_verification_mapping(
        evidence, current_git_identity=current_git_identity
    )
    if (
        evidence.checkpoint_sha256
        != authorization.development_checkpoint_sha256
        or evidence.gate_report_sha256
        != authorization.development_gate_report_sha256
        or verification != dict(authorization.development_verification)
        or _attempt_identity_sha256(authorization.development_manifest_sha256)
        != authorization.attempt_identity_sha256
    ):
        raise ContextualExpertAggregationExperimentError(
            "Exact development verification changed after authorization"
        )
    try:
        lock_bytes = actual_lock.read_bytes()
        lock_value = json.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Durable confirmation attempt lock is missing or unreadable"
        ) from exc
    expected_value = _attempt_lock_value(
        repo_root=root,
        development_manifest_path=authorization.development_manifest_path,
        development_manifest_sha256=authorization.development_manifest_sha256,
        development_checkpoint_sha256=authorization.development_checkpoint_sha256,
        development_gate_report_sha256=authorization.development_gate_report_sha256,
        prelock_git_identity=authorization.prelock_git_identity,
        development_verification=authorization.development_verification,
    )
    if (
        not isinstance(lock_value, dict)
        or lock_value != expected_value
        or sha256_bytes(lock_bytes) != authorization.attempt_lock_sha256
    ):
        raise ContextualExpertAggregationExperimentError(
            "Durable confirmation attempt lock does not match authorization"
        )
    return parent


def verify_authorized_price_lineage(
    repo_root: Path, spec: AuthorizedPriceSpec
) -> dict[str, Any]:
    """Verify the existing sealed provenance for an authorized price input."""

    _validate_price_spec(spec)
    if spec.source_bundle is None or spec.source_provenance_sha256 is None:
        raise ContextualExpertAggregationExperimentError(
            "Authorized price spec omits frozen source lineage"
        )
    source = verify_frozen_bundle_identity(repo_root, spec.source_bundle)
    provenance_name = _safe_flat_filename(
        spec.source_provenance_filename,
        field="source_provenance_filename",
    )
    if source.payload_sha256.get(provenance_name) != spec.source_provenance_sha256:
        raise ContextualExpertAggregationExperimentError(
            "Source bundle does not bind the frozen input provenance"
        )
    try:
        provenance_bytes = (source.directory / provenance_name).read_bytes()
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationExperimentError(
            "Frozen input provenance is unreadable"
        ) from exc
    if (
        not isinstance(provenance, dict)
        or sha256_bytes(provenance_bytes) != spec.source_provenance_sha256
        or source.manifest.get("source_provenance") != provenance
        or provenance.get("bounded_first_date") != spec.first_session
        or provenance.get("bounded_last_date") != spec.last_session
        or provenance.get("bounded_rows") != spec.rows
        or provenance.get("bounded_result_sha256") != spec.canonical_sha256
        or provenance.get("physical_snapshot_has_later_rows") is not False
        or provenance.get("rows_after_bound_returned") is not False
        or provenance.get("network_access") is not False
        or provenance.get("tracked_input")
        != {"path": spec.relative_path.as_posix(), "sha256": spec.raw_sha256}
    ):
        raise ContextualExpertAggregationExperimentError(
            "Frozen input provenance no longer matches the authorized price spec"
        )
    if spec.source_parent_bundle is not None:
        parent = verify_frozen_bundle_identity(
            repo_root, spec.source_parent_bundle
        )
        require_parent_link(
            source,
            parent,
            embedded_parent_filename=spec.embedded_parent_filename,
        )
    elif source.manifest.get("parent_manifest_sha256") is not None:
        raise ContextualExpertAggregationExperimentError(
            "Authorized source unexpectedly declares an unverified parent"
        )
    return provenance


def load_authorized_prices(
    repo_root: Path,
    *,
    stage: str,
    development_snapshot: LoadedPriceSnapshot | None = None,
    confirmation_authorization: ConfirmationAuthorization | None = None,
) -> LoadedPriceSnapshot:
    """Verify lineage/tracking, then load an exact authorized local stage."""

    try:
        spec = AUTHORIZED_PRICE_SPECS[stage]
    except KeyError as exc:
        raise ContextualExpertAggregationExperimentError(
            "Only development and confirmation price stages are authorized"
        ) from exc
    if stage == "confirmation":
        if development_snapshot is None:
            raise ContextualExpertAggregationExperimentError(
                "Confirmation requires the exact frozen development snapshot"
            )
        require_loaded_snapshot_integrity(
            development_snapshot,
            expected_spec=DEVELOPMENT_PRICE_SPEC,
        )
        if confirmation_authorization is None:
            raise ContextualExpertAggregationExperimentError(
                "Confirmation requires a durable verified attempt authorization"
            )
    elif development_snapshot is not None:
        raise ContextualExpertAggregationExperimentError(
            "Development does not accept a prior price snapshot"
        )
    elif confirmation_authorization is not None:
        raise ContextualExpertAggregationExperimentError(
            "Development does not accept a confirmation authorization"
        )
    root = _harden_path(repo_root, label="Repository root", require_exists=True)
    if stage == "confirmation":
        assert confirmation_authorization is not None
        # Revalidate the committed parent and durable lock before the first
        # operation capable of opening the through-2023 price input.
        verify_confirmation_authorization(root, confirmation_authorization)
    tracked = tracked_file_identity(
        root,
        root / spec.relative_path,
        expected_sha256=spec.raw_sha256,
        expected_git_blob=spec.git_blob,
        require_literal_local_bytes=True,
    )
    verify_authorized_price_lineage(root, spec)
    snapshot = load_bounded_price_snapshot(root / spec.relative_path, spec=spec)
    provenance = {
        **dict(snapshot.provenance),
        "tracked_input": {
            "path": tracked.path,
            "sha256": tracked.sha256,
            "git_blob": tracked.git_blob,
        },
        "authorized_loader_lineage": {
            "source_manifest_sha256": (
                None
                if spec.source_bundle is None
                else spec.source_bundle.manifest_self_sha256
            ),
            "source_provenance_sha256": spec.source_provenance_sha256,
            "source_parent_manifest_sha256": (
                None
                if spec.source_parent_bundle is None
                else spec.source_parent_bundle.manifest_self_sha256
            ),
        },
    }
    snapshot = _attest_loaded_snapshot(
        LoadedPriceSnapshot(
            spec=snapshot.spec,
            frame=snapshot.frame,
            raw_sha256=snapshot.raw_sha256,
            canonical_csv_bytes=snapshot.canonical_csv_bytes,
            provenance=provenance,
        ),
        authorized_lineage=True,
    )
    require_loaded_snapshot_integrity(snapshot, expected_spec=spec)
    if stage == "confirmation":
        assert development_snapshot is not None
        require_snapshot_prefix(
            development_snapshot,
            snapshot,
            development_path=root / DEVELOPMENT_PRICE_SPEC.relative_path,
            confirmation_path=root / CONFIRMATION_PRICE_SPEC.relative_path,
        )
    return snapshot


def _exclusive_write(path: Path, payload: bytes) -> None:
    destination = _lexical_absolute(path)
    _harden_path(
        destination.parent,
        label="Exclusive-write parent",
        require_exists=True,
    )
    if destination.exists() or _path_is_reparse_point(destination):
        raise ContextualExpertAggregationExperimentError(
            f"Could not durably write sealed artifact: {destination.name}"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(destination, flags, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short durable write")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise ContextualExpertAggregationExperimentError(
            f"Could not durably write sealed artifact: {destination.name}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def seal_exact_bundle(
    final_directory: Path,
    *,
    manifest_fields: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    expected_payload_names: set[str] | frozenset[str],
    deadline: StageDeadline,
    before_promote: Callable[[], None] | None = None,
) -> SealedBundle:
    """Write, independently verify, then atomically promote one exact bundle."""

    deadline.check("seal entry")
    final = _lexical_absolute(final_directory)
    parent = _harden_path(
        final.parent, label="Seal destination parent", require_exists=True
    )
    temporary_root = parent / f".{final.name}.sealing"
    temporary = temporary_root / final.name
    parent_identity = _directory_identity(parent, label="Seal destination parent")
    _harden_path(final, label="Final run directory")
    _harden_path(temporary_root, label="Temporary sealing directory")
    if not parent.is_dir():
        raise ContextualExpertAggregationExperimentError(
            "Seal destination parent must already exist"
        )
    if final.exists() or _path_is_reparse_point(final):
        raise ContextualExpertAggregationExperimentError(
            "Immutable final run directory already exists"
        )
    if temporary_root.exists() or _path_is_reparse_point(temporary_root):
        raise ContextualExpertAggregationExperimentError(
            "Stale sealing directory already exists"
        )
    if not isinstance(manifest_fields, Mapping) or not isinstance(payloads, Mapping):
        raise ContextualExpertAggregationExperimentError(
            "Seal metadata and payloads must be mappings"
        )
    if (
        "manifest_sha256" in manifest_fields
        or "payload_sha256" in manifest_fields
        or "contract_version" in manifest_fields
    ):
        raise ContextualExpertAggregationExperimentError(
            "Seal helper owns contract, manifest, and payload hashes"
        )
    stage = manifest_fields.get("stage")
    stage_pass = manifest_fields.get("stage_pass")
    run_id = manifest_fields.get("run_id")
    if stage not in ALLOWED_STAGES:
        raise ContextualExpertAggregationExperimentError(
            "Seal stage must be development or confirmation"
        )
    if type(stage_pass) is not bool:
        raise ContextualExpertAggregationExperimentError(
            "Seal stage_pass must be an exact boolean"
        )
    _safe_run_id(run_id)
    if final.name != run_id:
        raise ContextualExpertAggregationExperimentError(
            "Seal run_id must exactly match its immutable directory name"
        )
    if not isinstance(expected_payload_names, (set, frozenset)):
        raise ContextualExpertAggregationExperimentError(
            "Seal requires an explicit exact payload-name inventory"
        )
    expected_names = {
        _safe_flat_filename(name, field="expected_payload_names")
        for name in expected_payload_names
    }
    if any(
        name in {"stage_manifest.json", "checksums.json"}
        for name in expected_names
    ):
        raise ContextualExpertAggregationExperimentError(
            "Expected payload inventory contains reserved seal metadata"
        )
    clean_payloads: dict[str, bytes] = {}
    for filename, payload in payloads.items():
        name = _safe_flat_filename(filename, field="payloads")
        if name in {"stage_manifest.json", "checksums.json"}:
            raise ContextualExpertAggregationExperimentError(
                "Payload uses a reserved seal filename"
            )
        if not isinstance(payload, bytes):
            raise ContextualExpertAggregationExperimentError(
                "Every sealed payload must already be exact bytes"
            )
        clean_payloads[name] = payload
    if set(clean_payloads) != expected_names:
        raise ContextualExpertAggregationExperimentError(
            "Payloads do not match the explicit exact inventory"
        )
    payload_hashes = {
        name: sha256_bytes(payload)
        for name, payload in sorted(clean_payloads.items())
    }
    manifest = self_hashed_manifest(
        {
            "contract_version": CONTRACT_VERSION,
            **dict(manifest_fields),
            "payload_sha256": payload_hashes,
        }
    )
    manifest_bytes = pretty_json_bytes(manifest)
    checksums = {
        **payload_hashes,
        "stage_manifest.json": sha256_bytes(manifest_bytes),
    }
    checksums_bytes = pretty_json_bytes(checksums)
    deadline.check("after seal construction")

    temporary_root.mkdir()
    temporary.mkdir()
    temporary_root_identity = _directory_identity(
        temporary_root, label="Temporary sealing directory"
    )
    temporary_identity = _directory_identity(
        temporary, label="Temporary bundle directory"
    )
    promoted = False
    try:
        for filename, payload in clean_payloads.items():
            _exclusive_write(temporary / filename, payload)
        _exclusive_write(temporary / "stage_manifest.json", manifest_bytes)
        _exclusive_write(temporary / "checksums.json", checksums_bytes)
        if before_promote is not None:
            before_promote()
        _require_same_directory_identity(
            parent, parent_identity, label="Seal destination parent"
        )
        _require_same_directory_identity(
            temporary_root,
            temporary_root_identity,
            label="Temporary sealing directory",
        )
        _require_same_directory_identity(
            temporary,
            temporary_identity,
            label="Temporary bundle directory",
        )
        _harden_path(final, label="Final run directory")
        if final.exists() or _path_is_reparse_point(final):
            raise ContextualExpertAggregationExperimentError(
                "Final run directory appeared during sealing"
            )
        # A callback may inspect consistency, but it is not trusted to leave
        # the private directory unchanged.  Verify exact bytes after it.
        verified = verify_exact_bundle(
            temporary,
            expected_contract_version=CONTRACT_VERSION,
            expected_stage=stage,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_payload_names=expected_names,
        )
        deadline.check("immediately before final artifact promotion")
        _require_same_directory_identity(
            parent, parent_identity, label="Seal destination parent"
        )
        _require_same_directory_identity(
            temporary_root,
            temporary_root_identity,
            label="Temporary sealing directory",
        )
        _require_same_directory_identity(
            temporary,
            temporary_identity,
            label="Temporary bundle directory",
        )
        os.replace(temporary, final)
        _fsync_directory(parent)
        temporary_root.rmdir()
        promoted = True
    finally:
        if (
            not promoted
            and temporary_root.exists()
            and not _path_is_reparse_point(temporary_root)
            and not _path_is_reparse_point(temporary)
        ):
            try:
                _require_same_directory_identity(
                    parent, parent_identity, label="Seal destination parent"
                )
                shutil.rmtree(temporary_root)
            except ContextualExpertAggregationExperimentError:
                # Never follow or erase through a redirected destination.
                pass
    return SealedBundle(
        directory=final,
        manifest_path=final / "stage_manifest.json",
        manifest=dict(verified.manifest),
        checksums=dict(verified.checksums),
    )


__all__ = [
    "ALLOWED_STAGES",
    "ARM_ORDER",
    "ARTIFACTS_IMPLEMENTATION_PATH",
    "ARTIFACTS_TEST_PATH",
    "AUTHORIZED_PRICE_SPECS",
    "AuthorizedPriceSpec",
    "BundleIdentity",
    "CANONICAL_PRICE_COLUMNS",
    "CONFIRMATION_END",
    "CONFIRMATION_PRICE_SPEC",
    "CONFIRMATION_REGISTRY_DIRECTORY",
    "ConfirmationAuthorization",
    "CONTRACT_VERSION",
    "COMPOSITE_CHECKPOINT_SCHEMA_VERSION",
    "COST_ORDER",
    "ContextualExpertAggregationExperimentError",
    "DEVELOPMENT_CHECKPOINT_FILENAME",
    "DEVELOPMENT_END",
    "DEVELOPMENT_AUTHORIZATION_REQUIRED_PAYLOADS",
    "DEVELOPMENT_GATE_REPORT_FILENAME",
    "DEVELOPMENT_PRICE_FILENAME",
    "DEVELOPMENT_PRICE_SPEC",
    "DEVELOPMENT_PAYLOAD_NAMES",
    "DEVELOPMENT_REPORT_FILENAME",
    "DevelopmentVerificationEvidence",
    "DevelopmentVerifierRegistration",
    "EVALUATION_IMPLEMENTATION_PATH",
    "EVALUATION_TEST_PATH",
    "EXPECTED_BRANCH",
    "EXPECTED_ORIGIN_REPOSITORY",
    "EXPECTED_ORIGIN_URLS",
    "FROZEN_DEPENDENCY_PATHS",
    "FIXED_POLICY_ORDER",
    "LEDGER_IMPLEMENTATION_PATH",
    "LEDGER_TEST_PATH",
    "LoadedPriceSnapshot",
    "PHYSICAL_PRICE_COLUMNS",
    "POLICY_ORDER",
    "README_PATH",
    "REPLAY_IMPLEMENTATION_PATH",
    "REPLAY_TEST_PATH",
    "RUN_TIME_LIMIT_SECONDS",
    "SealedBundle",
    "StageDeadline",
    "STAGE_IMPLEMENTATION_PATH",
    "STAGE_TEST_PATH",
    "TrackedFileIdentity",
    "VerifiedBundle",
    "canonical_json_bytes",
    "canonical_price_csv_bytes",
    "authorize_confirmation_attempt",
    "clean_git_identity",
    "load_authorized_prices",
    "load_bounded_price_snapshot",
    "pretty_json_bytes",
    "require_dependency_continuity",
    "require_loaded_snapshot_integrity",
    "require_parent_link",
    "require_snapshot_prefix",
    "seal_exact_bundle",
    "self_hashed_manifest",
    "sha256_bytes",
    "tracked_file_identity",
    "verify_authorized_price_lineage",
    "verify_confirmation_authorization",
    "verify_exact_bundle",
    "verify_frozen_bundle_identity",
    "VERIFIER_IMPLEMENTATION_PATH",
    "VERIFIER_TEST_PATH",
]
