"""Sealed stage orchestration for the SEC/Gemma online overlay.

The runner is deliberately a parent coordinator rather than a network, model,
or market client.  Effectful work is delegated to one injected phase executor
that receives an opaque permit only after the corresponding one-shot attempt
has been durably consumed.  The parent samples its own monotonic clock around
every phase, applies the frozen budgets, validates all deterministic artifacts,
and releases semantic results only after one joint report is committed and
externally pinned.

Production adapters are intentionally narrow.  In particular, acquisition is
represented by quarantine commitments; raw SEC and Yahoo bytes remain private
to the acquisition adapter until a scored attempt has been consumed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import hmac
import math
from pathlib import Path
import re
import time
from typing import Any, Final, Protocol

from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
    ACQUISITION_VALIDATION_SCHEMA_VERSION,
    ACQUISITION_VALIDATION_VERIFIER_ID,
    CONFIRMATION as ACQUISITION_CONFIRMATION,
    DEVELOPMENT as ACQUISITION_DEVELOPMENT,
    FINAL as ACQUISITION_FINAL,
    AcquisitionExecutionResult,
    is_verified_acquisition_report,
)
from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    ATTEMPT_ID_BY_KIND,
    CONFIRMATION_SCORING,
    CONSUMED,
    DEVELOPMENT_ACQUISITION,
    DEVELOPMENT_SCORING,
    FINAL_SCORING,
    DETERMINISTIC_EVALUATION_SCHEMA_VERSION as ATTEMPT_EVALUATION_SCHEMA,
    JOINT_STAGE_REPORT_SCHEMA_VERSION as ATTEMPT_JOINT_SCHEMA,
    PLANNED,
    PREREQUISITE_ATTEMPT_BY_ID,
    TERMINAL_FAIL,
    TERMINAL_INDETERMINATE,
    TERMINAL_PASS,
    build_attempt_plan,
    build_implementation_manifest,
    issue_verified_acquisition_terminal_evidence,
    issue_verified_scored_terminal_evidence,
    store_record_receipt_material,
    validate_attempt_history,
    validate_attempt_plan,
    validate_implementation_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_BLOCKS,
    MAX_DETERMINISTIC_SECONDS,
    MAX_MARKET_REQUESTS_PER_STAGE,
    MAX_MARKET_SECONDS,
    MAX_MODEL_SECONDS,
    MAX_PUBLICATION_RECOVERY_SECONDS,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    MAX_TOTAL_RUNTIME_SECONDS,
    ACQUISITION_TERMINAL_EVIDENCE_FIELDS,
    PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256,
    PUBLICATION_NORMAL_OPERATION_SHA256,
    PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256,
    PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256,
    PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS,
    PUBLICATION_RECOVERY_INVOCATION_START_FIELDS,
    PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256,
    PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256,
    SCORED_TERMINAL_RECONSTRUCTION_FIELDS,
    SCORED_TERMINAL_EVIDENCE_FIELDS,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_metrics import (
    FROZEN_CONTROL_IDS,
    PERFORMANCE_WINDOWS,
    STAGE_CUTOFFS,
    build_stage_gate_report,
    build_stage_metrics,
    build_stage_metrics_input,
    validate_stage_gate_report,
    validate_stage_metrics,
    validate_stage_metrics_input,
)
from agent_benchmark.sec_gemma_online_risk_overlay_no_leverage import (
    validate_sec_gemma_online_risk_overlay_no_leverage_proof,
)
from agent_benchmark.sec_gemma_online_risk_overlay_replay import (
    replay_sec_gemma_online_risk_overlay_chronology,
    validate_sec_gemma_online_risk_overlay_chronology,
)
from agent_benchmark.sec_gemma_online_risk_overlay_runtime import (
    verify_installed_pinned_runtime,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    verify_live_source_tree,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    pre_push_authorization_material,
    publication_intent_material,
    remote_observation_material,
    transport_manifest_material,
    worker_ownership_material,
)
from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
    ACQUISITION_PASS,
    NORMAL_PUBLICATION,
    POST_PUSH,
    PRE_PUSH,
    PUBLICATION_RECOVERY,
    PUBLICATION_GENESIS_SHA256,
    SCORED_FAILED_GATE,
    SCORED_PASS,
    TERMINAL_FAIL as PUBLICATION_TERMINAL_FAIL,
    TERMINAL_PASS as PUBLICATION_TERMINAL_PASS,
    ExternalGitTagPublisher,
    PreparedExternalPublication,
    PreparedPublicationTransport,
    VerifiedExternalPublication,
    is_verified_external_publication,
    issue_verified_external_publication_from_observation,
    prepare_external_publication,
    prepare_isolated_publication_transport,
    reconstruct_prepared_external_publication,
    validate_external_publication,
)
from agent_benchmark.sec_gemma_online_risk_overlay_registry import (
    VerifiedFinalRegistryAuthorization,
    is_verified_final_registry_authorization,
)


RUNNER_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-runner-plan-v1"
)
PHASE_OUTPUT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-phase-output-v1"
)
PHASE_BUDGET_REPORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-phase-budget-report-v1"
)
STAGE_INPUT_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-stage-input-bundle-v1"
)
DETERMINISTIC_EVALUATION_SCHEMA_VERSION: Final[str] = (
    ATTEMPT_EVALUATION_SCHEMA
)
JOINT_STAGE_REPORT_SCHEMA_VERSION: Final[str] = (
    ATTEMPT_JOINT_SCHEMA
)
SEALED_STAGE_RESULT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-sealed-stage-result-v1"
)
VERIFY_RESULT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-verify-result-v1"
)

LOCAL_PREFLIGHT: Final[str] = "local_preflight"
DEVELOPMENT_ACQUISITION_COMMAND: Final[str] = "development_acquisition"
DEVELOPMENT_COMMAND: Final[str] = "development"
CONFIRMATION_COMMAND: Final[str] = "confirmation"
FINAL_COMMAND: Final[str] = "final"
VERIFY_COMMAND: Final[str] = "verify"
RUNNER_COMMANDS: Final[tuple[str, ...]] = (
    LOCAL_PREFLIGHT,
    DEVELOPMENT_ACQUISITION_COMMAND,
    DEVELOPMENT_COMMAND,
    CONFIRMATION_COMMAND,
    FINAL_COMMAND,
    VERIFY_COMMAND,
)

_COMMAND_TO_KIND: Final[dict[str, str]] = {
    DEVELOPMENT_ACQUISITION_COMMAND: DEVELOPMENT_ACQUISITION,
    DEVELOPMENT_COMMAND: DEVELOPMENT_SCORING,
    CONFIRMATION_COMMAND: CONFIRMATION_SCORING,
    FINAL_COMMAND: FINAL_SCORING,
}
_COMMAND_TO_METRIC_STAGE: Final[dict[str, str]] = {
    DEVELOPMENT_COMMAND: "development",
    CONFIRMATION_COMMAND: "confirmation",
    FINAL_COMMAND: "final",
}
_SCORING_COMMANDS: Final[frozenset[str]] = frozenset(
    _COMMAND_TO_METRIC_STAGE
)
_EFFECTFUL_COMMANDS: Final[frozenset[str]] = frozenset(_COMMAND_TO_KIND)
_ATTEMPT_ID_TO_COMMAND: Final[dict[str, str]] = {
    ATTEMPT_ID_BY_KIND[kind]: command
    for command, kind in _COMMAND_TO_KIND.items()
}

_PHASES: Final[dict[str, tuple[str, ...]]] = {
    LOCAL_PREFLIGHT: ("runtime_identity",),
    DEVELOPMENT_ACQUISITION_COMMAND: (
        "acquisition",
    ),
    DEVELOPMENT_COMMAND: (
        "runtime_identity",
        "gemma",
        "deterministic",
    ),
    CONFIRMATION_COMMAND: (
        "acquisition",
        "runtime_identity",
        "gemma",
        "deterministic",
    ),
    FINAL_COMMAND: (
        "acquisition",
        "runtime_identity",
        "gemma",
        "deterministic",
    ),
    VERIFY_COMMAND: (),
}
_PHASE_GROUP: Final[dict[str, str]] = {
    "acquisition": "acquisition",
    "runtime_identity": "deterministic",
    "gemma": "gemma",
    "deterministic": "deterministic",
}
_PHASE_EFFECTS: Final[dict[str, tuple[str, ...]]] = {
    "acquisition": (
        "official_sec_network",
        "market_network",
    ),
    "runtime_identity": ("ollama_runtime_identity",),
    "gemma": ("gemma_batch",),
    "deterministic": (
        "canonical_market_value_read",
        "chronological_replay",
    ),
}
_GROUP_CAPS: Final[dict[str, int]] = {
    "acquisition": MAX_SEC_SECONDS,
    "gemma": MAX_MODEL_SECONDS,
    "deterministic": MAX_DETERMINISTIC_SECONDS,
}
_GOVERNANCE_CONTINGENCY_SECONDS: Final[int] = (
    MAX_TOTAL_RUNTIME_SECONDS
    - MAX_SEC_SECONDS
    - MAX_MODEL_SECONDS
    - MAX_DETERMINISTIC_SECONDS
    - 1
)
_INTENT_PREPARATION_SECONDS: Final[int] = 89
_NORMAL_PUBLICATION_SECONDS: Final[int] = 90
_TERMINAL_FINALIZATION_SECONDS: Final[int] = 60
_PUBLICATION_QUIESCENCE_RESERVE_SECONDS: Final[float] = 15.0
_MODEL_CALL_CAPS: Final[dict[str, int]] = {
    DEVELOPMENT_COMMAND: 80,
    CONFIRMATION_COMMAND: 20,
    FINAL_COMMAND: 12,
}
_COUNTER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sec_request_count",
        "market_request_count",
        "model_call_count",
        "retry_count",
        "fallback_count",
        "model_pull_count",
        "paid_api_call_count",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_RE = re.compile(r"[a-z][a-z0-9_:-]{0,127}\Z")
_PHASE_PERMIT_SENTINEL = object()

_GOVERNANCE_CHECKS: Final[tuple[str, ...]] = (
    "source_manifest_verified",
    "exact_predecessor_verified",
    "attempt_consumed_before_effects",
    "phase_budgets_verified",
    "no_retry_or_fallback",
    "no_model_pull_or_paid_api",
    "runtime_identity_verified",
    "continuous_genesis_replay_verified",
    "all_required_arms_verified",
    "all_required_frozen_controls_verified",
    "metrics_recomputed_and_pinned",
    "gate_report_recomputed_and_pinned",
    "all_required_no_leverage_proofs_verified",
    "joint_report_committed_before_release",
)
_ACQUISITION_STAGE_BY_COMMAND: Final[dict[str, str]] = {
    DEVELOPMENT_ACQUISITION_COMMAND: ACQUISITION_DEVELOPMENT,
    CONFIRMATION_COMMAND: ACQUISITION_CONFIRMATION,
    FINAL_COMMAND: ACQUISITION_FINAL,
}
_ACQUISITION_VALIDATOR_CHECK_NAMES: Final[tuple[str, ...]] = (
    "exact_raw_bytes_replayed_sha256",
    "request_receipts_reconciled_sha256",
    "stage_and_attempt_scope_bound_sha256",
    "private_identity_digest_only_sha256",
    "market_prefix_continuity_replayed_sha256",
    "blinded_model_requests_replayed_sha256",
    "request_byte_retry_redirect_caps_reconciled_sha256",
)
PRODUCTION_EFFECT_BLOCKERS: Final[tuple[str, ...]] = (
    "implementation_must_be_committed_clean_and_pushed_to_frozen_origin",
    "exact_verified_production_authorities_must_be_supplied_before_registration",
    (
        "live_sec_yahoo_loopback_ollama_and_origin_tag_publication_are_not_"
        "exercised_by_local_preflight"
    ),
)


class SecGemmaOnlineRiskOverlayRunnerError(RuntimeError):
    """The command, phase, evidence, or sealed report failed closed."""


class SecGemmaOnlineRiskOverlayRunnerIndeterminate(
    SecGemmaOnlineRiskOverlayRunnerError
):
    """An effectful worker or external deadline ended indeterminately."""


class StageStore(Protocol):
    @property
    def store_instance_id(self) -> str: ...

    @property
    def store_session_nonce_sha256(self) -> str: ...

    def register_attempt(self, attempt_plan: Mapping[str, Any]) -> Any: ...

    def consume_attempt(self, attempt_id: str) -> Any: ...

    def authorize_effect(self, capability: Any, effect: str) -> None: ...

    def append_evidence(
        self,
        *,
        capability: Any,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> Any: ...

    def append_feature(
        self,
        *,
        capability: Any,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> Any: ...

    def append_prediction(
        self,
        *,
        capability: Any,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> Any: ...

    def append_lesson(
        self,
        *,
        capability: Any,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> Any: ...

    def append_ledger(
        self,
        *,
        capability: Any,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> Any: ...

    def append_artifact(
        self,
        *,
        capability: Any,
        effect: str,
        identity: str,
        payload: Mapping[str, Any],
    ) -> Any: ...

    def terminal_evidence_material(
        self, capability: Any
    ) -> Mapping[str, Any]: ...

    def terminal_anchor_binding(
        self, attempt_id: str
    ) -> Mapping[str, Any]: ...

    def commit_terminal_reconstruction_material(
        self, capability: Any, material: Mapping[str, Any]
    ) -> Any: ...

    def commit_publication_intent(
        self, capability: Any, intent: Mapping[str, Any]
    ) -> Any: ...

    def attempt_plan(self, attempt_id: str) -> Mapping[str, Any]: ...

    def governance_records(
        self, table: str, *, attempt_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    def terminal_reconstruction_authority(
        self, attempt_id: str
    ) -> Any: ...

    def publication_intent_authority(
        self, attempt_id: str
    ) -> Any: ...

    def publication_receipt_authority(
        self, attempt_id: str, external_publication: Any
    ) -> Any: ...

    def remote_observation_authority(
        self, attempt_id: str, remote_observation_sha256: str
    ) -> Any: ...

    def publication_worker_ownership_authority(
        self, attempt_id: str, worker_ownership_sha256: str
    ) -> Any: ...

    def pre_push_authorization_authority(
        self, attempt_id: str, pre_push_authorization_sha256: str
    ) -> Any: ...

    def recovery_completion_authority(
        self,
        attempt_id: str,
        recovery_invocation_completion_sha256: str,
    ) -> Any: ...

    def reconcile_publication_recovery_state(
        self, attempt_id: str
    ) -> None: ...

    def terminalization_claim_authority(
        self, attempt_id: str
    ) -> Any: ...

    def terminal_artifact_payload_and_receipt(
        self, attempt_id: str
    ) -> tuple[dict[str, Any], Any]: ...

    def pending_acquisition_phase_evidence(
        self, attempt_id: str
    ) -> dict[str, Any]: ...

    def issue_publication_recovery_capability(
        self,
        attempt_id: str,
        *,
        operation_kind: str,
        operation_sha256: str | None = None,
    ) -> Any: ...

    def commit_recovery_start(
        self, capability: Any, start: Mapping[str, Any]
    ) -> Any: ...

    def commit_transport_manifest(
        self, capability: Any, manifest: Mapping[str, Any]
    ) -> Any: ...

    def claim_publication_worker_ownership(
        self, capability: Any, ownership: Mapping[str, Any]
    ) -> Any: ...

    def commit_worker_quiescence(
        self, capability: Any, quiescence: Mapping[str, Any]
    ) -> Any: ...

    def commit_remote_observation(
        self,
        capability: Any,
        readback_evidence: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> Any: ...

    def commit_pre_push_authorization(
        self, capability: Any, authorization: Mapping[str, Any]
    ) -> Any: ...

    def commit_publication_conflict(
        self, capability: Any, conflict: Mapping[str, Any]
    ) -> Any: ...

    def commit_recovery_completion(
        self, capability: Any, completion: Mapping[str, Any]
    ) -> Any: ...

    def commit_publication_receipt(
        self,
        capability: Any,
        receipt: Mapping[str, Any],
        external_publication: Any,
    ) -> Any: ...

    def release_publication_capability(self, capability: Any) -> None: ...

    def issue_terminalization_capability(self, attempt_id: str) -> Any: ...

    def commit_terminalization_claim(
        self,
        capability: Any,
        terminal_evidence: Any,
        terminal_status: str,
    ) -> Any: ...

    def complete_terminalized_attempt(
        self, claim: Any, verified_terminal_evidence: Any
    ) -> Any: ...

    def predecessor_feature_rows(
        self, stage: str
    ) -> list[dict[str, Any]]: ...

    def finish_attempt(
        self,
        capability: Any,
        *,
        terminal_status: str,
        verified_terminal_evidence: Any | None = None,
    ) -> Any: ...

    def attempt_history(self, attempt_id: str) -> list[dict[str, Any]]: ...

    def snapshot(self) -> Mapping[str, Any]: ...


class PhaseExecutor(Protocol):
    """Execute one externally deadline-bound phase.

    A production implementation should run a killable child process and enforce
    ``deadline_monotonic`` itself.  The runner independently measures before and
    after the call and rejects any boundary overrun.
    """

    def execute(
        self,
        *,
        command: str,
        phase: str,
        permit: "PhasePermit | None",
        deadline_monotonic: float,
        prior_phase_outputs: Mapping[str, Mapping[str, Any]],
        acquisition_execution: AcquisitionExecutionResult | None,
    ) -> Mapping[str, Any]: ...

    def evaluate_deterministic(
        self,
        *,
        stage: str,
        input_bundle: Mapping[str, Any],
        deadline_monotonic: float,
    ) -> Mapping[str, Any]: ...


class AcquisitionAdapter(Protocol):
    """Run the exact SEC+Yahoo quarantine adapter as one indivisible phase."""

    def acquire(
        self,
        *,
        command: str,
        store: StageStore,
        capability: Any,
        deadline_monotonic: float,
    ) -> AcquisitionExecutionResult: ...


class ExternalReportPublisher(Protocol):
    def read_remote(
        self,
        *,
        prepared_publication: PreparedExternalPublication,
        transport: PreparedPublicationTransport,
        durable_transport_manifest: Any,
        transport_manifest_validator: Callable[
            [Any], Mapping[str, Any]
        ],
        worker_ownership: Any,
        worker_ownership_validator: Callable[
            [Any], Mapping[str, Any]
        ],
        store_instance_id: str,
        store_session_nonce_sha256: str,
        publication_intent_sha256: str,
        operation_kind: str,
        operation_sha256: str,
        observation_ordinal: int,
        observation_phase: str,
        prior_push_command_sha256: str,
        pre_observation_store_journal_sequence: int,
        pre_observation_store_journal_tip_sha256: str,
        timeout_seconds: float,
    ) -> Any: ...

    def push_once(
        self,
        *,
        prepared_publication: PreparedExternalPublication,
        transport: PreparedPublicationTransport,
        pre_push_authorization: Any,
        authorization_validator: Callable[
            [Any], Mapping[str, Any]
        ],
        absent_remote_observation: Any,
        observation_validator: Callable[
            [Any], Mapping[str, Any]
        ],
        timeout_seconds: float,
    ) -> Any: ...


class PublicationWorkerHandle(Protocol):
    @property
    def transport_root(self) -> Path: ...

    @property
    def source_object_directory(self) -> Path: ...

    @property
    def executable_pins(self) -> Mapping[str, Any]: ...

    @property
    def host_environment_values(self) -> Mapping[str, str]: ...

    @property
    def ownership_material(self) -> Mapping[str, Any]: ...

    def verify_executable_pins(
        self, pins: Mapping[str, Any]
    ) -> None: ...

    def abort_uncommitted(self) -> None: ...

    def quiesce(
        self, *, worker_ownership_sha256: str
    ) -> Mapping[str, Any]: ...


class PublicationWorkerRuntime(Protocol):
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
    ) -> PublicationWorkerHandle: ...


class SupervisedPublicationRecoveryWorkerAuthority(Protocol):
    """Opaque proof that this runner is executing in the supervised child.

    The production implementation binds this authority to the verified source
    tree, the identity-equal production composition, one preregistered attempt,
    the isolated worker process, and the supervisor's original deadline.  The
    returned supervisor window includes outer spawn and bootstrap time and may
    tighten, but never extend, the worker's local 300-second deadline.
    """

    def authorize_runner_recovery(
        self,
        *,
        repo_root: Path,
        implementation_manifest: Mapping[str, Any],
        production_authorities: Any,
        attempt_id: str,
        worker_entry_monotonic: float,
        worker_deadline_monotonic: float,
    ) -> tuple[float, float]: ...


class FinalRegistryAuthority(Protocol):
    """Register and externally pin the final successor before consumption."""

    def authorize_final(
        self,
        *,
        implementation_manifest: Mapping[str, Any],
        deadline_monotonic: float,
    ) -> VerifiedFinalRegistryAuthorization: ...


class DeterministicStageEvaluator(Protocol):
    def evaluate(
        self,
        *,
        stage: str,
        input_bundle: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def validate(
        self,
        evaluation: Mapping[str, Any],
        *,
        stage: str,
        input_bundle: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RunnerDependencies:
    live_implementation_manifest: Callable[[Path], Mapping[str, Any]]
    runtime_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    evaluator: DeterministicStageEvaluator
    issue_acquisition_terminal_evidence_fn: Callable[..., Any] = (
        issue_verified_acquisition_terminal_evidence
    )
    issue_scored_terminal_evidence_fn: Callable[..., Any] = (
        issue_verified_scored_terminal_evidence
    )


class PhasePermit:
    """Opaque proof that the store authorized this phase after consumption."""

    __slots__ = (
        "_attempt_id",
        "_capability",
        "_command",
        "_phase",
        "_effects",
        "_store",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        attempt_id: str,
        command: str,
        phase: str,
        effects: tuple[str, ...],
        store: StageStore,
        capability: Any,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _PHASE_PERMIT_SENTINEL:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Phase permits can only be issued by the consumed runner"
            )
        self._attempt_id = attempt_id
        self._command = command
        self._phase = phase
        self._effects = effects
        self._store = store
        self._capability = capability
        self._sentinel = _sentinel

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def command(self) -> str:
        return self._command

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def effects(self) -> tuple[str, ...]:
        return self._effects

    def release_acquisition_stage_slice(
        self,
        acquisition_execution: AcquisitionExecutionResult,
    ) -> dict[str, Any]:
        """Release the stage slice without exposing the store capability."""

        if (
            type(acquisition_execution) is not AcquisitionExecutionResult
            or self._phase != "deterministic"
            or "canonical_market_value_read" not in self._effects
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Phase permit cannot release this acquisition stage slice"
            )
        return acquisition_execution.stage_slice(
            store=self._store,
            capability=self._capability,
        )

    def release_acquisition_model_slice(
        self,
        acquisition_execution: AcquisitionExecutionResult,
    ) -> dict[str, Any]:
        """Release blinded model requests without exposing market values."""

        if (
            type(acquisition_execution) is not AcquisitionExecutionResult
            or self._phase != "gemma"
            or "gemma_batch" not in self._effects
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Phase permit cannot release this acquisition model slice"
            )
        return acquisition_execution.model_slice(
            store=self._store,
            capability=self._capability,
        )

    def __repr__(self) -> str:
        return (
            f"PhasePermit(command={self._command!r}, "
            f"phase={self._phase!r})"
        )


def is_phase_permit(value: Any) -> bool:
    """Return whether ``value`` is an exact consumed-runner phase permit."""

    return (
        type(value) is PhasePermit
        and getattr(value, "_sentinel", None) is _PHASE_PERMIT_SENTINEL
    )


def _default_live_implementation_manifest(
    repo_root: Path,
) -> Mapping[str, Any]:
    return build_implementation_manifest(
        verified_sources=verify_live_source_tree(repo_root)
    )


def _bytes_from_hex(value: Any, location: str) -> bytes:
    if type(value) is not str or len(value) % 2:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be even-length hexadecimal bytes"
        )
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be hexadecimal bytes"
        ) from exc


def _default_runtime_verifier(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    observed = _mapping(payload, "runtime phase payload")
    if set(observed) != {"version_response_hex", "show_response_hex"}:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Runtime phase payload keys changed"
        )
    return verify_installed_pinned_runtime(
        version_response_bytes=_bytes_from_hex(
            observed["version_response_hex"],
            "runtime version response",
        ),
        show_response_bytes=_bytes_from_hex(
            observed["show_response_hex"],
            "runtime show response",
        ),
    )


def default_runner_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        live_implementation_manifest=_default_live_implementation_manifest,
        runtime_verifier=_default_runtime_verifier,
        evaluator=DefaultDeterministicStageEvaluator(),
    )


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be one detached plain mapping"
        )
    return copy.deepcopy(value)


def _sequence(value: Any, location: str) -> list[Any]:
    if type(value) is not list:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be one detached plain list"
        )
    return copy.deepcopy(value)


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _identity(value: Any, location: str) -> str:
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be a canonical identity"
        )
    return value


def _self_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    location: str,
) -> str:
    observed = _mapping(value, location)
    digest = _sha256(observed.get(hash_field), f"{location}.{hash_field}")
    body = {key: child for key, child in observed.items() if key != hash_field}
    if not hmac.compare_digest(digest, canonical_sha256(body)):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} self-hash changed"
        )
    return digest


def _strict_elapsed(value: float, location: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
            f"{location} is not one nonnegative finite monotonic reading"
        )
    return float(value)


def _nonnegative_float_hex(value: Any, location: str) -> float:
    if type(value) is not str:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be one canonical float.hex string"
        )
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be one canonical float.hex string"
        ) from exc
    if (
        not math.isfinite(parsed)
        or parsed < 0.0
        or parsed.hex() != value
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} must be one nonnegative canonical float.hex value"
        )
    return parsed


def _controls_for_stage(stage: str) -> list[dict[str, str]]:
    if stage == "development":
        return [
            {
                "control_id": block_id,
                "fork_boundary_rule": first,
            }
            for block_id, first, _ in DEVELOPMENT_BLOCKS
        ]
    first = PERFORMANCE_WINDOWS[stage][0]
    return [
        {
            "control_id": FROZEN_CONTROL_IDS[stage][0],
            "fork_boundary_rule": (
                f"first_market_session_on_or_after:{first}"
            ),
        }
    ]


def _static_plan_body(
    *,
    command: str,
    implementation_manifest_sha256: str,
    attempt_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if command not in RUNNER_COMMANDS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Unknown SEC/Gemma runner command"
        )
    phases = [
        {
            "phase": phase,
            "budget_group": _PHASE_GROUP[phase],
            "effects": (
                []
                if command in {LOCAL_PREFLIGHT, VERIFY_COMMAND}
                else list(_PHASE_EFFECTS[phase])
            ),
        }
        for phase in _PHASES[command]
    ]
    metric_stage = _COMMAND_TO_METRIC_STAGE.get(command)
    body = {
        "schema_version": RUNNER_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": command,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "attempt_id": (
            None if attempt_plan is None else attempt_plan["attempt_id"]
        ),
        "attempt_kind": (
            None if attempt_plan is None else attempt_plan["attempt_kind"]
        ),
        "attempt_plan": (
            None if attempt_plan is None else copy.deepcopy(dict(attempt_plan))
        ),
        "attempt_plan_sha256": (
            None
            if attempt_plan is None
            else attempt_plan["attempt_plan_sha256"]
        ),
        "metric_stage": metric_stage,
        "phases": phases,
        "phase_group_caps_seconds": dict(_GROUP_CAPS),
        "market_seconds_at_most": MAX_MARKET_SECONDS,
        "market_requests_exactly": MAX_MARKET_REQUESTS_PER_STAGE,
        "total_seconds_strictly_below": MAX_TOTAL_RUNTIME_SECONDS,
        "retry_count": 0,
        "fallback_count": 0,
        "model_pull_count": 0,
        "paid_api_call_count": 0,
        "continuous_portfolio_genesis": (
            None if metric_stage is None else "2000-01-03"
        ),
        "required_arms": (
            []
            if metric_stage is None
            else [
                "semantic",
                "no_filing_meaning",
                "no_gemma_channel",
            ]
        ),
        "required_frozen_controls": (
            []
            if metric_stage is None
            else _controls_for_stage(metric_stage)
        ),
        "release_policy": "sealed_joint_report_only",
    }
    return body


def build_runner_plan(
    *,
    command: str,
    implementation_manifest: Mapping[str, Any],
    prerequisite_terminal_transition: Mapping[str, Any] | None = None,
    final_registry_authorization: (
        VerifiedFinalRegistryAuthorization | None
    ) = None,
) -> dict[str, Any]:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    if command in _EFFECTFUL_COMMANDS:
        if command == FINAL_COMMAND:
            if not is_verified_final_registry_authorization(
                final_registry_authorization
            ):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Final planning requires a verified pinned registry successor"
                )
        else:
            if final_registry_authorization is not None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Only final planning may use registry authorization"
                )
        attempt_plan = build_attempt_plan(
            implementation_manifest=implementation,
            attempt_id=ATTEMPT_ID_BY_KIND[_COMMAND_TO_KIND[command]],
            prerequisite_terminal_transition=(
                prerequisite_terminal_transition
            ),
            final_registry_authorization=final_registry_authorization,
        )
    else:
        if (
            prerequisite_terminal_transition is not None
            or final_registry_authorization is not None
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Read-only commands cannot claim attempt prerequisites or pins"
            )
        attempt_plan = None
    body = _static_plan_body(
        command=command,
        implementation_manifest_sha256=implementation[
            "implementation_manifest_sha256"
        ],
        attempt_plan=attempt_plan,
    )
    return {**body, "runner_plan_sha256": canonical_sha256(body)}


def validate_runner_plan(
    value: Mapping[str, Any],
    *,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _mapping(value, "runner plan")
    expected_keys = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "command",
        "implementation_manifest_sha256",
        "attempt_id",
        "attempt_kind",
        "attempt_plan",
        "attempt_plan_sha256",
        "metric_stage",
        "phases",
        "phase_group_caps_seconds",
        "market_seconds_at_most",
        "market_requests_exactly",
        "total_seconds_strictly_below",
        "retry_count",
        "fallback_count",
        "model_pull_count",
        "paid_api_call_count",
        "continuous_portfolio_genesis",
        "required_arms",
        "required_frozen_controls",
        "release_policy",
        "runner_plan_sha256",
    }
    if set(observed) != expected_keys:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Runner plan keys changed"
        )
    _self_hash(
        observed,
        hash_field="runner_plan_sha256",
        location="runner plan",
    )
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    command = observed["command"]
    if command not in RUNNER_COMMANDS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Runner plan command changed"
        )
    raw_attempt_plan = observed["attempt_plan"]
    if command in _EFFECTFUL_COMMANDS:
        if type(raw_attempt_plan) is not dict:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Effectful runner plan lacks its attempt plan"
            )
        attempt_plan = validate_attempt_plan(
            raw_attempt_plan,
            implementation_manifest=implementation,
        )
        expected_attempt_id = ATTEMPT_ID_BY_KIND[
            _COMMAND_TO_KIND[command]
        ]
        if attempt_plan["attempt_id"] != expected_attempt_id:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner plan crossed its fixed attempt"
            )
    else:
        if (
            raw_attempt_plan is not None
            or observed["attempt_id"] is not None
            or observed["attempt_kind"] is not None
            or observed["attempt_plan_sha256"] is not None
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Read-only runner plan acquired an attempt"
            )
        attempt_plan = None
    expected_body = _static_plan_body(
        command=command,
        implementation_manifest_sha256=implementation[
            "implementation_manifest_sha256"
        ],
        attempt_plan=attempt_plan,
    )
    expected = {
        **expected_body,
        "runner_plan_sha256": canonical_sha256(expected_body),
    }
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Runner plan differs from its canonical reconstruction"
        )
    return expected


def build_phase_output(
    *,
    command: str,
    phase: str,
    counters: Mapping[str, int],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    fixed_counters = _mapping(counters, "phase counters")
    if set(fixed_counters) != _COUNTER_KEYS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Phase counter keys changed"
        )
    for key, count in fixed_counters.items():
        if type(count) is not int or count < 0:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                f"Phase counter {key} must be a nonnegative integer"
            )
    fixed_payload = _mapping(payload, "phase payload")
    body = {
        "schema_version": PHASE_OUTPUT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": command,
        "phase": phase,
        "counters": fixed_counters,
        "payload": fixed_payload,
        "payload_sha256": canonical_sha256(fixed_payload),
    }
    return {**body, "phase_output_sha256": canonical_sha256(body)}


def _validate_phase_counters(
    command: str,
    phase: str,
    counters: Mapping[str, Any],
) -> dict[str, int]:
    observed = _mapping(counters, "phase counters")
    if set(observed) != _COUNTER_KEYS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Phase counter keys changed"
        )
    fixed: dict[str, int] = {}
    for key in sorted(_COUNTER_KEYS):
        count = observed[key]
        if type(count) is not int or count < 0:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                f"Phase counter {key} is invalid"
            )
        fixed[key] = count
    for forbidden in (
        "retry_count",
        "fallback_count",
        "model_pull_count",
        "paid_api_call_count",
    ):
        if fixed[forbidden] != 0:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Retries, fallbacks, pulls, and paid APIs are forbidden"
            )
    expected_nonzero: set[str]
    if phase == "acquisition":
        expected_nonzero = {
            "sec_request_count",
            "market_request_count",
        }
        if not 1 <= fixed["sec_request_count"] <= MAX_SEC_REQUESTS:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "SEC request count exceeds the fixed one-shot cap"
            )
        if fixed["market_request_count"] != MAX_MARKET_REQUESTS_PER_STAGE:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Yahoo acquisition must make exactly six requests"
            )
    elif phase == "gemma":
        expected_nonzero = {"model_call_count"}
        cap = _MODEL_CALL_CAPS[command]
        if not 1 <= fixed["model_call_count"] <= cap:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Gemma call count is outside the fixed stage cap"
            )
        if (
            command == DEVELOPMENT_COMMAND
            and fixed["model_call_count"] < 5
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Development requires the five sealed latency calls"
            )
    else:
        expected_nonzero = set()
    for key in (
        "sec_request_count",
        "market_request_count",
        "model_call_count",
    ):
        if key not in expected_nonzero and fixed[key] != 0:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                f"{phase} reports a forbidden {key}"
            )
    return fixed


def validate_phase_output(
    value: Mapping[str, Any],
    *,
    command: str,
    phase: str,
) -> dict[str, Any]:
    observed = _mapping(value, f"{phase} phase output")
    expected_keys = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "command",
        "phase",
        "counters",
        "payload",
        "payload_sha256",
        "phase_output_sha256",
    }
    if set(observed) != expected_keys:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{phase} phase output keys changed"
        )
    if (
        observed["schema_version"] != PHASE_OUTPUT_SCHEMA_VERSION
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["command"] != command
        or observed["phase"] != phase
        or phase not in _PHASES[command]
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{phase} phase identity changed"
        )
    counters = _validate_phase_counters(
        command, phase, observed["counters"]
    )
    payload = _mapping(observed["payload"], f"{phase} phase payload")
    if observed["payload_sha256"] != canonical_sha256(payload):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{phase} phase payload hash changed"
        )
    _self_hash(
        observed,
        hash_field="phase_output_sha256",
        location=f"{phase} phase output",
    )
    expected = build_phase_output(
        command=command,
        phase=phase,
        counters=counters,
        payload=payload,
    )
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{phase} phase output is noncanonical"
        )
    return expected


def _validate_acquisition_report_payload(
    value: Mapping[str, Any],
    *,
    command: str,
) -> dict[str, Any]:
    observed = _mapping(value, "verified acquisition report")
    expected_keys = {
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
    }
    if set(observed) != expected_keys:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Verified acquisition report schema changed"
        )
    expected_stage = _ACQUISITION_STAGE_BY_COMMAND[command]
    checks = _mapping(
        observed["checks"], "verified acquisition checks"
    )
    if (
        observed["schema_version"]
        != ACQUISITION_VALIDATION_SCHEMA_VERSION
        or observed["verifier_id"]
        != ACQUISITION_VALIDATION_VERIFIER_ID
        or observed["verdict"] != "pass"
        or observed["stage"] != expected_stage
        or observed["attempt_id"]
        != ATTEMPT_ID_BY_KIND[_COMMAND_TO_KIND[command]]
        or observed["attempt_kind"] != _COMMAND_TO_KIND[command]
        or set(checks) != set(_ACQUISITION_VALIDATOR_CHECK_NAMES)
        or any(
            type(name) is not str
            or type(digest) is not str
            or _SHA256_RE.fullmatch(digest) is None
            for name, digest in checks.items()
        )
        or observed["check_set_sha256"] != canonical_sha256(checks)
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Verified acquisition report is not exact stage-bound evidence"
        )
    for field in (
        "acquisition_plan_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "validation_sha256",
    ):
        _sha256(observed[field], f"verified acquisition {field}")
    _self_hash(
        observed,
        hash_field="validation_sha256",
        location="verified acquisition report",
    )
    return observed


def _normalize_acquisition_execution(
    value: Any,
    *,
    command: str,
    test_only: bool,
) -> tuple[dict[str, Any], Any | None, Any]:
    private_execution: Any | None = None
    if type(value) is AcquisitionExecutionResult:
        if not is_verified_acquisition_report(value.verified_report):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Acquisition result lacks its opaque validator report"
            )
        report = value.terminal_sealing_material()
        summary = value.public_summary()
        accounting = value.request_accounting()
        report_authority: Any = value.verified_report
        private_execution = value
    elif test_only and type(value) is dict and set(value) == {
        "verified_report",
        "public_summary",
        "request_accounting",
    }:
        report = _mapping(
            value["verified_report"], "test acquisition report"
        )
        summary = _mapping(
            value["public_summary"], "test acquisition public summary"
        )
        accounting = _mapping(
            value["request_accounting"],
            "test acquisition request accounting",
        )
        report_authority = report
    else:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Production acquisition requires opaque AcquisitionExecutionResult"
        )
    report = _validate_acquisition_report_payload(
        report, command=command
    )
    required_summary = {
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
    if set(summary) != required_summary:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Runner acquisition public summary changed"
        )
    _self_hash(
        summary,
        hash_field="public_summary_sha256",
        location="acquisition public summary",
    )
    required_accounting = {
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
    if set(accounting) != required_accounting:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Runner acquisition request accounting changed"
        )
    _self_hash(
        accounting,
        hash_field="accounting_sha256",
        location="acquisition request accounting",
    )
    if (
        summary["stage"] != report["stage"]
        or summary["attempt_id"] != report["attempt_id"]
        or summary["attempt_kind"] != report["attempt_kind"]
        or summary["acquisition_plan_sha256"]
        != report["acquisition_plan_sha256"]
        or summary["bundle_sha256"] != report["bundle_sha256"]
        or summary["manifest_sha256"] != report["manifest_sha256"]
        or summary["private_index_sha256"]
        != report["private_index_sha256"]
        or accounting["stage"] != report["stage"]
        or accounting["attempt_id"] != report["attempt_id"]
        or type(accounting["sec_request_count"]) is not int
        or not 1 <= accounting["sec_request_count"] <= MAX_SEC_REQUESTS
        or accounting["market_request_count"]
        != MAX_MARKET_REQUESTS_PER_STAGE
        or accounting["network_request_count"]
        != accounting["sec_request_count"]
        + accounting["market_request_count"]
        or accounting["retry_count"] != 0
        or accounting["redirect_count"] != 0
        or type(accounting["sec_bytes"]) is not int
        or type(accounting["market_bytes"]) is not int
        or accounting["sec_bytes"] < 0
        or accounting["market_bytes"] < 0
        or accounting["sec_bytes"] > MAX_SEC_BYTES
        or summary["market_response_count"]
        != accounting["market_request_count"]
        or summary["market_elapsed_seconds_hex"]
        != accounting["market_elapsed_seconds_hex"]
        or type(summary["model_request_count"]) is not int
        or summary["model_request_count"] < 1
        or summary["complete_batch"] is not True
        or summary["quarantine_only"] is not True
        or (
            not test_only
            and summary["production_authority"] is not True
        )
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Acquisition summary or accounting is incomplete"
        )
    _nonnegative_float_hex(
        accounting["market_elapsed_seconds_hex"],
        "acquisition market elapsed time",
    )
    _sha256(
        summary["model_slice_sha256"],
        "acquisition model slice",
    )
    _sha256(
        summary["stage_slice_sha256"],
        "acquisition stage slice",
    )
    payload = {
        "verified_acquisition_report": report,
        "public_summary": summary,
        "request_accounting": accounting,
    }
    return payload, private_execution, report_authority


def _runtime_payload(
    outputs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    try:
        return outputs["runtime_identity"]["payload"]
    except (KeyError, TypeError) as exc:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Scored execution lacks runtime identity evidence"
        ) from exc


def _latency_preflight_receipt(
    semantic_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _mapping(
        semantic_payload,
        "Gemma semantic payload",
    )
    rows = _sequence(
        payload.get("semantic_extraction_rows"),
        "Gemma semantic rows",
    )
    if not rows:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Gemma semantic rows lack latency preflight evidence"
        )
    canonical_receipt: dict[str, Any] | None = None
    canonical_digest: str | None = None
    for ordinal, raw_row in enumerate(rows, start=1):
        row = _mapping(raw_row, f"Gemma semantic row {ordinal}")
        receipt = _mapping(
            row.get("latency_preflight_receipt"),
            f"Gemma semantic row {ordinal} latency receipt",
        )
        digest = _sha256(
            row.get("latency_preflight_receipt_sha256"),
            f"Gemma semantic row {ordinal} latency receipt hash",
        )
        if (
            receipt.get("latency_preflight_receipt_sha256")
            != digest
            or canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "latency_preflight_receipt_sha256"
                }
            )
            != digest
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Gemma latency preflight receipt lost its self-hash"
            )
        if canonical_receipt is None:
            canonical_receipt = receipt
            canonical_digest = digest
        elif receipt != canonical_receipt or digest != canonical_digest:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Gemma rows disagree on latency preflight evidence"
            )
    assert canonical_receipt is not None
    return canonical_receipt


def _feature_order_key(
    row: Mapping[str, Any],
    *,
    location: str,
) -> tuple[str, str, str, str]:
    fixed = _mapping(row, location)
    decision = fixed.get("decision_session")
    acceptance = fixed.get("acceptance_datetime")
    accession = fixed.get("accession_number")
    digest = _sha256(
        fixed.get("feature_row_sha256"),
        f"{location} hash",
    )
    if (
        type(decision) is not str
        or not decision
        or (
            acceptance is not None
            and (type(acceptance) is not str or not acceptance)
        )
        or type(accession) is not str
        or not accession
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{location} chronology fields changed"
        )
    return decision, acceptance or "", accession, digest


def _cumulative_feature_rows(
    *,
    stage: str,
    predecessor_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if stage not in STAGE_CUTOFFS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Cumulative feature stage is not preregistered"
        )
    prior = [
        _mapping(row, f"{stage} predecessor feature row {ordinal}")
        for ordinal, row in enumerate(predecessor_rows, start=1)
    ]
    current = [
        _mapping(row, f"{stage} current feature row {ordinal}")
        for ordinal, row in enumerate(current_rows, start=1)
    ]
    first_current = {
        "development": "2000-01-01",
        "confirmation": "2019-01-01",
        "final": "2024-01-01",
    }[stage]
    if stage == "development" and prior:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Development replay cannot inherit predecessor features"
        )
    for ordinal, row in enumerate(prior, start=1):
        if _feature_order_key(
            row,
            location=f"{stage} predecessor feature row {ordinal}",
        )[0] >= first_current:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Predecessor feature row crossed the stage boundary"
            )
    for ordinal, row in enumerate(current, start=1):
        decision = _feature_order_key(
            row,
            location=f"{stage} current feature row {ordinal}",
        )[0]
        if not first_current <= decision <= STAGE_CUTOFFS[stage]:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Current feature row crossed the stage boundary"
            )
    combined = prior + current
    combined.sort(
        key=lambda row: _feature_order_key(
            row,
            location=f"{stage} cumulative feature row",
        )
    )
    hashes = [row["feature_row_sha256"] for row in combined]
    accessions = [row["accession_number"] for row in combined]
    if (
        len(hashes) != len(set(hashes))
        or len(accessions) != len(set(accessions))
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Cumulative feature rows are duplicated across stages"
        )
    predecessor_hashes = [
        row["feature_row_sha256"] for row in prior
    ]
    current_hashes = [
        row["feature_row_sha256"] for row in current
    ]
    return combined, predecessor_hashes, current_hashes


def _stage_input_bundle(
    *,
    stage: str,
    outputs: Mapping[str, Mapping[str, Any]],
    predecessor_feature_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    try:
        gemma_payload = _mapping(
            outputs["gemma"]["payload"], "Gemma phase payload"
        )
        deterministic_payload = _mapping(
            outputs["deterministic"]["payload"],
            "deterministic phase payload",
        )
    except KeyError as exc:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Scored execution lacks Gemma or deterministic input evidence"
        ) from exc
    if set(gemma_payload) != {
        "semantic_extraction_rows",
        "semantic_extraction_row_sha256s",
        "semantic_batch_receipt_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Gemma phase payload keys changed"
        )
    if set(deterministic_payload) != {
        "market_rows",
        "market_rows_sha256",
        "baseline_signals",
        "baseline_signals_sha256",
        "feature_rows",
        "feature_row_sha256s",
        "semantic_batch_receipt_sha256",
        "source_commitments",
    }:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Deterministic phase payload keys changed"
        )
    current_feature_rows = _sequence(
        deterministic_payload["feature_rows"],
        "deterministic feature rows",
    )
    current_feature_hashes = _sequence(
        deterministic_payload["feature_row_sha256s"],
        "deterministic feature row hashes",
    )
    if (
        len(current_feature_rows) != len(current_feature_hashes)
        or [
            row.get("feature_row_sha256")
            if type(row) is dict
            else None
            for row in current_feature_rows
        ]
        != current_feature_hashes
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Gemma feature rows lost their exact hash binding"
        )
    for index, digest in enumerate(current_feature_hashes):
        _sha256(digest, f"feature row hash {index}")
    (
        feature_rows,
        predecessor_feature_hashes,
        rebuilt_current_feature_hashes,
    ) = _cumulative_feature_rows(
        stage=stage,
        predecessor_rows=predecessor_feature_rows,
        current_rows=current_feature_rows,
    )
    if rebuilt_current_feature_hashes != current_feature_hashes:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Current feature row order differs from its deterministic pins"
        )
    feature_hashes = [
        row["feature_row_sha256"] for row in feature_rows
    ]
    _sha256(
        deterministic_payload["semantic_batch_receipt_sha256"],
        "deterministic semantic batch receipt",
    )
    semantic_rows = _sequence(
        gemma_payload["semantic_extraction_rows"],
        "Gemma semantic extraction rows",
    )
    semantic_hashes = _sequence(
        gemma_payload["semantic_extraction_row_sha256s"],
        "Gemma semantic extraction row hashes",
    )
    if (
        len(semantic_rows) != len(semantic_hashes)
        or [
            row.get("semantic_extraction_row_sha256")
            if type(row) is dict
            else None
            for row in semantic_rows
        ]
        != semantic_hashes
        or deterministic_payload["semantic_batch_receipt_sha256"]
        != gemma_payload["semantic_batch_receipt_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Gemma semantic rows lost their deterministic binding"
        )
    for index, digest in enumerate(semantic_hashes):
        _sha256(digest, f"semantic extraction row hash {index}")
    market_rows = _sequence(
        deterministic_payload["market_rows"], "stage market rows"
    )
    baseline_signals = _sequence(
        deterministic_payload["baseline_signals"],
        "stage baseline signals",
    )
    market_hash = _sha256(
        deterministic_payload["market_rows_sha256"],
        "stage market row batch hash",
    )
    baseline_hash = _sha256(
        deterministic_payload["baseline_signals_sha256"],
        "stage baseline signal batch hash",
    )
    if (
        canonical_sha256(market_rows) != market_hash
        or canonical_sha256(baseline_signals) != baseline_hash
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Stage market or baseline batch hash changed"
        )
    if not market_rows or type(market_rows[0]) is not dict:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Stage market rows are empty"
        )
    if market_rows[0].get("session") != "2000-01-03":
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Every scored replay must start at the 2000 portfolio genesis"
        )
    if (
        type(market_rows[-1]) is not dict
        or market_rows[-1].get("session") != STAGE_CUTOFFS[stage]
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Stage market prefix does not end at its fixed cutoff"
        )
    current_source_commitments = _mapping(
        deterministic_payload["source_commitments"],
        "stage source commitments",
    )
    source_commitments = {
        "current_stage_source_commitments": (
            current_source_commitments
        ),
        "current_stage_source_commitments_sha256": canonical_sha256(
            current_source_commitments
        ),
        "predecessor_feature_row_sha256s": (
            predecessor_feature_hashes
        ),
        "predecessor_feature_rows_sha256": canonical_sha256(
            list(predecessor_feature_rows)
        ),
    }
    body = {
        "schema_version": STAGE_INPUT_BUNDLE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "market_rows": market_rows,
        "market_rows_sha256": market_hash,
        "baseline_signals": baseline_signals,
        "baseline_signals_sha256": baseline_hash,
        "feature_rows": feature_rows,
        "feature_row_sha256s": feature_hashes,
        "feature_rows_sha256": canonical_sha256(feature_rows),
        "current_feature_row_sha256s": current_feature_hashes,
        "current_feature_rows_sha256": canonical_sha256(
            current_feature_rows
        ),
        "predecessor_feature_row_sha256s": (
            predecessor_feature_hashes
        ),
        "predecessor_feature_rows_sha256": canonical_sha256(
            list(predecessor_feature_rows)
        ),
        "semantic_batch_receipt_sha256": (
            deterministic_payload["semantic_batch_receipt_sha256"]
        ),
        "source_commitments": source_commitments,
        "source_commitments_sha256": canonical_sha256(
            source_commitments
        ),
    }
    return {**body, "stage_input_bundle_sha256": canonical_sha256(body)}


def _family(branch: Mapping[str, Any], account: str) -> dict[str, Any]:
    return {
        cost: copy.deepcopy(branch["ledgers"][cost][account])
        for cost in ("cost_5bps", "cost_10bps")
    }


def _first_market_session_on_or_after(
    market_rows: Sequence[Mapping[str, Any]],
    first: str,
) -> str:
    for row in market_rows:
        session = row.get("session")
        if type(session) is str and session >= first:
            return session
    raise SecGemmaOnlineRiskOverlayRunnerError(
        "Frozen-control boundary is outside the market prefix"
    )


def _proof_inputs(
    *,
    evaluation_replays: Mapping[str, Any],
    market_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    semantic = evaluation_replays["semantic"]["online"]
    no_meaning = evaluation_replays["no_filing_meaning"]["online"]
    no_gemma = evaluation_replays["no_gemma_channel"]["online"]
    inputs: dict[str, dict[str, Any]] = {}

    def add(
        proof_id: str,
        *,
        ledger: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> None:
        inputs[proof_id] = {
            "ledger": ledger,
            "market_rows": market_rows,
            "target_rows": targets,
            "policy_id": ledger["policy_id"],
            "cost_bps": ledger["cost_bps"],
        }

    for cost in ("cost_5bps", "cost_10bps"):
        add(
            f"semantic:{cost}",
            ledger=semantic["primary"]["ledgers"][cost]["combined"],
            targets=semantic["primary"]["target_stream"]["target_rows"],
        )
        add(
            f"baseline:{cost}",
            ledger=semantic["primary"]["ledgers"][cost]["baseline"],
            targets=semantic["baseline_target_stream"]["target_rows"],
        )
        add(
            f"aapl_buy_and_hold:{cost}",
            ledger=semantic["primary"]["ledgers"][cost][
                "aapl_buy_and_hold"
            ],
            targets=semantic["aapl_target_rows"],
        )
        add(
            f"no_filing_meaning:{cost}",
            ledger=no_meaning["primary"]["ledgers"][cost]["combined"],
            targets=no_meaning["primary"]["target_stream"]["target_rows"],
        )
        add(
            f"no_gemma_channel:{cost}",
            ledger=no_gemma["primary"]["ledgers"][cost]["combined"],
            targets=no_gemma["primary"]["target_stream"]["target_rows"],
        )
        for control_id, replay in evaluation_replays["semantic"][
            "frozen_controls"
        ].items():
            frozen = replay["frozen_control"]
            if frozen is None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Required frozen replay lacks its frozen branch"
                )
            add(
                f"frozen:{control_id}:{cost}",
                ledger=frozen["ledgers"][cost]["combined"],
                targets=frozen["target_stream"]["target_rows"],
            )
    return inputs


class DefaultDeterministicStageEvaluator:
    """Build and independently validate every replay, metric, gate, and proof."""

    def evaluate(
        self,
        *,
        stage: str,
        input_bundle: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        bundle = _mapping(input_bundle, "stage input bundle")
        _self_hash(
            bundle,
            hash_field="stage_input_bundle_sha256",
            location="stage input bundle",
        )
        if bundle.get("stage") != stage:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Stage input bundle crossed its metric stage"
            )
        market = bundle["market_rows"]
        baseline = bundle["baseline_signals"]
        features = bundle["feature_rows"]
        feature_hashes = bundle["feature_row_sha256s"]

        controls: dict[str, dict[str, Any]] = {}
        semantic_primary: dict[str, Any] | None = None
        for control in _controls_for_stage(stage):
            control_id = control["control_id"]
            rule = control["fork_boundary_rule"]
            boundary = (
                rule
                if not rule.startswith("first_market_session_on_or_after:")
                else _first_market_session_on_or_after(
                    market, rule.split(":", 1)[1]
                )
            )
            replay = replay_sec_gemma_online_risk_overlay_chronology(
                market_rows=market,
                expected_market_rows_sha256=bundle[
                    "market_rows_sha256"
                ],
                baseline_signals=baseline,
                expected_baseline_signals_sha256=bundle[
                    "baseline_signals_sha256"
                ],
                feature_rows=features,
                expected_feature_row_sha256s=feature_hashes,
                arm="semantic",
                replay_id=f"{stage}:semantic",
                frozen_before_boundary=boundary,
            )
            validate_sec_gemma_online_risk_overlay_chronology(
                replay,
                expected_chronological_replay_sha256=replay[
                    "chronological_replay_sha256"
                ],
                market_rows=market,
                expected_market_rows_sha256=bundle[
                    "market_rows_sha256"
                ],
                baseline_signals=baseline,
                expected_baseline_signals_sha256=bundle[
                    "baseline_signals_sha256"
                ],
                feature_rows=features,
                expected_feature_row_sha256s=feature_hashes,
                arm="semantic",
                replay_id=f"{stage}:semantic",
                frozen_before_boundary=boundary,
            )
            if semantic_primary is None:
                semantic_primary = replay
            elif (
                replay["primary"] != semantic_primary["primary"]
                or replay["learner_lessons"]
                != semantic_primary["learner_lessons"]
            ):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Semantic frozen controls changed the continuous primary"
                )
            controls[control_id] = replay
        assert semantic_primary is not None
        replays: dict[str, Any] = {
            "semantic": {
                "online": semantic_primary,
                "frozen_controls": controls,
            }
        }
        for arm in ("no_filing_meaning", "no_gemma_channel"):
            replay = replay_sec_gemma_online_risk_overlay_chronology(
                market_rows=market,
                expected_market_rows_sha256=bundle[
                    "market_rows_sha256"
                ],
                baseline_signals=baseline,
                expected_baseline_signals_sha256=bundle[
                    "baseline_signals_sha256"
                ],
                feature_rows=features,
                expected_feature_row_sha256s=feature_hashes,
                arm=arm,
                replay_id=f"{stage}:{arm}",
                frozen_before_boundary=None,
            )
            validate_sec_gemma_online_risk_overlay_chronology(
                replay,
                expected_chronological_replay_sha256=replay[
                    "chronological_replay_sha256"
                ],
                market_rows=market,
                expected_market_rows_sha256=bundle[
                    "market_rows_sha256"
                ],
                baseline_signals=baseline,
                expected_baseline_signals_sha256=bundle[
                    "baseline_signals_sha256"
                ],
                feature_rows=features,
                expected_feature_row_sha256s=feature_hashes,
                arm=arm,
                replay_id=f"{stage}:{arm}",
                frozen_before_boundary=None,
            )
            replays[arm] = {
                "online": replay,
                "frozen_controls": {},
            }

        semantic = semantic_primary
        no_meaning = replays["no_filing_meaning"]["online"]
        no_gemma = replays["no_gemma_channel"]["online"]
        frozen_ledgers: dict[str, Any] = {}
        frozen_predictions: dict[str, Any] = {}
        frozen_actions: dict[str, Any] = {}
        for control_id, replay in controls.items():
            frozen = replay["frozen_control"]
            if frozen is None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Required frozen control is missing"
                )
            frozen_ledgers[control_id] = _family(frozen, "combined")
            frozen_predictions[control_id] = frozen["predictions"]
            frozen_actions[control_id] = frozen["policy_replay"][
                "policy_actions"
            ]
        metrics_input = build_stage_metrics_input(
            stage=stage,
            ledgers={
                "semantic": _family(semantic["primary"], "combined"),
                "baseline": _family(semantic["primary"], "baseline"),
                "aapl_buy_and_hold": _family(
                    semantic["primary"], "aapl_buy_and_hold"
                ),
                "no_filing_meaning": _family(
                    no_meaning["primary"], "combined"
                ),
                "no_gemma_channel": _family(
                    no_gemma["primary"], "combined"
                ),
            },
            frozen_control_ledgers=frozen_ledgers,
            feature_rows=features,
            semantic_predictions=semantic["primary"]["predictions"],
            no_filing_meaning_predictions=no_meaning["primary"][
                "predictions"
            ],
            no_gemma_channel_predictions=no_gemma["primary"][
                "predictions"
            ],
            frozen_predictions=frozen_predictions,
            learner_lessons=semantic["learner_lessons"],
            semantic_policy_actions=semantic["primary"]["policy_replay"][
                "policy_actions"
            ],
            no_filing_meaning_policy_actions=no_meaning["primary"][
                "policy_replay"
            ]["policy_actions"],
            no_gemma_channel_policy_actions=no_gemma["primary"][
                "policy_replay"
            ]["policy_actions"],
            frozen_policy_actions=frozen_actions,
            semantic_overlay_episodes=semantic["primary"]["target_stream"][
                "overlay_episodes"
            ],
        )
        validate_stage_metrics_input(
            metrics_input,
            expected_stage_metrics_input_sha256=metrics_input[
                "stage_metrics_input_sha256"
            ],
        )
        metrics = build_stage_metrics(
            metrics_input,
            expected_stage_metrics_input_sha256=metrics_input[
                "stage_metrics_input_sha256"
            ],
        )
        validate_stage_metrics(
            metrics,
            expected_stage_metrics_sha256=metrics[
                "stage_metrics_sha256"
            ],
            metrics_input=metrics_input,
            expected_stage_metrics_input_sha256=metrics_input[
                "stage_metrics_input_sha256"
            ],
        )
        gate = build_stage_gate_report(
            metrics,
            expected_stage_metrics_sha256=metrics[
                "stage_metrics_sha256"
            ],
        )
        validate_stage_gate_report(
            gate,
            expected_gate_report_sha256=gate["gate_report_sha256"],
            metrics=metrics,
            expected_stage_metrics_sha256=metrics[
                "stage_metrics_sha256"
            ],
        )

        proof_inputs = _proof_inputs(
            evaluation_replays=replays,
            market_rows=market,
        )
        proofs: dict[str, Any] = {}
        for proof_id, proof_input in sorted(proof_inputs.items()):
            ledger = proof_input["ledger"]
            proofs[proof_id] = (
                validate_sec_gemma_online_risk_overlay_no_leverage_proof(
                    ledger,
                    expected_ledger_sha256=ledger["ledger_sha256"],
                    market_rows=proof_input["market_rows"],
                    target_rows=proof_input["target_rows"],
                    policy_id=proof_input["policy_id"],
                    cost_bps=proof_input["cost_bps"],
                )
            )
        body = {
            "schema_version": DETERMINISTIC_EVALUATION_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "stage": stage,
            "stage_input_bundle_sha256": bundle[
                "stage_input_bundle_sha256"
            ],
            "replays": replays,
            "replays_sha256": canonical_sha256(replays),
            "metrics_input": metrics_input,
            "metrics_input_sha256": metrics_input[
                "stage_metrics_input_sha256"
            ],
            "stage_metrics": metrics,
            "stage_metrics_sha256": metrics["stage_metrics_sha256"],
            "gate_report": gate,
            "gate_report_sha256": gate["gate_report_sha256"],
            "no_leverage_proofs": proofs,
            "no_leverage_proofs_sha256": canonical_sha256(proofs),
        }
        return {
            **body,
            "deterministic_evaluation_sha256": canonical_sha256(body),
        }

    def validate(
        self,
        evaluation: Mapping[str, Any],
        *,
        stage: str,
        input_bundle: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        observed = _validate_evaluation_envelope(
            evaluation,
            stage=stage,
            input_bundle=input_bundle,
        )
        rebuilt = self.evaluate(stage=stage, input_bundle=input_bundle)
        if observed != rebuilt:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Deterministic evaluation differs from exact reconstruction"
            )
        return observed


def _required_proof_ids(stage: str) -> set[str]:
    ids = {
        f"{account}:{cost}"
        for account in (
            "semantic",
            "baseline",
            "aapl_buy_and_hold",
            "no_filing_meaning",
            "no_gemma_channel",
        )
        for cost in ("cost_5bps", "cost_10bps")
    }
    ids.update(
        {
            f"frozen:{control_id}:{cost}"
            for control_id in FROZEN_CONTROL_IDS[stage]
            for cost in ("cost_5bps", "cost_10bps")
        }
    )
    return ids


def _validate_gate_binding(
    gate: Mapping[str, Any],
    *,
    stage: str,
    require_pass: bool,
) -> dict[str, Any]:
    observed = _mapping(gate, "stage gate report")
    required = build_contract_manifest()["gates"][stage]
    checks = _mapping(observed.get("checks"), "stage gate checks")
    if set(checks) != set(required):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Stage report checks are not the exact preregistered gate keys"
        )
    if any(type(value) is not bool for value in checks.values()):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Stage gate checks must be exact Booleans"
        )
    failures = [name for name, passed in checks.items() if not passed]
    if (
        observed.get("passed") is not (not failures)
        or observed.get("failed_checks") != failures
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Scored stage gate verdict differs from its exact checks"
        )
    if require_pass and failures:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "A scored stage cannot terminal-pass with a failed gate"
        )
    gate_hash = _sha256(
        observed.get("gate_report_sha256"), "gate report hash"
    )
    _self_hash(
        observed,
        hash_field="gate_report_sha256",
        location="stage gate report",
    )
    if gate_hash != observed["gate_report_sha256"]:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Gate report hash changed"
        )
    return observed


def _validate_evaluation_envelope(
    evaluation: Mapping[str, Any],
    *,
    stage: str,
    input_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _mapping(evaluation, "deterministic evaluation")
    expected_keys = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "stage",
        "stage_input_bundle_sha256",
        "replays",
        "replays_sha256",
        "metrics_input",
        "metrics_input_sha256",
        "stage_metrics",
        "stage_metrics_sha256",
        "gate_report",
        "gate_report_sha256",
        "no_leverage_proofs",
        "no_leverage_proofs_sha256",
        "deterministic_evaluation_sha256",
    }
    if set(observed) != expected_keys:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Deterministic evaluation keys changed"
        )
    if (
        observed["schema_version"] != DETERMINISTIC_EVALUATION_SCHEMA_VERSION
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["stage"] != stage
        or observed["stage_input_bundle_sha256"]
        != input_bundle["stage_input_bundle_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Deterministic evaluation identity changed"
        )
    _self_hash(
        observed,
        hash_field="deterministic_evaluation_sha256",
        location="deterministic evaluation",
    )
    replays = _mapping(observed["replays"], "evaluation replays")
    if set(replays) != {
        "semantic",
        "no_filing_meaning",
        "no_gemma_channel",
    }:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Evaluation lacks an exact required arm"
        )
    for arm in replays:
        arm_value = _mapping(replays[arm], f"{arm} replay family")
        if set(arm_value) != {"online", "frozen_controls"}:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                f"{arm} replay family keys changed"
            )
        online = _mapping(arm_value["online"], f"{arm} online replay")
        _sha256(
            online.get("chronological_replay_sha256"),
            f"{arm} chronological replay hash",
        )
        controls = _mapping(
            arm_value["frozen_controls"], f"{arm} frozen controls"
        )
        expected_controls = (
            set(FROZEN_CONTROL_IDS[stage])
            if arm == "semantic"
            else set()
        )
        if set(controls) != expected_controls:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                f"{arm} frozen-control inventory changed"
            )
        for control_id, replay in controls.items():
            replay_value = _mapping(
                replay, f"{arm} frozen replay {control_id}"
            )
            _sha256(
                replay_value.get("chronological_replay_sha256"),
                f"{arm} frozen replay hash {control_id}",
            )
            if replay_value.get("frozen_control") is None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Required frozen replay lacks its control output"
                )
    if observed["replays_sha256"] != canonical_sha256(replays):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Evaluation replay batch hash changed"
        )
    for payload_field, hash_field, inner_hash_field in (
        (
            "metrics_input",
            "metrics_input_sha256",
            "stage_metrics_input_sha256",
        ),
        (
            "stage_metrics",
            "stage_metrics_sha256",
            "stage_metrics_sha256",
        ),
        ("gate_report", "gate_report_sha256", "gate_report_sha256"),
    ):
        payload = _mapping(
            observed[payload_field], f"evaluation {payload_field}"
        )
        digest = _sha256(
            observed[hash_field], f"evaluation {hash_field}"
        )
        if payload.get(inner_hash_field) != digest:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                f"Evaluation {payload_field} lost its hash binding"
            )
    _validate_gate_binding(
        observed["gate_report"], stage=stage, require_pass=False
    )
    proofs = _mapping(
        observed["no_leverage_proofs"],
        "evaluation no-leverage proofs",
    )
    if set(proofs) != _required_proof_ids(stage):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Evaluation lacks an exact required no-leverage proof"
        )
    for proof_id, proof in proofs.items():
        proof_value = _mapping(proof, f"no-leverage proof {proof_id}")
        _sha256(
            proof_value.get("proof_sha256"),
            f"no-leverage proof hash {proof_id}",
        )
    if (
        observed["no_leverage_proofs_sha256"]
        != canonical_sha256(proofs)
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "No-leverage proof batch hash changed"
        )
    return observed


def _phase_receipt(
    *,
    output: Mapping[str, Any],
    elapsed: float,
    deadline: float,
    market_elapsed_seconds_hex: str | None,
) -> dict[str, Any]:
    body = {
        "phase": output["phase"],
        "phase_output_sha256": output["phase_output_sha256"],
        "payload_sha256": output["payload_sha256"],
        "counters": copy.deepcopy(output["counters"]),
        "elapsed_seconds_hex": elapsed.hex(),
        "market_elapsed_seconds_hex": market_elapsed_seconds_hex,
        "deadline_monotonic_hex": deadline.hex(),
    }
    return {**body, "phase_receipt_sha256": canonical_sha256(body)}


def _receipt_elapsed_seconds(
    receipts: Sequence[Mapping[str, Any]],
) -> float:
    total = 0.0
    for receipt in receipts:
        total += _nonnegative_float_hex(
            receipt.get("elapsed_seconds_hex"),
            "phase receipt elapsed time",
        )
    return total


def _extend_deterministic_phase_receipt(
    receipts: Sequence[Mapping[str, Any]],
    *,
    additional_elapsed: float,
    deadline: float,
) -> list[dict[str, Any]]:
    elapsed = _strict_elapsed(
        additional_elapsed,
        "deterministic evaluation elapsed time",
    )
    fixed = copy.deepcopy(list(receipts))
    matches = [
        index
        for index, receipt in enumerate(fixed)
        if receipt.get("phase") == "deterministic"
    ]
    if len(matches) != 1:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Scored attempt lacks one deterministic phase receipt"
        )
    index = matches[0]
    receipt = fixed[index]
    body = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "phase_receipt_sha256"
    }
    prior_elapsed = _nonnegative_float_hex(
        body["elapsed_seconds_hex"],
        "deterministic payload elapsed time",
    )
    body["elapsed_seconds_hex"] = (
        prior_elapsed + elapsed
    ).hex()
    body["deadline_monotonic_hex"] = deadline.hex()
    fixed[index] = {
        **body,
        "phase_receipt_sha256": canonical_sha256(body),
    }
    return fixed


@dataclass(frozen=True, slots=True)
class _GovernanceDeadlines:
    intent_prepare: float
    publication: float
    finalization: float


def _governance_deadlines(
    *,
    start: float,
    strict_parent_deadline: float,
    receipts: Sequence[Mapping[str, Any]],
) -> _GovernanceDeadlines:
    phase_elapsed = _receipt_elapsed_seconds(receipts)
    governance_start = start + phase_elapsed
    intent_prepare = min(
        strict_parent_deadline,
        governance_start + _INTENT_PREPARATION_SECONDS,
    )
    publication = min(
        strict_parent_deadline,
        governance_start
        + _INTENT_PREPARATION_SECONDS
        + _NORMAL_PUBLICATION_SECONDS,
    )
    finalization = min(
        strict_parent_deadline,
        governance_start
        + _INTENT_PREPARATION_SECONDS
        + _NORMAL_PUBLICATION_SECONDS
        + _TERMINAL_FINALIZATION_SECONDS,
    )
    if (
        _INTENT_PREPARATION_SECONDS
        + _NORMAL_PUBLICATION_SECONDS
        + _TERMINAL_FINALIZATION_SECONDS
        != _GOVERNANCE_CONTINGENCY_SECONDS
        or intent_prepare <= governance_start
        or publication <= intent_prepare
        or finalization <= publication
        or finalization - start >= MAX_TOTAL_RUNTIME_SECONDS
    ):
        raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
            "Governance partition deadlines are outside the frozen parent cap"
        )
    return _GovernanceDeadlines(
        intent_prepare=intent_prepare,
        publication=publication,
        finalization=finalization,
    )


def _budget_report(
    *,
    command: str,
    receipts: Sequence[Mapping[str, Any]],
    start: float,
    end: float,
) -> dict[str, Any]:
    """Build the immutable in-joint budget attestation.

    For effectful attempts ``end`` is the conservative terminal deadline:
    measured phase time plus the frozen governance contingency.  Returning a
    successful sealed result is separately guarded against this same deadline
    after its canonical hash has been constructed.
    """
    group_elapsed = {name: 0.0 for name in _GROUP_CAPS}
    market_elapsed = 0.0
    counters = {key: 0 for key in _COUNTER_KEYS}
    fixed_receipts = copy.deepcopy(list(receipts))
    for receipt in fixed_receipts:
        phase = receipt["phase"]
        elapsed = float.fromhex(receipt["elapsed_seconds_hex"])
        group_elapsed[_PHASE_GROUP[phase]] += elapsed
        market_hex = receipt["market_elapsed_seconds_hex"]
        if phase == "acquisition":
            market_elapsed += _nonnegative_float_hex(
                market_hex,
                "acquisition phase market elapsed time",
            )
        elif market_hex is not None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Non-acquisition phase exposed market elapsed time"
            )
        for key, count in receipt["counters"].items():
            counters[key] += count
    total = end - start
    checks = {
        "acquisition_seconds_at_most": (
            group_elapsed["acquisition"] <= MAX_SEC_SECONDS
        ),
        "market_seconds_at_most": (
            market_elapsed <= MAX_MARKET_SECONDS
        ),
        "gemma_seconds_at_most": (
            group_elapsed["gemma"] <= MAX_MODEL_SECONDS
        ),
        "deterministic_seconds_at_most": (
            group_elapsed["deterministic"]
            <= MAX_DETERMINISTIC_SECONDS
        ),
        "total_seconds_strictly_below": (
            total < MAX_TOTAL_RUNTIME_SECONDS
        ),
        "market_requests_exactly_six": (
            counters["market_request_count"]
            in {
                0,
                MAX_MARKET_REQUESTS_PER_STAGE,
            }
        ),
        "no_retry_or_fallback": (
            counters["retry_count"] == 0
            and counters["fallback_count"] == 0
        ),
        "no_model_pull_or_paid_api": (
            counters["model_pull_count"] == 0
            and counters["paid_api_call_count"] == 0
        ),
    }
    if command in {
        DEVELOPMENT_ACQUISITION_COMMAND,
        CONFIRMATION_COMMAND,
        FINAL_COMMAND,
    } and counters["market_request_count"] != MAX_MARKET_REQUESTS_PER_STAGE:
        checks["market_requests_exactly_six"] = False
    if any(value is not True for value in checks.values()):
        raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
            "One or more external phase budgets were exceeded"
        )
    body = {
        "schema_version": PHASE_BUDGET_REPORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": command,
        "phase_receipts": fixed_receipts,
        "phase_receipts_sha256": canonical_sha256(fixed_receipts),
        "group_elapsed_seconds_hex": {
            key: value.hex() for key, value in group_elapsed.items()
        },
        "market_elapsed_seconds_hex": market_elapsed.hex(),
        "total_elapsed_seconds_hex": total.hex(),
        "aggregate_counters": counters,
        "checks": checks,
    }
    return {**body, "phase_budget_report_sha256": canonical_sha256(body)}


def _receipt_dict(value: Any) -> dict[str, Any]:
    fields = (
        "table",
        "identity",
        "attempt_id",
        "payload_sha256",
        "journal_sequence",
        "journal_entry_sha256",
    )
    result: dict[str, Any] = {}
    for field in fields:
        if not hasattr(value, field):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Store receipt shape changed"
            )
        result[field] = getattr(value, field)
    return result


def _with_self_hash(
    body: Mapping[str, Any], hash_field: str
) -> dict[str, Any]:
    fixed = copy.deepcopy(dict(body))
    if hash_field in fixed:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            f"{hash_field} must not be caller supplied"
        )
    return {**fixed, hash_field: canonical_sha256(fixed)}


def _receipt_sha256(value: Any) -> str:
    try:
        material = store_record_receipt_material(value)
    except Exception as exc:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Store receipt authority changed"
        ) from exc
    return canonical_sha256(material)


def _opaque_material(value: Any) -> dict[str, Any]:
    method = getattr(value, "as_dict", None)
    if not callable(method):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Opaque durable authority exposes no validated material"
        )
    return _mapping(method(), "opaque durable authority")


def _append_phase_evidence(
    *,
    store: StageStore,
    capability: Any,
    command: str,
    phase_outputs: Mapping[str, Mapping[str, Any]],
    phase_receipts: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    by_phase = {item["phase"]: item for item in phase_receipts}
    persisted: dict[str, dict[str, Any]] = {}
    attempt_id = _identity(
        getattr(capability, "attempt_id", None),
        "phase evidence attempt",
    )
    for phase, output in phase_outputs.items():
        effect = _PHASE_EFFECTS[phase][0]
        evidence = {
            "command": command,
            "phase": phase,
            "phase_output_sha256": output["phase_output_sha256"],
            "payload_sha256": output["payload_sha256"],
            "phase_receipt": by_phase[phase],
        }
        if phase == "acquisition":
            evidence["verified_acquisition_report"] = copy.deepcopy(
                output["payload"]["verified_acquisition_report"]
            )
            evidence["public_summary"] = copy.deepcopy(
                output["payload"]["public_summary"]
            )
            evidence["request_accounting"] = copy.deepcopy(
                output["payload"]["request_accounting"]
            )
        if phase == "gemma":
            semantic_payload = copy.deepcopy(output["payload"])
            latency_receipt = _latency_preflight_receipt(
                semantic_payload
            )
            evidence["semantic_payload"] = semantic_payload
            evidence["latency_preflight_receipt"] = latency_receipt
            evidence["latency_preflight_receipt_sha256"] = (
                latency_receipt[
                    "latency_preflight_receipt_sha256"
                ]
            )
        if phase == "runtime_identity":
            evidence["runtime_receipt"] = copy.deepcopy(runtime_receipt)
        receipt = store.append_evidence(
            capability=capability,
            effect=effect,
            identity=f"phase:{attempt_id}:{phase}",
            payload=evidence,
        )
        persisted[phase] = {
            "payload": copy.deepcopy(evidence),
            "payload_sha256": canonical_sha256(evidence),
            "store_receipt_sha256": _receipt_sha256(receipt),
        }
    return persisted


def _iter_predictions(
    evaluation: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for arm, family in evaluation["replays"].items():
        online = family["online"]
        for row in online["primary"]["predictions"]:
            rows.append((f"{arm}:online", row))
        for control_id, replay in family["frozen_controls"].items():
            frozen = replay["frozen_control"]
            for row in frozen["predictions"]:
                rows.append((f"{arm}:frozen:{control_id}", row))
    return rows


def _iter_ledgers(
    evaluation: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    replays = evaluation["replays"]
    semantic = replays["semantic"]["online"]
    no_meaning = replays["no_filing_meaning"]["online"]
    no_gemma = replays["no_gemma_channel"]["online"]
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for cost in ("cost_5bps", "cost_10bps"):
        rows.extend(
            [
                (
                    f"semantic:{cost}",
                    semantic["primary"]["ledgers"][cost]["combined"],
                ),
                (
                    f"baseline:{cost}",
                    semantic["primary"]["ledgers"][cost]["baseline"],
                ),
                (
                    f"aapl_buy_and_hold:{cost}",
                    semantic["primary"]["ledgers"][cost][
                        "aapl_buy_and_hold"
                    ],
                ),
                (
                    f"no_filing_meaning:{cost}",
                    no_meaning["primary"]["ledgers"][cost]["combined"],
                ),
                (
                    f"no_gemma_channel:{cost}",
                    no_gemma["primary"]["ledgers"][cost]["combined"],
                ),
            ]
        )
        for control_id, replay in replays["semantic"][
            "frozen_controls"
        ].items():
            rows.append(
                (
                    f"frozen:{control_id}:{cost}",
                    replay["frozen_control"]["ledgers"][cost]["combined"],
                )
            )
    return rows


def _persist_scored_evaluation(
    *,
    store: StageStore,
    capability: Any,
    input_bundle: Mapping[str, Any],
    current_feature_rows: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> None:
    stage = input_bundle["stage"]
    for row in current_feature_rows:
        store.append_feature(
            capability=capability,
            effect="gemma_batch",
            identity=f"feature:{row['feature_row_sha256']}",
            payload=row,
        )
    for namespace, row in _iter_predictions(evaluation):
        store.append_prediction(
            capability=capability,
            effect="chronological_replay",
            identity=(
                f"prediction:{stage}:{namespace}:"
                f"{row['prediction_row_sha256']}"
            ),
            payload=row,
        )
    semantic = evaluation["replays"]["semantic"]["online"]
    for row in semantic["learner_lessons"]:
        store.append_lesson(
            capability=capability,
            effect="chronological_replay",
            identity=f"lesson:{stage}:{row['lesson_row_sha256']}",
            payload=row,
        )
    for ledger_id, ledger in _iter_ledgers(evaluation):
        store.append_ledger(
            capability=capability,
            effect="chronological_replay",
            identity=f"ledger:{stage}:{ledger_id}",
            payload=ledger,
        )


def _joint_report(
    *,
    command: str,
    attempt_plan: Mapping[str, Any],
    implementation_manifest: Mapping[str, Any],
    phase_budget_report: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    acquisition_summary: Mapping[str, Any] | None,
    governance_checks: Mapping[str, bool],
) -> dict[str, Any]:
    body = {
        "schema_version": JOINT_STAGE_REPORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": command,
        "metric_stage": _COMMAND_TO_METRIC_STAGE.get(command),
        "attempt_id": attempt_plan["attempt_id"],
        "attempt_plan_sha256": attempt_plan["attempt_plan_sha256"],
        "implementation_manifest_sha256": implementation_manifest[
            "implementation_manifest_sha256"
        ],
        "phase_budget_report": copy.deepcopy(
            dict(phase_budget_report)
        ),
        "phase_budget_report_sha256": phase_budget_report[
            "phase_budget_report_sha256"
        ],
        "runtime_receipt": (
            None
            if runtime_receipt is None
            else copy.deepcopy(dict(runtime_receipt))
        ),
        "runtime_receipt_sha256": (
            None
            if runtime_receipt is None
            else canonical_sha256(runtime_receipt)
        ),
        "deterministic_evaluation": (
            None if evaluation is None else copy.deepcopy(dict(evaluation))
        ),
        "deterministic_evaluation_sha256": (
            None
            if evaluation is None
            else evaluation["deterministic_evaluation_sha256"]
        ),
        "acquisition_summary": (
            None
            if acquisition_summary is None
            else copy.deepcopy(dict(acquisition_summary))
        ),
        "acquisition_summary_sha256": (
            None
            if acquisition_summary is None
            else canonical_sha256(acquisition_summary)
        ),
        "governance_checks": copy.deepcopy(dict(governance_checks)),
        "governance_checks_sha256": canonical_sha256(
            governance_checks
        ),
    }
    return {**body, "joint_stage_report_sha256": canonical_sha256(body)}


def _build_terminal_reconstruction_material(
    *,
    implementation_manifest: Mapping[str, Any],
    store: StageStore,
    attempt_plan: Mapping[str, Any],
    terminal_status: str,
    report_kind: str,
    terminal_artifact: Mapping[str, Any],
    terminal_artifact_receipt: Any,
    report_material: Mapping[str, Any],
    persisted_phase_evidence: Mapping[str, Mapping[str, Any]],
    acquisition_execution: AcquisitionExecutionResult | None = None,
    gate_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    plan = validate_attempt_plan(
        attempt_plan,
        implementation_manifest=implementation,
    )
    artifact = _mapping(
        terminal_artifact, "terminal reconstruction artifact"
    )
    receipt = store_record_receipt_material(
        terminal_artifact_receipt
    )
    receipt_sha256 = canonical_sha256(receipt)
    material = _mapping(
        report_material, "terminal reconstruction record commitment"
    )
    common = {
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": implementation[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
    }
    if plan["attempt_kind"] == DEVELOPMENT_ACQUISITION:
        if report_kind != ACQUISITION_PASS or gate_checks is not None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Acquisition reconstruction report kind changed"
            )
        acquisition_evidence = _mapping(
            persisted_phase_evidence.get("acquisition"),
            "sealed acquisition phase evidence",
        )
        if acquisition_execution is not None:
            handle = acquisition_execution.vault_handle
            seal_sha256 = handle.seal_sha256
        else:
            seal_sha256 = canonical_sha256(
                {
                    "test_only_acquisition_report_sha256": artifact[
                        "validation_sha256"
                    ]
                }
            )
        vault_commitments = {
            "bundle_sha256": artifact["bundle_sha256"],
            "manifest_sha256": artifact["manifest_sha256"],
            "private_index_sha256": artifact[
                "private_index_sha256"
            ],
            "seal_sha256": seal_sha256,
        }
        body = {
            "schema_version": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "acquisition-terminal-reconstruction-v1"
            ),
            "reconstruction_verifier_id": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "runner-terminal-reconstruction-verifier-v1"
            ),
            **common,
            "stage": "development",
            "terminal_status": terminal_status,
            "report_kind": report_kind,
            "terminal_artifact_sha256": receipt["payload_sha256"],
            "terminal_artifact_store_receipt_sha256": receipt_sha256,
            "acquisition_validation_sha256": artifact[
                "validation_sha256"
            ],
            "bundle_sha256": artifact["bundle_sha256"],
            "manifest_sha256": artifact["manifest_sha256"],
            "private_index_sha256": artifact[
                "private_index_sha256"
            ],
            "check_set_sha256": artifact["check_set_sha256"],
            "record_counts": copy.deepcopy(
                material["record_counts"]
            ),
            "record_commitment_sha256": material[
                "record_commitment_sha256"
            ],
            "acquisition_artifact_receipt_sha256": receipt_sha256,
            "sealed_acquisition_phase_evidence_sha256": (
                canonical_sha256(acquisition_evidence)
            ),
            "sealed_vault_commitments_sha256": canonical_sha256(
                vault_commitments
            ),
        }
        result = _with_self_hash(
            body, "terminal_reconstruction_material_sha256"
        )
        if tuple(result) != ACQUISITION_TERMINAL_RECONSTRUCTION_FIELDS:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Acquisition reconstruction fields changed"
            )
        return result

    checks = _mapping(
        gate_checks, "scored reconstruction gate checks"
    )
    evaluation = _mapping(
        artifact["deterministic_evaluation"],
        "scored reconstruction deterministic evaluation",
    )
    failed = [
        name for name, passed in checks.items() if passed is not True
    ]
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "scored-terminal-reconstruction-v1"
        ),
        "reconstruction_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "runner-terminal-reconstruction-verifier-v1"
        ),
        **common,
        "stage": artifact["metric_stage"],
        "terminal_status": terminal_status,
        "report_kind": report_kind,
        "terminal_artifact_sha256": receipt["payload_sha256"],
        "terminal_artifact_store_receipt_sha256": receipt_sha256,
        "stage_input_bundle_sha256": evaluation[
            "stage_input_bundle_sha256"
        ],
        "deterministic_evaluation_sha256": evaluation[
            "deterministic_evaluation_sha256"
        ],
        "stage_metrics_input_sha256": evaluation[
            "metrics_input_sha256"
        ],
        "stage_metrics_sha256": evaluation["stage_metrics_sha256"],
        "gate_report_sha256": evaluation["gate_report_sha256"],
        "no_leverage_proofs_sha256": evaluation[
            "no_leverage_proofs_sha256"
        ],
        "joint_stage_report_sha256": artifact[
            "joint_stage_report_sha256"
        ],
        "joint_artifact_receipt_sha256": receipt_sha256,
        "record_counts": copy.deepcopy(material["record_counts"]),
        "record_commitment_sha256": material[
            "record_commitment_sha256"
        ],
        "gate_checks": checks,
        "gate_check_set_sha256": canonical_sha256(checks),
        "failed_gate_names": failed,
    }
    result = _with_self_hash(
        body, "terminal_reconstruction_material_sha256"
    )
    if tuple(result) != SCORED_TERMINAL_RECONSTRUCTION_FIELDS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Scored reconstruction fields changed"
        )
    return result


def _build_publication_intent(
    *,
    implementation_manifest: Mapping[str, Any],
    store: StageStore,
    attempt_plan: Mapping[str, Any],
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
    artifact_receipt: Any,
    reconstruction: Any,
    report_material: Mapping[str, Any],
    elapsed_seconds: float,
    predecessor_publication_sha256: str,
    prepared_publication: PreparedExternalPublication,
) -> dict[str, Any]:
    snapshot = _mapping(
        store.snapshot(), "publication intent store snapshot"
    )
    reconstruction_material = _opaque_material(reconstruction)
    prepared = _mapping(
        prepared_publication.intent_material,
        "prepared publication intent material",
    )
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-intent-v1"
        ),
        "intent_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-intent-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation_manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": implementation_manifest[
            "implementation_commit"
        ],
        "attempt_id": attempt_plan["attempt_id"],
        "attempt_kind": attempt_plan["attempt_kind"],
        "attempt_plan_sha256": attempt_plan["attempt_plan_sha256"],
        "terminal_status": terminal_status,
        "report_kind": report_kind,
        "artifact_sha256": artifact_sha256,
        "artifact_store_receipt_sha256": _receipt_sha256(
            artifact_receipt
        ),
        "terminal_reconstruction_material_sha256": (
            reconstruction_material[
                "terminal_reconstruction_material_sha256"
            ]
        ),
        "terminal_reconstruction_material_store_receipt_sha256": (
            _receipt_sha256(reconstruction.store_receipt)
        ),
        "record_counts": copy.deepcopy(
            report_material["record_counts"]
        ),
        "record_commitment_sha256": report_material[
            "record_commitment_sha256"
        ],
        "store_instance_id": store.store_instance_id,
        "store_journal_sequence": snapshot["journal_entry_count"],
        "store_journal_tip_sha256": snapshot["journal_tip_sha256"],
        "normal_attempt_elapsed_at_intent_prepare_hex": (
            elapsed_seconds.hex()
        ),
        "predecessor_publication_sha256": (
            predecessor_publication_sha256
        ),
        **prepared,
        "intent_status": "publication_pending",
        "research_effect_authority_invalidated": True,
        "semantic_result_release_blocked": True,
        "next_stage_authority_blocked": True,
        "external_cost_usd": 0,
    }
    return _with_self_hash(body, "publication_intent_sha256")


def _publication_pending_result(
    *,
    command: str,
    attempt_id: str,
    publication_intent: Any,
) -> dict[str, Any]:
    intent = _opaque_material(publication_intent)
    body = {
        "schema_version": (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "publication-pending-result-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": command,
        "attempt_id": attempt_id,
        "result_status": "publication_pending",
        "publication_intent_sha256": intent[
            "publication_intent_sha256"
        ],
        "publication_intent_store_receipt_sha256": _receipt_sha256(
            publication_intent.store_receipt
        ),
        "semantic_result_released": False,
        "next_stage_authority_blocked": True,
        "publication_recovery_required": True,
        "external_cost_usd": 0,
    }
    return {
        **body,
        "publication_pending_result_sha256": canonical_sha256(body),
    }


def _build_pre_push_authorization(
    *,
    implementation_manifest: Mapping[str, Any],
    store: StageStore,
    attempt_id: str,
    publication_intent_sha256: str,
    capability: Any,
    worker_ownership: Any,
    remote_observation: Any,
    operation_ordinal: int | None = None,
) -> dict[str, Any]:
    owner = _opaque_material(worker_ownership)
    observation = _opaque_material(remote_observation)
    if capability.operation_kind == NORMAL_PUBLICATION:
        fixed_operation_ordinal = 0
    else:
        if type(operation_ordinal) is not int or operation_ordinal < 1:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Recovery push authorization requires its exact invocation ordinal"
            )
        fixed_operation_ordinal = operation_ordinal
    marker_body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-pre-push-authorization-v1"
        ),
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": (
            store.store_session_nonce_sha256
        ),
        "attempt_id": attempt_id,
        "publication_intent_sha256": publication_intent_sha256,
        "authorization_operation_kind": capability.operation_kind,
        "authorization_operation_sha256": capability.operation_sha256,
        "authorization_operation_ordinal": fixed_operation_ordinal,
        "worker_ownership_sha256": owner[
            "worker_ownership_sha256"
        ],
    }
    body = {
        "schema_version": marker_body["schema_version"],
        "authorization_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-pre-push-authorization-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation_manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": (
            store.store_session_nonce_sha256
        ),
        "attempt_id": attempt_id,
        "publication_intent_sha256": publication_intent_sha256,
        "authorization_operation_kind": capability.operation_kind,
        "authorization_operation_sha256": capability.operation_sha256,
        "authorization_operation_ordinal": fixed_operation_ordinal,
        "worker_ownership_sha256": owner[
            "worker_ownership_sha256"
        ],
        "pre_push_authorization_marker_key": canonical_sha256(
            marker_body
        ),
        "remote_observation_sha256": observation[
            "publication_remote_observation_sha256"
        ],
        "tag_ref": observation["tag_ref"],
        "expected_tag_object_sha1": observation[
            "expected_tag_object_sha1"
        ],
        "authorization_status": "push_authorized_once",
        "push_command_limit": 1,
    }
    return _with_self_hash(body, "pre_push_authorization_sha256")


def _build_publication_conflict(
    *,
    implementation_manifest: Mapping[str, Any],
    store: StageStore,
    attempt_id: str,
    publication_intent_sha256: str,
    capability: Any,
    worker_ownership: Any,
    remote_observation: Any,
) -> dict[str, Any]:
    owner = _opaque_material(worker_ownership)
    observation = _opaque_material(remote_observation)
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-conflict-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation_manifest[
            "implementation_manifest_sha256"
        ],
        "store_instance_id": store.store_instance_id,
        "attempt_id": attempt_id,
        "publication_intent_sha256": publication_intent_sha256,
        "observation_operation_kind": capability.operation_kind,
        "observation_operation_sha256": capability.operation_sha256,
        "tag_ref": observation["tag_ref"],
        "remote_observation_sha256": observation[
            "publication_remote_observation_sha256"
        ],
        "conflict_reason": "durable_remote_ref_conflict",
        "poisoned": True,
        "prior_governance_record_sha256": owner[
            "worker_ownership_sha256"
        ],
    }
    return _with_self_hash(body, "publication_conflict_sha256")


def _build_publication_receipt(
    *,
    implementation_manifest: Mapping[str, Any],
    store: StageStore,
    attempt_id: str,
    publication_intent: Any,
    remote_observation: Any,
    pre_push_authorization: Any | None,
    external_publication: VerifiedExternalPublication,
    recovery_completion: Any | None = None,
) -> dict[str, Any]:
    snapshot = _mapping(
        store.snapshot(), "publication receipt store snapshot"
    )
    intent = _opaque_material(publication_intent)
    observation = _opaque_material(remote_observation)
    publication = external_publication.publication
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-receipt-v1"
        ),
        "receipt_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-receipt-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation_manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "attempt_id": attempt_id,
        "publication_intent_sha256": intent[
            "publication_intent_sha256"
        ],
        "publication_intent_store_receipt_sha256": _receipt_sha256(
            publication_intent.store_receipt
        ),
        "publication_remote_observation_sha256": observation[
            "publication_remote_observation_sha256"
        ],
        "recovery_invocation_completion_sha256": (
            PUBLICATION_NORMAL_NO_RECOVERY_COMPLETION_SHA256
            if recovery_completion is None
            else _opaque_material(recovery_completion)[
                "recovery_invocation_completion_sha256"
            ]
        ),
        "pre_push_authorization_sha256": (
            PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
            if pre_push_authorization is None
            else _opaque_material(pre_push_authorization)[
                "pre_push_authorization_sha256"
            ]
        ),
        "external_publication_sha256": publication[
            "publication_sha256"
        ],
        "remote_tag_object_sha1": publication[
            "remote_tag_object_sha1"
        ],
        "remote_peeled_commit": publication["remote_peeled_commit"],
        "pre_receipt_store_journal_sequence": snapshot[
            "journal_entry_count"
        ],
        "pre_receipt_store_journal_tip_sha256": snapshot[
            "journal_tip_sha256"
        ],
        "receipt_status": "publication_verified",
        "publication_capability_invalidated": True,
        "terminalization_capability_required": True,
    }
    return _with_self_hash(body, "publication_receipt_sha256")


def _build_publication_recovery_start(
    *,
    implementation_manifest: Mapping[str, Any],
    store: StageStore,
    attempt_id: str,
    publication_intent: Any,
) -> dict[str, Any]:
    intent = _opaque_material(publication_intent)
    starts = store.governance_records(
        "publication_recovery_invocation_starts",
        attempt_id=attempt_id,
    )
    completions = store.governance_records(
        "publication_recovery_invocation_completions",
        attempt_id=attempt_id,
    )
    if len(starts) != len(completions):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Publication recovery start/completion chain was not reconciled"
        )
    if completions:
        prior_completion = completions[-1]
        prior_completion_sha256 = prior_completion[
            "recovery_invocation_completion_sha256"
        ]
        prior_cumulative = prior_completion[
            "cumulative_recovery_seconds"
        ]
    else:
        prior_completion_sha256 = (
            PUBLICATION_RECOVERY_COMPLETION_GENESIS_SHA256
        )
        prior_cumulative = "0x0.0p+0"
    snapshot = _mapping(
        store.snapshot(), "publication recovery start store snapshot"
    )
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-start-v1"
        ),
        "start_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-start-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation_manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": (
            store.store_session_nonce_sha256
        ),
        "attempt_id": attempt_id,
        "publication_intent_sha256": intent[
            "publication_intent_sha256"
        ],
        "invocation_ordinal": len(starts) + 1,
        "prior_recovery_completion_sha256": prior_completion_sha256,
        "prior_cumulative_recovery_seconds": prior_cumulative,
        "start_status": "started",
        "invocation_seconds_cap": MAX_PUBLICATION_RECOVERY_SECONDS,
        "durable_pre_push_authorization_required": True,
        "pre_start_store_journal_sequence": snapshot[
            "journal_entry_count"
        ],
        "pre_start_store_journal_tip_sha256": snapshot[
            "journal_tip_sha256"
        ],
    }
    result = _with_self_hash(
        body, "recovery_invocation_start_sha256"
    )
    if tuple(result) != PUBLICATION_RECOVERY_INVOCATION_START_FIELDS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Publication recovery start fields changed"
        )
    return result


def _build_publication_recovery_completion(
    *,
    implementation_manifest: Mapping[str, Any],
    store: StageStore,
    attempt_id: str,
    publication_intent: Any,
    recovery_start: Any,
    outcome: str,
    remote_observation: Any | None,
    pre_push_authorization: Any | None,
    push_command_count_upper_bound: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    intent = _opaque_material(publication_intent)
    start = _opaque_material(recovery_start)
    if (
        type(elapsed_seconds) not in {int, float}
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0.0
        or float(elapsed_seconds) > MAX_PUBLICATION_RECOVERY_SECONDS
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Publication recovery elapsed time exceeded its fixed cap"
        )
    elapsed = float(elapsed_seconds)
    cumulative = (
        float.fromhex(start["prior_cumulative_recovery_seconds"])
        + elapsed
    )
    remote_hash = (
        PUBLICATION_RECOVERY_NO_REMOTE_OBSERVATION_SHA256
        if remote_observation is None
        else _opaque_material(remote_observation)[
            "publication_remote_observation_sha256"
        ]
    )
    authorization_hash = (
        PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
        if pre_push_authorization is None
        else _opaque_material(pre_push_authorization)[
            "pre_push_authorization_sha256"
        ]
    )
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-completion-v1"
        ),
        "completion_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-recovery-completion-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation_manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": implementation_manifest[
            "implementation_commit"
        ],
        "store_instance_id": store.store_instance_id,
        "store_session_nonce_sha256": (
            store.store_session_nonce_sha256
        ),
        "attempt_id": attempt_id,
        "publication_intent_sha256": intent[
            "publication_intent_sha256"
        ],
        "recovery_invocation_start_sha256": start[
            "recovery_invocation_start_sha256"
        ],
        "invocation_ordinal": start["invocation_ordinal"],
        "prior_recovery_completion_sha256": start[
            "prior_recovery_completion_sha256"
        ],
        "pre_push_authorization_sha256": authorization_hash,
        "completion_status": "completed",
        "outcome": outcome,
        "remote_observation_sha256": remote_hash,
        "push_command_count_upper_bound": (
            push_command_count_upper_bound
        ),
        "elapsed_seconds": elapsed.hex(),
        "cumulative_recovery_seconds": cumulative.hex(),
    }
    result = _with_self_hash(
        body, "recovery_invocation_completion_sha256"
    )
    if tuple(result) != PUBLICATION_RECOVERY_INVOCATION_COMPLETION_FIELDS:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Publication recovery completion fields changed"
        )
    return result


def _failure_result(
    *,
    command: str,
    attempt_id: str,
    terminal_status: str,
    diagnostic_code: str,
    terminal_transition: Mapping[str, Any],
    terminal_evidence: Mapping[str, Any] | None = None,
    external_publication: Mapping[str, Any] | None = None,
    terminal_artifact: Mapping[str, Any] | None = None,
    terminal_artifact_receipt: Any | None = None,
) -> dict[str, Any]:
    optional_values = (
        terminal_evidence,
        external_publication,
        terminal_artifact,
        terminal_artifact_receipt,
    )
    has_evidence = all(value is not None for value in optional_values)
    has_partial_evidence = any(
        value is not None for value in optional_values
    ) and not has_evidence
    if has_partial_evidence:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Sealed failure terminal evidence must be complete or absent"
        )
    if terminal_status == TERMINAL_FAIL:
        if (
            command not in _SCORING_COMMANDS
            or diagnostic_code != "gate_failure_sealed"
            or not has_evidence
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Terminal-fail is reserved for a complete sealed scored gate failure"
            )
    elif terminal_status == TERMINAL_INDETERMINATE:
        if has_evidence:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Terminal-indeterminate cannot release semantic terminal evidence"
            )
    else:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Failure result status must be terminal-fail or terminal-indeterminate"
        )
    body = {
        "schema_version": SEALED_STAGE_RESULT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": command,
        "attempt_id": attempt_id,
        "terminal_status": terminal_status,
        "passed": False,
        "diagnostic_code": _identity(
            diagnostic_code, "failure diagnostic code"
        ),
        "terminal_transition": copy.deepcopy(dict(terminal_transition)),
        "terminal_transition_sha256": terminal_transition[
            "transition_sha256"
        ],
        "terminal_evidence": (
            None
            if terminal_evidence is None
            else copy.deepcopy(dict(terminal_evidence))
        ),
        "external_publication": (
            None
            if external_publication is None
            else copy.deepcopy(dict(external_publication))
        ),
        "terminal_artifact": (
            None
            if terminal_artifact is None
            else copy.deepcopy(dict(terminal_artifact))
        ),
        "terminal_artifact_receipt": (
            None
            if terminal_artifact_receipt is None
            else _receipt_dict(terminal_artifact_receipt)
        ),
    }
    return {**body, "sealed_stage_result_sha256": canonical_sha256(body)}


def _success_result(
    *,
    command: str,
    attempt_id: str,
    terminal_transition: Mapping[str, Any],
    terminal_evidence: Mapping[str, Any],
    external_publication: Mapping[str, Any],
    terminal_artifact: Mapping[str, Any],
    terminal_artifact_receipt: Any,
) -> dict[str, Any]:
    body = {
        "schema_version": SEALED_STAGE_RESULT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": command,
        "attempt_id": attempt_id,
        "terminal_status": TERMINAL_PASS,
        "passed": True,
        "diagnostic_code": None,
        "terminal_transition": copy.deepcopy(dict(terminal_transition)),
        "terminal_transition_sha256": terminal_transition[
            "transition_sha256"
        ],
        "terminal_evidence": copy.deepcopy(dict(terminal_evidence)),
        "external_publication": copy.deepcopy(
            dict(external_publication)
        ),
        "terminal_artifact": copy.deepcopy(dict(terminal_artifact)),
        "terminal_artifact_receipt": _receipt_dict(
            terminal_artifact_receipt
        ),
    }
    return {**body, "sealed_stage_result_sha256": canonical_sha256(body)}


def validate_sealed_stage_result(
    value: Mapping[str, Any],
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_plan: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _mapping(value, "sealed stage result")
    expected_keys = {
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
    if set(observed) != expected_keys:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Sealed stage result keys changed"
        )
    _self_hash(
        observed,
        hash_field="sealed_stage_result_sha256",
        location="sealed stage result",
    )
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    plan = validate_attempt_plan(
        attempt_plan,
        implementation_manifest=implementation,
    )
    if (
        observed["schema_version"] != SEALED_STAGE_RESULT_SCHEMA_VERSION
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["attempt_id"] != plan["attempt_id"]
        or observed["command"] not in _EFFECTFUL_COMMANDS
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Sealed stage result identity changed"
        )
    transition = _mapping(
        observed["terminal_transition"], "terminal transition"
    )
    if (
        transition.get("transition_sha256")
        != observed["terminal_transition_sha256"]
        or transition.get("attempt_id") != plan["attempt_id"]
        or transition.get("status") != observed["terminal_status"]
        or transition.get("terminal") is not True
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Sealed result terminal transition changed"
        )
    evidence_values = (
        observed["terminal_evidence"],
        observed["external_publication"],
        observed["terminal_artifact"],
        observed["terminal_artifact_receipt"],
    )
    has_evidence = all(item is not None for item in evidence_values)
    if has_evidence != any(item is not None for item in evidence_values):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Sealed result terminal evidence is partial"
        )
    if observed["passed"] is True:
        if (
            observed["terminal_status"] != TERMINAL_PASS
            or observed["diagnostic_code"] is not None
            or not has_evidence
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Passing result lacks complete terminal evidence"
            )
    else:
        if observed["diagnostic_code"] is None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Failed result terminal identity changed"
            )
        if observed["terminal_status"] == TERMINAL_FAIL:
            if (
                observed["command"] not in _SCORING_COMMANDS
                or observed["diagnostic_code"] != "gate_failure_sealed"
                or not has_evidence
            ):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Terminal-fail lacks a complete sealed scored gate failure"
                )
        elif observed["terminal_status"] == TERMINAL_INDETERMINATE:
            if has_evidence:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Terminal-indeterminate released semantic terminal evidence"
                )
        else:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Failed result terminal identity changed"
            )
    if not has_evidence:
        return observed
    evidence = _mapping(
        observed["terminal_evidence"], "sealed terminal evidence"
    )
    publication = validate_external_publication(
        observed["external_publication"],
        implementation_manifest=implementation,
    )
    artifact = _mapping(
        observed["terminal_artifact"], "sealed terminal artifact"
    )
    receipt = _mapping(
        observed["terminal_artifact_receipt"],
        "sealed terminal artifact receipt",
    )
    _self_hash(
        evidence,
        hash_field="terminal_evidence_sha256",
        location="sealed terminal evidence",
    )
    artifact_receipt_field = (
        "acquisition_artifact_receipt_sha256"
        if observed["command"] == DEVELOPMENT_ACQUISITION_COMMAND
        else "joint_artifact_receipt_sha256"
    )
    if (
        evidence.get("attempt_id") != plan["attempt_id"]
        or evidence.get("terminal_status") != observed["terminal_status"]
        or evidence.get("external_publication_sha256")
        != publication["publication_sha256"]
        or evidence.get(artifact_receipt_field)
        != canonical_sha256(receipt)
        or receipt.get("attempt_id") != plan["attempt_id"]
        or receipt.get("table") != "artifacts"
        or receipt.get("payload_sha256") != canonical_sha256(artifact)
    ):
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "Sealed terminal evidence lost its store or publication binding"
        )
    if observed["command"] == DEVELOPMENT_ACQUISITION_COMMAND:
        if set(evidence) != set(ACQUISITION_TERMINAL_EVIDENCE_FIELDS):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Acquisition terminal evidence fields changed"
            )
        report = _validate_acquisition_report_payload(
            artifact,
            command=observed["command"],
        )
        if publication["artifact_sha256"] != report["validation_sha256"]:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Acquisition publication crossed its validation artifact"
            )
    else:
        if set(evidence) != set(SCORED_TERMINAL_EVIDENCE_FIELDS):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Scored terminal evidence fields changed"
            )
        joint_hash = _self_hash(
            artifact,
            hash_field="joint_stage_report_sha256",
            location="sealed joint stage report",
        )
        if publication["artifact_sha256"] != joint_hash:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Scored publication crossed its joint report"
            )
        evaluation = _mapping(
            artifact["deterministic_evaluation"],
            "sealed deterministic evaluation",
        )
        stage = _COMMAND_TO_METRIC_STAGE[observed["command"]]
        gate = _validate_gate_binding(
            evaluation["gate_report"],
            stage=stage,
            require_pass=observed["passed"] is True,
        )
        if (gate["passed"] is True) != (observed["passed"] is True):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Sealed scored verdict differs from its gate report"
            )
    return observed


class SecGemmaOnlineRiskOverlayRunner:
    """Run one command against one exclusively owned attempt store."""

    def __init__(
        self,
        *,
        repo_root: Path,
        implementation_manifest: Mapping[str, Any],
        store: StageStore,
        phase_executor: PhaseExecutor,
        acquisition_adapter: AcquisitionAdapter | None,
        report_publisher: ExternalReportPublisher,
        publication_runtime: PublicationWorkerRuntime,
        final_registry_authority: FinalRegistryAuthority | None = None,
        production_authorities: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        dependencies: RunnerDependencies | None = None,
        test_only_allow_effects: bool = False,
    ) -> None:
        if not isinstance(repo_root, Path) or not repo_root.is_absolute():
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner repo_root must be an absolute pathlib.Path"
            )
        self._repo_root = repo_root
        self._implementation = validate_implementation_manifest(
            implementation_manifest
        )
        self._store = store
        self._executor = phase_executor
        self._acquisition_adapter = acquisition_adapter
        self._publisher = report_publisher
        self._publication_runtime = publication_runtime
        self._final_registry_authority = final_registry_authority
        self._production_authorities = production_authorities
        self._clock = clock
        self._deps = dependencies or default_runner_dependencies()
        self._test_only_allow_effects = test_only_allow_effects

    def _verify_production_authorities(self, command: str) -> None:
        try:
            from agent_benchmark.sec_gemma_online_risk_overlay_production import (
                is_verified_production_authorities,
            )
        except ImportError as exc:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Verified production authorities are unavailable"
            ) from exc
        authority = self._production_authorities
        if (
            not is_verified_production_authorities(authority)
            or authority.phase_executor is not self._executor
            or authority.acquisition_adapter is not self._acquisition_adapter
            or authority.report_publisher is not self._publisher
            or authority.publication_runtime
            is not self._publication_runtime
            or authority.final_registry_authority
            is not self._final_registry_authority
            or authority.store is not self._store
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner production authorities are foreign or incomplete"
            )
        evaluator = self._deps.evaluator
        if (
            self._clock is not time.monotonic
            or self._deps.live_implementation_manifest
            is not _default_live_implementation_manifest
            or self._deps.runtime_verifier is not _default_runtime_verifier
            or type(evaluator) is not DefaultDeterministicStageEvaluator
            or getattr(evaluator, "__dict__", None) != {}
            or self._deps.issue_acquisition_terminal_evidence_fn
            is not issue_verified_acquisition_terminal_evidence
            or self._deps.issue_scored_terminal_evidence_fn
            is not issue_verified_scored_terminal_evidence
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner production dependencies are injected or mutable"
            )
        try:
            result = authority.reverify()
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner production authority reverification failed"
            ) from exc
        if result is not None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Production authority reverification returned mutable authority"
            )
        try:
            readiness = authority.assert_ready_for_command(command)
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner production acquisition state is not restart-safe"
            ) from exc
        if readiness is not None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Production readiness returned mutable authority"
            )

    def _verify_production_recovery_authorities(self) -> None:
        try:
            from agent_benchmark.sec_gemma_online_risk_overlay_production import (
                is_verified_production_authorities,
            )
        except ImportError as exc:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Verified production recovery authorities are unavailable"
            ) from exc
        authority = self._production_authorities
        if (
            not is_verified_production_authorities(authority)
            or authority.phase_executor is not self._executor
            or authority.acquisition_adapter is not self._acquisition_adapter
            or authority.report_publisher is not self._publisher
            or authority.publication_runtime
            is not self._publication_runtime
            or authority.final_registry_authority
            is not self._final_registry_authority
            or authority.store is not self._store
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner publication-recovery authorities are foreign or incomplete"
            )
        evaluator = self._deps.evaluator
        if (
            self._clock is not time.monotonic
            or self._deps.live_implementation_manifest
            is not _default_live_implementation_manifest
            or self._deps.runtime_verifier is not _default_runtime_verifier
            or type(evaluator) is not DefaultDeterministicStageEvaluator
            or getattr(evaluator, "__dict__", None) != {}
            or self._deps.issue_acquisition_terminal_evidence_fn
            is not issue_verified_acquisition_terminal_evidence
            or self._deps.issue_scored_terminal_evidence_fn
            is not issue_verified_scored_terminal_evidence
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner production recovery dependencies are injected or mutable"
            )
        try:
            result = authority.reverify()
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Runner production recovery authority reverification failed"
            ) from exc
        if result is not None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Production recovery reverification returned mutable authority"
            )

    def _authorize_supervised_publication_recovery_worker(
        self,
        *,
        authority: SupervisedPublicationRecoveryWorkerAuthority | None,
        attempt_id: str,
        invocation_started_at: float,
        local_deadline: float,
    ) -> tuple[float, float]:
        try:
            from agent_benchmark.sec_gemma_online_risk_overlay_production import (
                is_verified_supervised_publication_recovery_worker_authority
                as verifier,
            )
        except ImportError:
            verifier = None
        if not callable(verifier) or not verifier(authority):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Production publication recovery requires an opaque "
                "supervised-worker authority"
            )
        authorize = getattr(
            authority,
            "authorize_runner_recovery",
            None,
        )
        if not callable(authorize):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Supervised publication-recovery worker authority is incomplete"
            )
        try:
            supervised_window = authorize(
                repo_root=self._repo_root,
                implementation_manifest=self._implementation,
                production_authorities=self._production_authorities,
                attempt_id=attempt_id,
                worker_entry_monotonic=invocation_started_at,
                worker_deadline_monotonic=local_deadline,
            )
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Supervised publication-recovery worker authority rejected "
                "this runner"
            ) from exc
        if type(supervised_window) is not tuple or len(supervised_window) != 2:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Supervised publication-recovery window is invalid"
            )
        supervised_started_at, supervised_deadline = supervised_window
        if (
            type(supervised_started_at) is not float
            or type(supervised_deadline) is not float
            or not math.isfinite(supervised_started_at)
            or not math.isfinite(supervised_deadline)
            or supervised_started_at <= 0.0
            or supervised_started_at > invocation_started_at
            or supervised_deadline <= invocation_started_at
            or supervised_deadline
            != supervised_started_at + MAX_PUBLICATION_RECOVERY_SECONDS
            or supervised_deadline > local_deadline
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Supervised publication-recovery window is invalid"
            )
        self._require_before_deadline(
            supervised_deadline,
            "after supervised publication-recovery worker authorization",
        )
        return supervised_started_at, supervised_deadline

    def _verify_live_implementation(self) -> None:
        observed = validate_implementation_manifest(
            self._deps.live_implementation_manifest(self._repo_root)
        )
        if observed != self._implementation:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Live source evidence differs from the runner manifest"
            )

    def _load_predecessor_feature_rows(
        self,
        command: str,
    ) -> list[dict[str, Any]]:
        if command not in _SCORING_COMMANDS:
            return []
        stage = _COMMAND_TO_METRIC_STAGE[command]
        rows = _sequence(
            self._store.predecessor_feature_rows(stage),
            f"{stage} predecessor feature rows",
        )
        canonical, _, _ = _cumulative_feature_rows(
            stage=stage,
            predecessor_rows=rows,
            current_rows=(),
        )
        if canonical != rows:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Predecessor feature rows are not in canonical decision order"
            )
        return rows

    def _verify_predecessor_feature_rows_unchanged(
        self,
        *,
        command: str,
        expected_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        observed = self._load_predecessor_feature_rows(command)
        if (
            canonical_sha256(observed)
            != canonical_sha256(list(expected_rows))
            or observed != list(expected_rows)
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Predecessor feature evidence changed before consumption"
            )

    def _predecessor_transition(
        self, command: str
    ) -> Mapping[str, Any] | None:
        if command not in _EFFECTFUL_COMMANDS:
            return None
        attempt_id = ATTEMPT_ID_BY_KIND[_COMMAND_TO_KIND[command]]
        predecessor_id = PREREQUISITE_ATTEMPT_BY_ID[attempt_id]
        if predecessor_id is None:
            return None
        history = self._store.attempt_history(predecessor_id)
        if len(history) != 3 or history[-1].get("status") != TERMINAL_PASS:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Exact predecessor terminal-pass is required"
            )
        return history[-1]

    def _predecessor_publication_sha256(
        self,
        command: str,
        *,
        final_registry_authorization: (
            VerifiedFinalRegistryAuthorization | None
        ) = None,
    ) -> str:
        if command == DEVELOPMENT_ACQUISITION_COMMAND:
            return PUBLICATION_GENESIS_SHA256
        if command == FINAL_COMMAND:
            if not is_verified_final_registry_authorization(
                final_registry_authorization
            ):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Final publication lacks its registry predecessor"
                )
            return final_registry_authorization.external_publication.publication_sha256
        attempt_id = ATTEMPT_ID_BY_KIND[_COMMAND_TO_KIND[command]]
        predecessor_id = PREREQUISITE_ATTEMPT_BY_ID[attempt_id]
        if predecessor_id is None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Publication predecessor is unavailable"
            )
        binding = _mapping(
            self._store.terminal_anchor_binding(predecessor_id),
            "predecessor terminal anchor binding",
        )
        publication = validate_external_publication(
            binding["external_publication"],
            implementation_manifest=self._implementation,
        )
        if publication["attempt_id"] != predecessor_id:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Publication predecessor crossed its attempt"
            )
        return publication["publication_sha256"]

    def plan(
        self,
        command: str,
        *,
        final_registry_authorization: (
            VerifiedFinalRegistryAuthorization | None
        ) = None,
    ) -> dict[str, Any]:
        predecessor = self._predecessor_transition(command)
        return build_runner_plan(
            command=command,
            implementation_manifest=self._implementation,
            prerequisite_terminal_transition=predecessor,
            final_registry_authorization=final_registry_authorization,
        )

    def _read_clock(self, location: str) -> float:
        try:
            return _strict_elapsed(self._clock(), location)
        except SecGemmaOnlineRiskOverlayRunnerError:
            raise
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                f"{location} monotonic clock failed"
            ) from exc

    def _require_before_deadline(
        self,
        deadline_monotonic: float,
        location: str,
    ) -> float:
        observed = self._read_clock(location)
        if observed >= deadline_monotonic:
            raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                f"{location} reached the strict parent deadline"
            )
        return observed

    def _complete_scored_deterministic_work(
        self,
        *,
        stage: str,
        outputs: Mapping[str, Mapping[str, Any]],
        predecessor_feature_rows: Sequence[Mapping[str, Any]],
        receipts: Sequence[Mapping[str, Any]],
        attempt_deadline: float,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        Mapping[str, Any],
        Mapping[str, Any],
        list[dict[str, Any]],
    ]:
        before = self._read_clock("deterministic evaluation before")
        deterministic_used = 0.0
        for receipt in receipts:
            phase = receipt.get("phase")
            if phase not in _PHASE_GROUP:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Phase receipt contains an unknown budget group"
                )
            if _PHASE_GROUP[phase] == "deterministic":
                deterministic_used += _nonnegative_float_hex(
                    receipt.get("elapsed_seconds_hex"),
                    "deterministic phase receipt elapsed time",
                )
        remaining = MAX_DETERMINISTIC_SECONDS - deterministic_used
        overall_remaining = attempt_deadline - before
        allowed = min(remaining, overall_remaining)
        if allowed <= 0.0:
            raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                "Deterministic evaluation has no remaining hard deadline"
            )
        deadline = before + allowed
        input_bundle = _stage_input_bundle(
            stage=stage,
            outputs=outputs,
            predecessor_feature_rows=predecessor_feature_rows,
        )
        if self._test_only_allow_effects:
            raw_evaluation = self._deps.evaluator.evaluate(
                stage=stage,
                input_bundle=input_bundle,
            )
            evaluation = self._deps.evaluator.validate(
                raw_evaluation,
                stage=stage,
                input_bundle=input_bundle,
            )
        else:
            try:
                evaluation = self._executor.evaluate_deterministic(
                    stage=stage,
                    input_bundle=input_bundle,
                    deadline_monotonic=deadline,
                )
            except SecGemmaOnlineRiskOverlayRunnerError:
                raise
            except Exception as exc:
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    "Deterministic evaluation worker ended indeterminately"
                ) from exc
        evaluation = _validate_evaluation_envelope(
            evaluation,
            stage=stage,
            input_bundle=input_bundle,
        )
        gate = _validate_gate_binding(
            evaluation["gate_report"],
            stage=stage,
            require_pass=False,
        )
        current_feature_hashes = set(
            input_bundle["current_feature_row_sha256s"]
        )
        current_feature_rows = [
            row
            for row in input_bundle["feature_rows"]
            if row["feature_row_sha256"] in current_feature_hashes
        ]
        if len(current_feature_rows) != len(current_feature_hashes):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Current feature persistence selection changed"
            )
        after = self._read_clock("deterministic evaluation after")
        if after < before:
            raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                "Monotonic clock moved backwards during deterministic "
                "evaluation"
            )
        elapsed = after - before
        if (
            after >= attempt_deadline
            or after > deadline
            or deterministic_used + elapsed
            > MAX_DETERMINISTIC_SECONDS
        ):
            raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                "Deterministic replay, metrics, gates, or proofs exceeded "
                "the frozen hard deadline"
            )
        fixed_receipts = _extend_deterministic_phase_receipt(
            receipts,
            additional_elapsed=elapsed,
            deadline=deadline,
        )
        return (
            input_bundle,
            current_feature_rows,
            evaluation,
            gate,
            fixed_receipts,
        )

    def _execute_phases(
        self,
        *,
        command: str,
        attempt_id: str | None,
        capability: Any | None,
        attempt_start: float,
        attempt_deadline: float,
    ) -> tuple[
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
        Any | None,
        AcquisitionExecutionResult | None,
        Mapping[str, Any] | None,
    ]:
        start = attempt_start
        prior = start
        group_used = {name: 0.0 for name in _GROUP_CAPS}
        market_used = 0.0
        outputs: dict[str, dict[str, Any]] = {}
        receipts: list[dict[str, Any]] = []
        acquisition_execution: AcquisitionExecutionResult | None = None
        acquisition_report_authority: Any | None = None
        runtime_receipt: Mapping[str, Any] | None = None
        for phase in _PHASES[command]:
            before = self._read_clock(f"{phase} before")
            if before < prior:
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    "Monotonic clock moved backwards"
                )
            if before >= attempt_deadline:
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    "Attempt reached the strict total-runtime boundary"
                )
            group = _PHASE_GROUP[phase]
            remaining_group = _GROUP_CAPS[group] - group_used[group]
            if remaining_group < 0.0:
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    f"{group} budget was exhausted"
                )
            phase_remaining = remaining_group
            overall_remaining = attempt_deadline - before
            allowed = min(phase_remaining, overall_remaining)
            if allowed <= 0.0:
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    f"{phase} has no remaining external deadline"
                )
            deadline = before + allowed
            permit: PhasePermit | None = None
            if command in _EFFECTFUL_COMMANDS:
                if capability is None or attempt_id is None:
                    raise SecGemmaOnlineRiskOverlayRunnerError(
                        "Effectful phase lacks a consumed capability"
                    )
                effects = _PHASE_EFFECTS[phase]
                for effect in effects:
                    self._store.authorize_effect(capability, effect)
                permit = PhasePermit(
                    attempt_id=attempt_id,
                    command=command,
                    phase=phase,
                    effects=effects,
                    store=self._store,
                    capability=capability,
                    _sentinel=_PHASE_PERMIT_SENTINEL,
                )
            try:
                if phase == "acquisition":
                    if self._acquisition_adapter is None:
                        raise SecGemmaOnlineRiskOverlayRunnerError(
                            "Acquisition command lacks its exact adapter"
                        )
                    raw_execution = self._acquisition_adapter.acquire(
                        command=command,
                        store=self._store,
                        capability=capability,
                        deadline_monotonic=deadline,
                    )
                    (
                        payload,
                        acquisition_execution,
                        acquisition_report_authority,
                    ) = (
                        _normalize_acquisition_execution(
                            raw_execution,
                            command=command,
                            test_only=self._test_only_allow_effects,
                        )
                    )
                    accounting = payload["request_accounting"]
                    raw = build_phase_output(
                        command=command,
                        phase=phase,
                        counters={
                            "sec_request_count": accounting[
                                "sec_request_count"
                            ],
                            "market_request_count": accounting[
                                "market_request_count"
                            ],
                            "model_call_count": 0,
                            "retry_count": 0,
                            "fallback_count": 0,
                            "model_pull_count": 0,
                            "paid_api_call_count": 0,
                        },
                        payload=payload,
                    )
                else:
                    raw = self._executor.execute(
                        command=command,
                        phase=phase,
                        permit=permit,
                        deadline_monotonic=deadline,
                        prior_phase_outputs=copy.deepcopy(outputs),
                        acquisition_execution=acquisition_execution,
                    )
            except SecGemmaOnlineRiskOverlayRunnerError:
                raise
            except Exception as exc:
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    f"{phase} worker ended indeterminately"
                ) from exc
            output = validate_phase_output(
                raw, command=command, phase=phase
            )
            market_elapsed_seconds_hex: str | None = None
            if phase == "acquisition":
                market_elapsed_seconds_hex = output["payload"][
                    "request_accounting"
                ]["market_elapsed_seconds_hex"]
                market_elapsed = _nonnegative_float_hex(
                    market_elapsed_seconds_hex,
                    "acquisition market elapsed time",
                )
                market_used += market_elapsed
                if market_elapsed > MAX_MARKET_SECONDS:
                    raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                        "Acquisition exceeded its Yahoo market deadline"
                    )
            if phase == "runtime_identity":
                if runtime_receipt is not None:
                    raise SecGemmaOnlineRiskOverlayRunnerError(
                        "Runtime identity was verified more than once"
                    )
                runtime_receipt = _mapping(
                    self._deps.runtime_verifier(
                        _runtime_payload({phase: output})
                    ),
                    "runtime identity receipt",
                )
            after = self._read_clock(f"{phase} after")
            if after < before:
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    "Monotonic clock moved backwards inside a phase"
                )
            elapsed = after - before
            group_used[group] += elapsed
            if (
                after >= attempt_deadline
                or after > deadline
                or group_used[group] > _GROUP_CAPS[group]
                or market_used > MAX_MARKET_SECONDS
            ):
                raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                    f"{phase} exceeded its external parent deadline"
                )
            outputs[phase] = output
            receipts.append(
                _phase_receipt(
                    output=output,
                    elapsed=elapsed,
                    deadline=deadline,
                    market_elapsed_seconds_hex=(
                        market_elapsed_seconds_hex
                    ),
                )
            )
            prior = after
        return (
            outputs,
            receipts,
            acquisition_report_authority,
            acquisition_execution,
            runtime_receipt,
        )

    def run_local_preflight(self) -> dict[str, Any]:
        start = self._read_clock("preflight start")
        deadline = start + MAX_TOTAL_RUNTIME_SECONDS
        self._verify_live_implementation()
        plan = self.plan(LOCAL_PREFLIGHT)
        (
            outputs,
            receipts,
            _,
            _,
            runtime_receipt,
        ) = self._execute_phases(
            command=LOCAL_PREFLIGHT,
            attempt_id=None,
            capability=None,
            attempt_start=start,
            attempt_deadline=deadline,
        )
        end = self._read_clock("preflight end")
        budgets = _budget_report(
            command=LOCAL_PREFLIGHT,
            receipts=receipts,
            start=start,
            end=end,
        )
        if runtime_receipt is None:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Preflight did not verify the pinned runtime"
            )
        snapshot = _mapping(
            self._store.snapshot(), "preflight store snapshot"
        )
        body = {
            "schema_version": VERIFY_RESULT_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "command": LOCAL_PREFLIGHT,
            "runner_plan_sha256": plan["runner_plan_sha256"],
            "implementation_manifest_sha256": self._implementation[
                "implementation_manifest_sha256"
            ],
            "runtime_receipt_sha256": canonical_sha256(
                runtime_receipt
            ),
            "phase_budget_report_sha256": budgets[
                "phase_budget_report_sha256"
            ],
            "store_snapshot_sha256": canonical_sha256(snapshot),
            "effectful_production_ready": False,
            "production_effect_blockers": list(
                PRODUCTION_EFFECT_BLOCKERS
            ),
            "production_effect_blockers_sha256": canonical_sha256(
                PRODUCTION_EFFECT_BLOCKERS
            ),
            "passed": False,
        }
        return {**body, "verify_result_sha256": canonical_sha256(body)}

    def _publication_timeout(
        self, deadline_monotonic: float, location: str
    ) -> float:
        now = self._require_before_deadline(
            deadline_monotonic, location
        )
        remaining = deadline_monotonic - now
        if not math.isfinite(remaining) or remaining <= 0.0:
            raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                f"{location} has no publication time remaining"
            )
        command_seconds = (
            remaining - _PUBLICATION_QUIESCENCE_RESERVE_SECONDS
        )
        if command_seconds <= 0.0:
            raise SecGemmaOnlineRiskOverlayRunnerIndeterminate(
                f"{location} lacks the frozen publication quiescence reserve"
            )
        return min(command_seconds, 300.0)

    def _commit_worker_quiescence(
        self,
        *,
        capability: Any,
        worker: PublicationWorkerHandle,
        durable_ownership: Any,
    ) -> Any:
        ownership = _opaque_material(durable_ownership)
        quiescence = worker.quiesce(
            worker_ownership_sha256=ownership[
                "worker_ownership_sha256"
            ]
        )
        return self._store.commit_worker_quiescence(
            capability, quiescence
        )

    def _authority_validator(
        self, production_validator: Callable[[Any], Mapping[str, Any]]
    ) -> Callable[[Any], Mapping[str, Any]]:
        return (
            _opaque_material
            if self._test_only_allow_effects
            else production_validator
        )

    def _read_publication_remote(
        self,
        *,
        prepared: PreparedExternalPublication,
        transport: PreparedPublicationTransport,
        durable_transport: Any,
        durable_ownership: Any,
        publication_intent_sha256: str,
        capability: Any,
        ordinal: int,
        phase: str,
        prior_push_command_sha256: str,
        deadline_monotonic: float,
    ) -> Any:
        snapshot = _mapping(
            self._store.snapshot(),
            "pre-publication-readback store snapshot",
        )
        return self._publisher.read_remote(
            prepared_publication=prepared,
            transport=transport,
            durable_transport_manifest=durable_transport,
            transport_manifest_validator=self._authority_validator(
                transport_manifest_material
            ),
            store_instance_id=self._store.store_instance_id,
            store_session_nonce_sha256=(
                self._store.store_session_nonce_sha256
            ),
            publication_intent_sha256=publication_intent_sha256,
            operation_kind=capability.operation_kind,
            operation_sha256=capability.operation_sha256,
            worker_ownership=durable_ownership,
            worker_ownership_validator=self._authority_validator(
                worker_ownership_material
            ),
            observation_ordinal=ordinal,
            observation_phase=phase,
            prior_push_command_sha256=prior_push_command_sha256,
            pre_observation_store_journal_sequence=snapshot[
                "journal_entry_count"
            ],
            pre_observation_store_journal_tip_sha256=snapshot[
                "journal_tip_sha256"
            ],
            timeout_seconds=self._publication_timeout(
                deadline_monotonic,
                f"before publication {phase} readback",
            ),
        )

    def _run_normal_publication(
        self,
        *,
        attempt_id: str,
        publication_intent: Any,
        prepared: PreparedExternalPublication,
        deadline_monotonic: float,
    ) -> Any | None:
        intent = _mapping(
            self._authority_validator(publication_intent_material)(
                publication_intent
            ),
            "validated publication intent",
        )
        capability = self._store.issue_publication_recovery_capability(
            attempt_id,
            operation_kind=NORMAL_PUBLICATION,
            operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
        )
        worker: PublicationWorkerHandle | None = None
        durable_ownership: Any | None = None
        quiescent = False
        try:
            worker = self._publication_runtime.begin(
                implementation_manifest=self._implementation,
                store_instance_id=self._store.store_instance_id,
                store_session_nonce_sha256=(
                    self._store.store_session_nonce_sha256
                ),
                attempt_id=attempt_id,
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                operation_kind=capability.operation_kind,
                operation_sha256=capability.operation_sha256,
            )
            snapshot = _mapping(
                self._store.snapshot(),
                "pre-publication-transport store snapshot",
            )
            transport = prepare_isolated_publication_transport(
                transport_root=worker.transport_root,
                source_object_directory=(
                    worker.source_object_directory
                ),
                prepared_publication=prepared,
                executable_pins=worker.executable_pins,
                host_environment_values=worker.host_environment_values,
                identity_verifier=worker.verify_executable_pins,
                store_instance_id=self._store.store_instance_id,
                store_session_nonce_sha256=(
                    self._store.store_session_nonce_sha256
                ),
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                operation_kind=capability.operation_kind,
                operation_sha256=capability.operation_sha256,
                pre_transport_store_journal_sequence=snapshot[
                    "journal_entry_count"
                ],
                pre_transport_store_journal_tip_sha256=snapshot[
                    "journal_tip_sha256"
                ],
            )
            durable_transport = self._store.commit_transport_manifest(
                capability, transport.manifest
            )
            durable_ownership = (
                self._store.claim_publication_worker_ownership(
                    capability, worker.ownership_material
                )
            )
            first = self._read_publication_remote(
                prepared=prepared,
                transport=transport,
                durable_transport=durable_transport,
                durable_ownership=durable_ownership,
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                capability=capability,
                ordinal=1,
                phase=PRE_PUSH,
                prior_push_command_sha256=(
                    PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
                ),
                deadline_monotonic=deadline_monotonic,
            )
            if first.observation is None:
                self._commit_worker_quiescence(
                    capability=capability,
                    worker=worker,
                    durable_ownership=durable_ownership,
                )
                quiescent = True
                self._store.release_publication_capability(capability)
                return None
            durable_first = self._store.commit_remote_observation(
                capability, first.evidence, first.observation
            )
            if first.observed_ref_state == "conflicting":
                conflict = _build_publication_conflict(
                    implementation_manifest=self._implementation,
                    store=self._store,
                    attempt_id=attempt_id,
                    publication_intent_sha256=intent[
                        "publication_intent_sha256"
                    ],
                    capability=capability,
                    worker_ownership=durable_ownership,
                    remote_observation=durable_first,
                )
                self._store.commit_publication_conflict(
                    capability, conflict
                )
                self._commit_worker_quiescence(
                    capability=capability,
                    worker=worker,
                    durable_ownership=durable_ownership,
                )
                quiescent = True
                self._store.release_publication_capability(capability)
                return None
            authorization: Any | None = None
            durable_exact = durable_first
            if first.observed_ref_state == "absent":
                authorization_material = _build_pre_push_authorization(
                    implementation_manifest=self._implementation,
                    store=self._store,
                    attempt_id=attempt_id,
                    publication_intent_sha256=intent[
                        "publication_intent_sha256"
                    ],
                    capability=capability,
                    worker_ownership=durable_ownership,
                    remote_observation=durable_first,
                )
                authorization = (
                    self._store.commit_pre_push_authorization(
                        capability, authorization_material
                    )
                )
                push = self._publisher.push_once(
                    prepared_publication=prepared,
                    transport=transport,
                    pre_push_authorization=authorization,
                    authorization_validator=self._authority_validator(
                        pre_push_authorization_material
                    ),
                    absent_remote_observation=durable_first,
                    observation_validator=self._authority_validator(
                        remote_observation_material
                    ),
                    timeout_seconds=self._publication_timeout(
                        deadline_monotonic,
                        "before publication push",
                    ),
                )
                second = self._read_publication_remote(
                    prepared=prepared,
                    transport=transport,
                    durable_transport=durable_transport,
                    durable_ownership=durable_ownership,
                    publication_intent_sha256=intent[
                        "publication_intent_sha256"
                    ],
                    capability=capability,
                    ordinal=2,
                    phase=POST_PUSH,
                    prior_push_command_sha256=(
                        push.push_command_sha256
                    ),
                    deadline_monotonic=deadline_monotonic,
                )
                if second.observation is None:
                    self._commit_worker_quiescence(
                        capability=capability,
                        worker=worker,
                        durable_ownership=durable_ownership,
                    )
                    quiescent = True
                    self._store.release_publication_capability(
                        capability
                    )
                    return None
                durable_second = self._store.commit_remote_observation(
                    capability, second.evidence, second.observation
                )
                if second.observed_ref_state == "conflicting":
                    conflict = _build_publication_conflict(
                        implementation_manifest=self._implementation,
                        store=self._store,
                        attempt_id=attempt_id,
                        publication_intent_sha256=intent[
                            "publication_intent_sha256"
                        ],
                        capability=capability,
                        worker_ownership=durable_ownership,
                        remote_observation=durable_second,
                    )
                    self._store.commit_publication_conflict(
                        capability, conflict
                    )
                    self._commit_worker_quiescence(
                        capability=capability,
                        worker=worker,
                        durable_ownership=durable_ownership,
                    )
                    quiescent = True
                    self._store.release_publication_capability(
                        capability
                    )
                    return None
                if second.observed_ref_state != "exact_expected":
                    self._commit_worker_quiescence(
                        capability=capability,
                        worker=worker,
                        durable_ownership=durable_ownership,
                    )
                    quiescent = True
                    self._store.release_publication_capability(
                        capability
                    )
                    return None
                durable_exact = durable_second
            elif first.observed_ref_state != "exact_expected":
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Publication readback state changed"
                )
            self._commit_worker_quiescence(
                capability=capability,
                worker=worker,
                durable_ownership=durable_ownership,
            )
            quiescent = True
            external_publication = (
                issue_verified_external_publication_from_observation(
                    prepared_publication=prepared,
                    durable_remote_observation=durable_exact,
                    observation_validator=self._authority_validator(
                        remote_observation_material
                    ),
                )
            )
            receipt_material = _build_publication_receipt(
                implementation_manifest=self._implementation,
                store=self._store,
                attempt_id=attempt_id,
                publication_intent=publication_intent,
                remote_observation=durable_exact,
                pre_push_authorization=authorization,
                external_publication=external_publication,
            )
            return self._store.commit_publication_receipt(
                capability,
                receipt_material,
                external_publication,
            )
        except Exception:
            if (
                worker is not None
                and durable_ownership is not None
                and not quiescent
            ):
                try:
                    self._commit_worker_quiescence(
                        capability=capability,
                        worker=worker,
                        durable_ownership=durable_ownership,
                    )
                    quiescent = True
                except Exception:
                    pass
            elif worker is not None and durable_ownership is None:
                try:
                    worker.abort_uncommitted()
                except Exception:
                    pass
            if quiescent or durable_ownership is None:
                try:
                    self._store.release_publication_capability(
                        capability
                    )
                except Exception:
                    pass
            return None

    def _run_recovery_publication(
        self,
        *,
        attempt_id: str,
        publication_intent: Any,
        prepared: PreparedExternalPublication,
        invocation_started_at: float,
        deadline_monotonic: float,
    ) -> Any | None:
        intent = _mapping(
            self._authority_validator(publication_intent_material)(
                publication_intent
            ),
            "validated recovery publication intent",
        )
        start_material = _build_publication_recovery_start(
            implementation_manifest=self._implementation,
            store=self._store,
            attempt_id=attempt_id,
            publication_intent=publication_intent,
        )
        capability = self._store.issue_publication_recovery_capability(
            attempt_id,
            operation_kind=PUBLICATION_RECOVERY,
            operation_sha256=start_material[
                "recovery_invocation_start_sha256"
            ],
        )
        recovery_start = self._store.commit_recovery_start(
            capability, start_material
        )
        start = _opaque_material(recovery_start)
        worker: PublicationWorkerHandle | None = None
        durable_ownership: Any | None = None
        quiescent = False
        completion: Any | None = None
        last_observation: Any | None = None
        last_phase: str | None = None
        last_state: str | None = None
        authorization: Any | None = None
        push_attempted = False
        conflict_committed = False

        def make_quiescent() -> None:
            nonlocal quiescent
            if quiescent:
                return
            if worker is None:
                quiescent = True
                return
            if durable_ownership is None:
                worker.abort_uncommitted()
                quiescent = True
                return
            self._commit_worker_quiescence(
                capability=capability,
                worker=worker,
                durable_ownership=durable_ownership,
            )
            quiescent = True

        def finish_recovery(
            *,
            outcome: str,
            observation: Any | None,
            push_authorization: Any | None,
            push_count: int,
        ) -> Any:
            nonlocal completion
            make_quiescent()
            now = self._require_before_deadline(
                deadline_monotonic,
                "before publication recovery completion",
            )
            elapsed = now - invocation_started_at
            material = _build_publication_recovery_completion(
                implementation_manifest=self._implementation,
                store=self._store,
                attempt_id=attempt_id,
                publication_intent=publication_intent,
                recovery_start=recovery_start,
                outcome=outcome,
                remote_observation=observation,
                pre_push_authorization=push_authorization,
                push_command_count_upper_bound=push_count,
                elapsed_seconds=elapsed,
            )
            completion = self._store.commit_recovery_completion(
                capability, material
            )
            return completion

        try:
            self._require_before_deadline(
                deadline_monotonic,
                "before publication recovery worker",
            )
            worker = self._publication_runtime.begin(
                implementation_manifest=self._implementation,
                store_instance_id=self._store.store_instance_id,
                store_session_nonce_sha256=(
                    self._store.store_session_nonce_sha256
                ),
                attempt_id=attempt_id,
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                operation_kind=capability.operation_kind,
                operation_sha256=capability.operation_sha256,
            )
            snapshot = _mapping(
                self._store.snapshot(),
                "pre-recovery-transport store snapshot",
            )
            transport = prepare_isolated_publication_transport(
                transport_root=worker.transport_root,
                source_object_directory=worker.source_object_directory,
                prepared_publication=prepared,
                executable_pins=worker.executable_pins,
                host_environment_values=worker.host_environment_values,
                identity_verifier=worker.verify_executable_pins,
                store_instance_id=self._store.store_instance_id,
                store_session_nonce_sha256=(
                    self._store.store_session_nonce_sha256
                ),
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                operation_kind=capability.operation_kind,
                operation_sha256=capability.operation_sha256,
                pre_transport_store_journal_sequence=snapshot[
                    "journal_entry_count"
                ],
                pre_transport_store_journal_tip_sha256=snapshot[
                    "journal_tip_sha256"
                ],
            )
            durable_transport = self._store.commit_transport_manifest(
                capability, transport.manifest
            )
            durable_ownership = (
                self._store.claim_publication_worker_ownership(
                    capability, worker.ownership_material
                )
            )
            first = self._read_publication_remote(
                prepared=prepared,
                transport=transport,
                durable_transport=durable_transport,
                durable_ownership=durable_ownership,
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                capability=capability,
                ordinal=1,
                phase=PRE_PUSH,
                prior_push_command_sha256=(
                    PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
                ),
                deadline_monotonic=deadline_monotonic,
            )
            if first.observation is None:
                finish_recovery(
                    outcome="remote_unavailable_before_observation",
                    observation=None,
                    push_authorization=None,
                    push_count=0,
                )
                return None
            last_observation = self._store.commit_remote_observation(
                capability, first.evidence, first.observation
            )
            last_phase = PRE_PUSH
            last_state = first.observed_ref_state
            if last_state == "conflicting":
                conflict = _build_publication_conflict(
                    implementation_manifest=self._implementation,
                    store=self._store,
                    attempt_id=attempt_id,
                    publication_intent_sha256=intent[
                        "publication_intent_sha256"
                    ],
                    capability=capability,
                    worker_ownership=durable_ownership,
                    remote_observation=last_observation,
                )
                self._store.commit_publication_conflict(
                    capability, conflict
                )
                conflict_committed = True
                finish_recovery(
                    outcome=(
                        "remote_conflict_poisoned_without_push"
                    ),
                    observation=last_observation,
                    push_authorization=None,
                    push_count=0,
                )
                return None
            if last_state == "exact_expected":
                recovery_completion = finish_recovery(
                    outcome="remote_exact_without_push",
                    observation=last_observation,
                    push_authorization=None,
                    push_count=0,
                )
                external_publication = (
                    issue_verified_external_publication_from_observation(
                        prepared_publication=prepared,
                        durable_remote_observation=last_observation,
                        observation_validator=self._authority_validator(
                            remote_observation_material
                        ),
                    )
                )
                receipt_material = _build_publication_receipt(
                    implementation_manifest=self._implementation,
                    store=self._store,
                    attempt_id=attempt_id,
                    publication_intent=publication_intent,
                    remote_observation=last_observation,
                    pre_push_authorization=None,
                    external_publication=external_publication,
                    recovery_completion=recovery_completion,
                )
                return self._store.commit_publication_receipt(
                    capability,
                    receipt_material,
                    external_publication,
                )
            if last_state != "absent":
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Recovery pre-push readback state changed"
                )
            authorization_material = _build_pre_push_authorization(
                implementation_manifest=self._implementation,
                store=self._store,
                attempt_id=attempt_id,
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                capability=capability,
                worker_ownership=durable_ownership,
                remote_observation=last_observation,
                operation_ordinal=start["invocation_ordinal"],
            )
            authorization = self._store.commit_pre_push_authorization(
                capability, authorization_material
            )
            push_attempted = True
            push = self._publisher.push_once(
                prepared_publication=prepared,
                transport=transport,
                pre_push_authorization=authorization,
                authorization_validator=self._authority_validator(
                    pre_push_authorization_material
                ),
                absent_remote_observation=last_observation,
                observation_validator=self._authority_validator(
                    remote_observation_material
                ),
                timeout_seconds=self._publication_timeout(
                    deadline_monotonic,
                    "before publication recovery push",
                ),
            )
            second = self._read_publication_remote(
                prepared=prepared,
                transport=transport,
                durable_transport=durable_transport,
                durable_ownership=durable_ownership,
                publication_intent_sha256=intent[
                    "publication_intent_sha256"
                ],
                capability=capability,
                ordinal=2,
                phase=POST_PUSH,
                prior_push_command_sha256=push.push_command_sha256,
                deadline_monotonic=deadline_monotonic,
            )
            if second.observation is None:
                finish_recovery(
                    outcome="push_issued_unconfirmed",
                    observation=last_observation,
                    push_authorization=authorization,
                    push_count=1,
                )
                return None
            last_observation = self._store.commit_remote_observation(
                capability, second.evidence, second.observation
            )
            last_phase = POST_PUSH
            last_state = second.observed_ref_state
            if last_state == "conflicting":
                conflict = _build_publication_conflict(
                    implementation_manifest=self._implementation,
                    store=self._store,
                    attempt_id=attempt_id,
                    publication_intent_sha256=intent[
                        "publication_intent_sha256"
                    ],
                    capability=capability,
                    worker_ownership=durable_ownership,
                    remote_observation=last_observation,
                )
                self._store.commit_publication_conflict(
                    capability, conflict
                )
                conflict_committed = True
                finish_recovery(
                    outcome="post_push_conflict_poisoned",
                    observation=last_observation,
                    push_authorization=authorization,
                    push_count=1,
                )
                return None
            if last_state == "absent":
                finish_recovery(
                    outcome="post_push_ref_absent",
                    observation=last_observation,
                    push_authorization=authorization,
                    push_count=1,
                )
                return None
            if last_state != "exact_expected":
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Recovery post-push readback state changed"
                )
            recovery_completion = finish_recovery(
                outcome="published_exact_after_push",
                observation=last_observation,
                push_authorization=authorization,
                push_count=1,
            )
            external_publication = (
                issue_verified_external_publication_from_observation(
                    prepared_publication=prepared,
                    durable_remote_observation=last_observation,
                    observation_validator=self._authority_validator(
                        remote_observation_material
                    ),
                )
            )
            receipt_material = _build_publication_receipt(
                implementation_manifest=self._implementation,
                store=self._store,
                attempt_id=attempt_id,
                publication_intent=publication_intent,
                remote_observation=last_observation,
                pre_push_authorization=authorization,
                external_publication=external_publication,
                recovery_completion=recovery_completion,
            )
            return self._store.commit_publication_receipt(
                capability,
                receipt_material,
                external_publication,
            )
        except Exception:
            try:
                make_quiescent()
            except Exception:
                return None
            if completion is None:
                try:
                    if conflict_committed:
                        outcome = (
                            "post_push_conflict_poisoned"
                            if last_phase == POST_PUSH
                            else "remote_conflict_poisoned_without_push"
                        )
                        observation = last_observation
                        push_count = 1 if last_phase == POST_PUSH else 0
                    elif (
                        last_state == "exact_expected"
                        and last_phase == POST_PUSH
                    ):
                        outcome = "published_exact_after_push"
                        observation = last_observation
                        push_count = 1
                    elif last_state == "exact_expected":
                        outcome = "remote_exact_without_push"
                        observation = last_observation
                        push_count = 0
                    elif last_phase == POST_PUSH and last_state == "absent":
                        outcome = "post_push_ref_absent"
                        observation = last_observation
                        push_count = 1
                    elif last_state == "absent" and authorization is None:
                        outcome = (
                            "remote_absent_authorization_not_committed"
                        )
                        observation = last_observation
                        push_count = 0
                    elif authorization is not None and push_attempted:
                        outcome = "push_issued_unconfirmed"
                        observation = last_observation
                        push_count = 1
                    elif authorization is not None:
                        outcome = "authorized_push_not_issued"
                        observation = last_observation
                        push_count = 0
                    else:
                        outcome = "interrupted_before_completion"
                        observation = None
                        push_count = 0
                    finish_recovery(
                        outcome=outcome,
                        observation=observation,
                        push_authorization=authorization,
                        push_count=push_count,
                    )
                except Exception:
                    pass
            if quiescent:
                try:
                    self._store.release_publication_capability(
                        capability
                    )
                except Exception:
                    pass
            return None

    def _reconcile_local_publication_receipt(
        self,
        *,
        attempt_id: str,
        publication_intent: Any,
        prepared: PreparedExternalPublication,
    ) -> Any | None:
        receipt_rows = self._store.governance_records(
            "publication_receipts", attempt_id=attempt_id
        )
        if len(receipt_rows) > 1:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Publication recovery found multiple durable receipts"
            )
        if receipt_rows:
            material = receipt_rows[0]
            observation = self._store.remote_observation_authority(
                attempt_id,
                material["publication_remote_observation_sha256"],
            )
            external_publication = (
                issue_verified_external_publication_from_observation(
                    prepared_publication=prepared,
                    durable_remote_observation=observation,
                    observation_validator=self._authority_validator(
                        remote_observation_material
                    ),
                )
            )
            return self._store.publication_receipt_authority(
                attempt_id, external_publication
            )

        conflicts = self._store.governance_records(
            "publication_conflicts", attempt_id=attempt_id
        )
        if conflicts:
            if len(conflicts) != 1:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Publication recovery found multiple conflict poisons"
                )
            return None

        observations = self._store.governance_records(
            "publication_remote_observations", attempt_id=attempt_id
        )
        normal_conflicts = [
            row
            for row in observations
            if row["observation_operation_kind"] == NORMAL_PUBLICATION
            and row["observed_ref_state"] == "conflicting"
        ]
        if normal_conflicts:
            if len(normal_conflicts) != 1:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Normal publication has multiple terminal conflict observations"
                )
            if self._store.governance_records(
                "publication_recovery_invocation_starts",
                attempt_id=attempt_id,
            ):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Recovery action followed an authoritative normal conflict"
                )
            observation_material = normal_conflicts[0]
            observation = self._store.remote_observation_authority(
                attempt_id,
                observation_material[
                    "publication_remote_observation_sha256"
                ],
            )
            ownership = (
                self._store.publication_worker_ownership_authority(
                    attempt_id,
                    observation_material["worker_ownership_sha256"],
                )
            )
            capability = (
                self._store.issue_publication_recovery_capability(
                    attempt_id,
                    operation_kind=NORMAL_PUBLICATION,
                    operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
                )
            )
            conflict = _build_publication_conflict(
                implementation_manifest=self._implementation,
                store=self._store,
                attempt_id=attempt_id,
                publication_intent_sha256=_opaque_material(
                    publication_intent
                )["publication_intent_sha256"],
                capability=capability,
                worker_ownership=ownership,
                remote_observation=observation,
            )
            self._store.commit_publication_conflict(
                capability, conflict
            )
            self._store.release_publication_capability(capability)
            return None
        normal_exact = [
            row
            for row in observations
            if row["observation_operation_kind"] == NORMAL_PUBLICATION
            and row["observed_ref_state"] == "exact_expected"
        ]
        if normal_exact:
            if len(normal_exact) != 1:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Normal publication has multiple exact terminal observations"
                )
            observation_material = normal_exact[0]
            observation = self._store.remote_observation_authority(
                attempt_id,
                observation_material[
                    "publication_remote_observation_sha256"
                ],
            )
            authorizations = [
                row
                for row in self._store.governance_records(
                    "publication_pre_push_authorizations",
                    attempt_id=attempt_id,
                )
                if row["authorization_operation_kind"]
                == NORMAL_PUBLICATION
            ]
            if len(authorizations) > 1:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Normal publication has multiple push authorizations"
                )
            authorization = (
                None
                if not authorizations
                else self._store.pre_push_authorization_authority(
                    attempt_id,
                    authorizations[0][
                        "pre_push_authorization_sha256"
                    ],
                )
            )
            capability = (
                self._store.issue_publication_recovery_capability(
                    attempt_id,
                    operation_kind=NORMAL_PUBLICATION,
                    operation_sha256=(
                        PUBLICATION_NORMAL_OPERATION_SHA256
                    ),
                )
            )
            external_publication = (
                issue_verified_external_publication_from_observation(
                    prepared_publication=prepared,
                    durable_remote_observation=observation,
                    observation_validator=self._authority_validator(
                        remote_observation_material
                    ),
                )
            )
            receipt_material = _build_publication_receipt(
                implementation_manifest=self._implementation,
                store=self._store,
                attempt_id=attempt_id,
                publication_intent=publication_intent,
                remote_observation=observation,
                pre_push_authorization=authorization,
                external_publication=external_publication,
            )
            return self._store.commit_publication_receipt(
                capability,
                receipt_material,
                external_publication,
            )

        completions = self._store.governance_records(
            "publication_recovery_invocation_completions",
            attempt_id=attempt_id,
        )
        eligible = [
            row
            for row in completions
            if row["outcome"]
            in {
                "remote_exact_without_push",
                "published_exact_after_push",
            }
        ]
        if not eligible:
            return None
        if len(eligible) != 1 or eligible[-1] != completions[-1]:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Publication recovery receipt eligibility is ambiguous"
            )
        completion_material = eligible[0]
        completion = self._store.recovery_completion_authority(
            attempt_id,
            completion_material[
                "recovery_invocation_completion_sha256"
            ],
        )
        observation = self._store.remote_observation_authority(
            attempt_id,
            completion_material["remote_observation_sha256"],
        )
        authorization_hash = completion_material[
            "pre_push_authorization_sha256"
        ]
        authorization = (
            None
            if authorization_hash
            == PUBLICATION_RECOVERY_NO_PRE_PUSH_AUTHORIZATION_SHA256
            else self._store.pre_push_authorization_authority(
                attempt_id, authorization_hash
            )
        )
        capability = self._store.issue_publication_recovery_capability(
            attempt_id,
            operation_kind=PUBLICATION_RECOVERY,
            operation_sha256=completion_material[
                "recovery_invocation_start_sha256"
            ],
        )
        external_publication = (
            issue_verified_external_publication_from_observation(
                prepared_publication=prepared,
                durable_remote_observation=observation,
                observation_validator=self._authority_validator(
                    remote_observation_material
                ),
            )
        )
        receipt_material = _build_publication_receipt(
            implementation_manifest=self._implementation,
            store=self._store,
            attempt_id=attempt_id,
            publication_intent=publication_intent,
            remote_observation=observation,
            pre_push_authorization=authorization,
            external_publication=external_publication,
            recovery_completion=completion,
        )
        return self._store.commit_publication_receipt(
            capability,
            receipt_material,
            external_publication,
        )

    def _rehydrate_terminal_result(
        self,
        *,
        command: str,
        attempt_id: str,
        terminal_transition: Mapping[str, Any],
    ) -> dict[str, Any]:
        status = terminal_transition.get("status")
        if status not in {TERMINAL_PASS, TERMINAL_FAIL}:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Only a semantic terminal result can be rehydrated"
            )
        binding = _mapping(
            self._store.terminal_anchor_binding(attempt_id),
            "rehydrated terminal anchor binding",
        )
        artifact, artifact_receipt = (
            self._store.terminal_artifact_payload_and_receipt(
                attempt_id
            )
        )
        terminal_evidence = _mapping(
            binding["terminal_evidence"],
            "rehydrated terminal evidence",
        )
        publication = validate_external_publication(
            binding["external_publication"],
            implementation_manifest=self._implementation,
        )
        if status == TERMINAL_PASS:
            return _success_result(
                command=command,
                attempt_id=attempt_id,
                terminal_transition=terminal_transition,
                terminal_evidence=terminal_evidence,
                external_publication=publication,
                terminal_artifact=artifact,
                terminal_artifact_receipt=artifact_receipt,
            )
        if command not in _SCORING_COMMANDS:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Acquisition cannot rehydrate a semantic terminal failure"
            )
        return _failure_result(
            command=command,
            attempt_id=attempt_id,
            terminal_status=TERMINAL_FAIL,
            diagnostic_code="gate_failure_sealed",
            terminal_transition=terminal_transition,
            terminal_evidence=terminal_evidence,
            external_publication=publication,
            terminal_artifact=artifact,
            terminal_artifact_receipt=artifact_receipt,
        )

    def _pending_acquisition_report_authority(
        self, attempt_id: str
    ) -> Any:
        self._store.pending_acquisition_phase_evidence(attempt_id)
        if self._test_only_allow_effects:
            adapter_method = getattr(
                self._acquisition_adapter,
                "rehydrate_pending_acquisition_report",
                None,
            )
            if not callable(adapter_method):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Test recovery lacks an opaque pending acquisition report"
                )
            report = adapter_method(attempt_id=attempt_id)
        else:
            authority_method = getattr(
                self._production_authorities,
                "rehydrate_pending_acquisition_report",
                None,
            )
            if not callable(authority_method):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Production recovery lacks pending acquisition rehydration"
                )
            report = authority_method(attempt_id=attempt_id)
        if not is_verified_acquisition_report(report):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Pending acquisition recovery returned foreign authority"
            )
        return report

    def recover_publication(
        self,
        attempt_id: str,
        *,
        supervised_worker_authority: (
            SupervisedPublicationRecoveryWorkerAuthority | None
        ) = None,
    ) -> dict[str, Any]:
        """Enter one bounded recovery worker or the explicit test-only path.

        A production caller cannot execute recovery silently in its own
        process.  The public production supervisor must start a fresh isolated
        worker and supply the opaque, source-bound worker authority minted for
        that child.  Tests retain a direct path so deterministic recovery logic
        can be exercised without spawning a process.
        """
        worker_entry_started_at = self._read_clock(
            "publication recovery invocation entry"
        )
        local_deadline = (
            worker_entry_started_at + MAX_PUBLICATION_RECOVERY_SECONDS
        )
        self._require_before_deadline(
            local_deadline,
            "at publication recovery invocation entry",
        )
        if type(attempt_id) is not str or attempt_id not in _ATTEMPT_ID_TO_COMMAND:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Publication recovery attempt is not preregistered"
            )
        if self._test_only_allow_effects:
            if supervised_worker_authority is not None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Test publication recovery cannot claim a production "
                    "supervised-worker authority"
                )
            recovery_started_at = worker_entry_started_at
            invocation_deadline = local_deadline
        else:
            recovery_started_at, invocation_deadline = (
                self._authorize_supervised_publication_recovery_worker(
                    authority=supervised_worker_authority,
                    attempt_id=attempt_id,
                    invocation_started_at=worker_entry_started_at,
                    local_deadline=local_deadline,
                )
            )
        return self._recover_publication_in_supervised_worker(
            attempt_id,
            recovery_started_at=recovery_started_at,
            invocation_deadline=invocation_deadline,
        )

    def _recover_publication_in_supervised_worker(
        self,
        attempt_id: str,
        *,
        recovery_started_at: float,
        invocation_deadline: float,
    ) -> dict[str, Any]:
        """Run the frozen recovery body after the public isolation gate."""

        command = _ATTEMPT_ID_TO_COMMAND[attempt_id]
        self._verify_live_implementation()
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery implementation verification",
        )
        history = self._store.attempt_history(attempt_id)
        attempt_plan = validate_attempt_plan(
            self._store.attempt_plan(attempt_id),
            implementation_manifest=self._implementation,
        )
        validated_history = validate_attempt_history(
            attempt_plan=attempt_plan,
            implementation_manifest=self._implementation,
            transitions=history,
        )
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery attempt validation",
        )
        if validated_history[-1]["status"] in {
            TERMINAL_PASS,
            TERMINAL_FAIL,
        }:
            result = self._rehydrate_terminal_result(
                command=command,
                attempt_id=attempt_id,
                terminal_transition=validated_history[-1],
            )
            self._require_before_deadline(
                invocation_deadline,
                "after terminal publication recovery rehydration",
            )
            return result
        if (
            len(validated_history) != 2
            or validated_history[-1]["status"] != CONSUMED
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Publication recovery requires one consumed pending attempt"
            )
        if self._test_only_allow_effects:
            if self._production_authorities is not None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Test recovery cannot claim production authority"
                )
        else:
            self._verify_production_recovery_authorities()
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery authority verification",
        )

        self._store.reconcile_publication_recovery_state(attempt_id)
        self._require_before_deadline(
            invocation_deadline,
            "after local publication recovery reconciliation",
        )
        reconstruction = (
            self._store.terminal_reconstruction_authority(attempt_id)
        )
        reconstruction_material = _opaque_material(reconstruction)
        publication_intent = self._store.publication_intent_authority(
            attempt_id
        )
        intent = _mapping(
            self._authority_validator(publication_intent_material)(
                publication_intent
            ),
            "validated pending publication intent",
        )
        if (
            reconstruction_material["attempt_plan_sha256"]
            != attempt_plan["attempt_plan_sha256"]
            or intent["attempt_plan_sha256"]
            != attempt_plan["attempt_plan_sha256"]
            or intent["terminal_reconstruction_material_sha256"]
            != reconstruction_material[
                "terminal_reconstruction_material_sha256"
            ]
        ):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Pending publication crossed its frozen attempt reconstruction"
            )
        prepared = reconstruct_prepared_external_publication(
            implementation_manifest=self._implementation,
            publication_intent=publication_intent,
            intent_validator=self._authority_validator(
                publication_intent_material
            ),
        )
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery reconstruction",
        )
        publication_receipt = self._reconcile_local_publication_receipt(
            attempt_id=attempt_id,
            publication_intent=publication_intent,
            prepared=prepared,
        )
        self._require_before_deadline(
            invocation_deadline,
            "after local publication receipt reconciliation",
        )
        conflicts = self._store.governance_records(
            "publication_conflicts", attempt_id=attempt_id
        )
        if publication_receipt is None and not conflicts:
            publication_receipt = self._run_recovery_publication(
                attempt_id=attempt_id,
                publication_intent=publication_intent,
                prepared=prepared,
                invocation_started_at=recovery_started_at,
                deadline_monotonic=invocation_deadline,
            )
            self._require_before_deadline(
                invocation_deadline,
                "after supervised publication recovery",
            )
            self._store.reconcile_publication_recovery_state(attempt_id)
            self._require_before_deadline(
                invocation_deadline,
                "after same-session recovery reconciliation",
            )
            publication_receipt = (
                self._reconcile_local_publication_receipt(
                    attempt_id=attempt_id,
                    publication_intent=publication_intent,
                    prepared=prepared,
                )
            )
            self._require_before_deadline(
                invocation_deadline,
                "after same-session publication receipt reconciliation",
            )
            conflicts = self._store.governance_records(
                "publication_conflicts", attempt_id=attempt_id
            )
        if publication_receipt is None:
            self._require_before_deadline(
                invocation_deadline,
                "before publication recovery pending handoff",
            )
            return _publication_pending_result(
                command=command,
                attempt_id=attempt_id,
                publication_intent=publication_intent,
            )
        publication = publication_receipt.external_publication
        if not is_verified_external_publication(publication):
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Recovered receipt lost its opaque external publication"
            )
        artifact, artifact_receipt = (
            self._store.terminal_artifact_payload_and_receipt(
                attempt_id
            )
        )
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery artifact rehydration",
        )
        report_material = {
            "attempt_id": attempt_id,
            "attempt_plan_sha256": attempt_plan[
                "attempt_plan_sha256"
            ],
            "record_counts": copy.deepcopy(
                reconstruction_material["record_counts"]
            ),
            "record_commitment_sha256": reconstruction_material[
                "record_commitment_sha256"
            ],
        }
        terminal_status = reconstruction_material["terminal_status"]
        if attempt_plan["attempt_kind"] == DEVELOPMENT_ACQUISITION:
            acquisition_report = (
                self._pending_acquisition_report_authority(attempt_id)
            )
            terminal_evidence = (
                self._deps.issue_acquisition_terminal_evidence_fn(
                    implementation_manifest=self._implementation,
                    attempt_plan=attempt_plan,
                    acquisition_report=acquisition_report,
                    report_material=report_material,
                    acquisition_artifact_receipt=artifact_receipt,
                    publication_intent=publication_intent,
                    publication_receipt=publication_receipt,
                )
            )
        else:
            terminal_evidence = (
                self._deps.issue_scored_terminal_evidence_fn(
                    implementation_manifest=self._implementation,
                    attempt_plan=attempt_plan,
                    joint_stage_report=artifact,
                    report_material=report_material,
                    joint_artifact_receipt=artifact_receipt,
                    gate_checks=reconstruction_material[
                        "gate_checks"
                    ],
                    publication_intent=publication_intent,
                    publication_receipt=publication_receipt,
                )
            )
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery terminal evidence",
        )
        claims = self._store.governance_records(
            "terminalization_claims", attempt_id=attempt_id
        )
        if len(claims) > 1:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Publication recovery found multiple terminalization claims"
            )
        if claims:
            claim = self._store.terminalization_claim_authority(
                attempt_id
            )
        else:
            terminalization = (
                self._store.issue_terminalization_capability(attempt_id)
            )
            claim = self._store.commit_terminalization_claim(
                terminalization,
                terminal_evidence,
                terminal_status,
            )
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery terminalization claim",
        )
        self._store.complete_terminalized_attempt(
            claim, terminal_evidence
        )
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery terminal completion",
        )
        terminal = self._store.attempt_history(attempt_id)[-1]
        result = self._rehydrate_terminal_result(
            command=command,
            attempt_id=attempt_id,
            terminal_transition=terminal,
        )
        self._require_before_deadline(
            invocation_deadline,
            "after publication recovery sealed result",
        )
        return result

    def run(
        self,
        command: str,
    ) -> dict[str, Any]:
        if command == LOCAL_PREFLIGHT:
            return self.run_local_preflight()
        if command == VERIFY_COMMAND:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Use verify() with one sealed stage result"
            )
        if command not in _EFFECTFUL_COMMANDS:
            raise SecGemmaOnlineRiskOverlayRunnerError(
                "Unknown effectful runner command"
            )
        attempt_start = self._read_clock("parent attempt start")
        attempt_deadline = attempt_start + MAX_TOTAL_RUNTIME_SECONDS
        if self._test_only_allow_effects:
            if self._production_authorities is not None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Test-only execution cannot claim production authority"
                )
        else:
            self._verify_production_authorities(command)
        self._verify_live_implementation()
        predecessor_feature_rows = (
            self._load_predecessor_feature_rows(command)
        )
        final_authorization: (
            VerifiedFinalRegistryAuthorization | None
        ) = None
        if command == FINAL_COMMAND:
            if self._final_registry_authority is None:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Final execution lacks its externally pinned registry authority"
                )
            final_authorization = (
                self._final_registry_authority.authorize_final(
                    implementation_manifest=self._implementation,
                    deadline_monotonic=attempt_deadline,
                )
            )
            if not is_verified_final_registry_authorization(
                final_authorization
            ):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Final registry authority returned no opaque authorization"
                )
        self._require_before_deadline(
            attempt_deadline,
            "before attempt planning",
        )
        plan = self.plan(
            command,
            final_registry_authorization=final_authorization,
        )
        plan = validate_runner_plan(
            plan, implementation_manifest=self._implementation
        )
        attempt_plan = plan["attempt_plan"]
        assert attempt_plan is not None
        attempt_id = attempt_plan["attempt_id"]
        predecessor_publication_sha256 = (
            self._predecessor_publication_sha256(
                command,
                final_registry_authorization=final_authorization,
            )
        )
        self._verify_live_implementation()
        self._verify_predecessor_feature_rows_unchanged(
            command=command,
            expected_rows=predecessor_feature_rows,
        )
        if not self._test_only_allow_effects:
            self._verify_production_authorities(command)
        self._require_before_deadline(
            attempt_deadline,
            "before attempt registration",
        )
        self._store.register_attempt(attempt_plan)
        capability: Any | None = None
        publication_intent: Any | None = None
        governance_deadlines: _GovernanceDeadlines | None = None
        terminal_committed = False
        try:
            self._verify_live_implementation()
            self._verify_predecessor_feature_rows_unchanged(
                command=command,
                expected_rows=predecessor_feature_rows,
            )
            if not self._test_only_allow_effects:
                self._verify_production_authorities(command)
            self._require_before_deadline(
                attempt_deadline,
                "before attempt consumption",
            )
            capability = self._store.consume_attempt(attempt_id)
            history = self._store.attempt_history(attempt_id)
            validate_attempt_history(
                attempt_plan=attempt_plan,
                implementation_manifest=self._implementation,
                transitions=history,
            )
            if history[-1]["status"] != CONSUMED:
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Attempt was not durably consumed before effects"
                )
            (
                outputs,
                receipts,
                acquisition_report_authority,
                acquisition_execution,
                runtime_receipt,
            ) = (
                self._execute_phases(
                command=command,
                attempt_id=attempt_id,
                capability=capability,
                attempt_start=attempt_start,
                attempt_deadline=attempt_deadline,
                )
            )
            if (
                command in _SCORING_COMMANDS
                and runtime_receipt is None
            ):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Scoring reached Gemma without pinned runtime identity"
                )

            evaluation: Mapping[str, Any] | None = None
            if command == DEVELOPMENT_ACQUISITION_COMMAND:
                governance_deadlines = _governance_deadlines(
                    start=attempt_start,
                    strict_parent_deadline=attempt_deadline,
                    receipts=receipts,
                )
                _budget_report(
                    command=command,
                    receipts=receipts,
                    start=attempt_start,
                    end=governance_deadlines.finalization,
                )
                self._require_before_deadline(
                    governance_deadlines.intent_prepare,
                    "before acquisition evidence persistence",
                )
                persisted_phase_evidence = _append_phase_evidence(
                    store=self._store,
                    capability=capability,
                    command=command,
                    phase_outputs=outputs,
                    phase_receipts=receipts,
                    runtime_receipt=runtime_receipt,
                )
                acquisition_payload = outputs["acquisition"]["payload"]
                verified_acquisition = (
                    _validate_acquisition_report_payload(
                        acquisition_payload[
                            "verified_acquisition_report"
                        ],
                        command=command,
                    )
                )
                artifact_receipt = self._store.append_artifact(
                    capability=capability,
                    effect="deterministic_private_quarantine",
                    identity=f"terminal_artifact:{attempt_id}",
                    payload=verified_acquisition,
                )
                report_material = self._store.terminal_evidence_material(
                    capability
                )
                reconstruction_material = (
                    _build_terminal_reconstruction_material(
                        implementation_manifest=self._implementation,
                        store=self._store,
                        attempt_plan=attempt_plan,
                        terminal_status=TERMINAL_PASS,
                        report_kind=ACQUISITION_PASS,
                        terminal_artifact=verified_acquisition,
                        terminal_artifact_receipt=artifact_receipt,
                        report_material=report_material,
                        persisted_phase_evidence=(
                            persisted_phase_evidence
                        ),
                        acquisition_execution=acquisition_execution,
                    )
                )
                reconstruction = (
                    self._store.commit_terminal_reconstruction_material(
                        capability, reconstruction_material
                    )
                )
                prepared = prepare_external_publication(
                    implementation_manifest=self._implementation,
                    attempt_id=attempt_id,
                    terminal_status=PUBLICATION_TERMINAL_PASS,
                    report_kind=ACQUISITION_PASS,
                    artifact_sha256=verified_acquisition[
                        "validation_sha256"
                    ],
                    predecessor_publication_sha256=(
                        predecessor_publication_sha256
                    ),
                )
                elapsed_at_intent = (
                    self._require_before_deadline(
                        governance_deadlines.intent_prepare,
                        "before acquisition publication intent",
                    )
                    - attempt_start
                )
                intent_material = _build_publication_intent(
                    implementation_manifest=self._implementation,
                    store=self._store,
                    attempt_plan=attempt_plan,
                    terminal_status=TERMINAL_PASS,
                    report_kind=ACQUISITION_PASS,
                    artifact_sha256=verified_acquisition[
                        "validation_sha256"
                    ],
                    artifact_receipt=artifact_receipt,
                    reconstruction=reconstruction,
                    report_material=report_material,
                    elapsed_seconds=elapsed_at_intent,
                    predecessor_publication_sha256=(
                        predecessor_publication_sha256
                    ),
                    prepared_publication=prepared,
                )
                publication_intent = self._store.commit_publication_intent(
                    capability, intent_material
                )
                publication_started_at = self._require_before_deadline(
                    governance_deadlines.intent_prepare,
                    "after acquisition publication intent",
                )
                publication_deadline = min(
                    governance_deadlines.publication,
                    publication_started_at
                    + _NORMAL_PUBLICATION_SECONDS,
                )
                governance_deadlines = _GovernanceDeadlines(
                    intent_prepare=governance_deadlines.intent_prepare,
                    publication=publication_deadline,
                    finalization=min(
                        governance_deadlines.finalization,
                        publication_deadline
                        + _TERMINAL_FINALIZATION_SECONDS,
                    ),
                )
                publication_receipt = self._run_normal_publication(
                    attempt_id=attempt_id,
                    publication_intent=publication_intent,
                    prepared=prepared,
                    deadline_monotonic=governance_deadlines.publication,
                )
                finalization_started_at = self._require_before_deadline(
                    governance_deadlines.publication,
                    "after acquisition publication",
                )
                governance_deadlines = _GovernanceDeadlines(
                    intent_prepare=governance_deadlines.intent_prepare,
                    publication=governance_deadlines.publication,
                    finalization=min(
                        governance_deadlines.finalization,
                        finalization_started_at
                        + _TERMINAL_FINALIZATION_SECONDS,
                    ),
                )
                if publication_receipt is None:
                    self._require_before_deadline(
                        governance_deadlines.finalization,
                        "before acquisition pending handoff",
                    )
                    return _publication_pending_result(
                        command=command,
                        attempt_id=attempt_id,
                        publication_intent=publication_intent,
                    )
                publication = publication_receipt.external_publication
                if not is_verified_external_publication(publication):
                    raise SecGemmaOnlineRiskOverlayRunnerError(
                        "Acquisition publication receipt lost its remote authority"
                    )
                terminal_evidence = (
                    self._deps.issue_acquisition_terminal_evidence_fn(
                        implementation_manifest=self._implementation,
                        attempt_plan=attempt_plan,
                        acquisition_report=acquisition_report_authority,
                        report_material=report_material,
                        acquisition_artifact_receipt=artifact_receipt,
                        publication_intent=publication_intent,
                        publication_receipt=publication_receipt,
                    )
                )
                self._require_before_deadline(
                    governance_deadlines.finalization,
                    "before acquisition terminal anchor",
                )
                terminalization = (
                    self._store.issue_terminalization_capability(
                        attempt_id
                    )
                )
                claim = self._store.commit_terminalization_claim(
                    terminalization,
                    terminal_evidence,
                    TERMINAL_PASS,
                )
                self._store.complete_terminalized_attempt(
                    claim, terminal_evidence
                )
                terminal_committed = True
                terminal = self._store.attempt_history(attempt_id)[-1]
                result = _success_result(
                    command=command,
                    attempt_id=attempt_id,
                    terminal_transition=terminal,
                    terminal_evidence=terminal_evidence.evidence,
                    external_publication=publication.publication,
                    terminal_artifact=verified_acquisition,
                    terminal_artifact_receipt=artifact_receipt,
                )
                self._require_before_deadline(
                    governance_deadlines.finalization,
                    "after sealed acquisition result",
                )
                return result
            else:
                stage = _COMMAND_TO_METRIC_STAGE[command]
                (
                    input_bundle,
                    current_feature_rows,
                    evaluation,
                    gate,
                    receipts,
                ) = self._complete_scored_deterministic_work(
                    stage=stage,
                    outputs=outputs,
                    predecessor_feature_rows=predecessor_feature_rows,
                    receipts=receipts,
                    attempt_deadline=attempt_deadline,
                )
                governance_deadlines = _governance_deadlines(
                    start=attempt_start,
                    strict_parent_deadline=attempt_deadline,
                    receipts=receipts,
                )
                budgets = _budget_report(
                    command=command,
                    receipts=receipts,
                    start=attempt_start,
                    end=governance_deadlines.finalization,
                )
                self._require_before_deadline(
                    governance_deadlines.intent_prepare,
                    "before scored evidence persistence",
                )
                persisted_phase_evidence = _append_phase_evidence(
                    store=self._store,
                    capability=capability,
                    command=command,
                    phase_outputs=outputs,
                    phase_receipts=receipts,
                    runtime_receipt=runtime_receipt,
                )
                _persist_scored_evaluation(
                    store=self._store,
                    capability=capability,
                    input_bundle=input_bundle,
                    current_feature_rows=current_feature_rows,
                    evaluation=evaluation,
                )
                self._require_before_deadline(
                    governance_deadlines.intent_prepare,
                    "after scored evidence persistence",
                )
                governance_checks = {
                    key: True for key in _GOVERNANCE_CHECKS
                }

            assert evaluation is not None
            joint = _joint_report(
                command=command,
                attempt_plan=attempt_plan,
                implementation_manifest=self._implementation,
                phase_budget_report=budgets,
                runtime_receipt=runtime_receipt,
                evaluation=evaluation,
                acquisition_summary=None,
                governance_checks=governance_checks,
            )
            self._require_before_deadline(
                governance_deadlines.intent_prepare,
                "after joint report construction",
            )
            artifact_receipt = self._store.append_artifact(
                capability=capability,
                effect="joint_report_seal",
                identity=f"terminal_artifact:{attempt_id}",
                payload=joint,
            )
            report_material = self._store.terminal_evidence_material(
                capability
            )
            gate_passed = evaluation["gate_report"]["passed"] is True
            terminal_status = (
                TERMINAL_PASS if gate_passed else TERMINAL_FAIL
            )
            publication_status = (
                PUBLICATION_TERMINAL_PASS
                if gate_passed
                else PUBLICATION_TERMINAL_FAIL
            )
            report_kind = (
                SCORED_PASS if gate_passed else SCORED_FAILED_GATE
            )
            reconstruction_material = (
                _build_terminal_reconstruction_material(
                    implementation_manifest=self._implementation,
                    store=self._store,
                    attempt_plan=attempt_plan,
                    terminal_status=terminal_status,
                    report_kind=report_kind,
                    terminal_artifact=joint,
                    terminal_artifact_receipt=artifact_receipt,
                    report_material=report_material,
                    persisted_phase_evidence=persisted_phase_evidence,
                    gate_checks=gate["checks"],
                )
            )
            reconstruction = (
                self._store.commit_terminal_reconstruction_material(
                    capability, reconstruction_material
                )
            )
            prepared = prepare_external_publication(
                implementation_manifest=self._implementation,
                attempt_id=attempt_id,
                terminal_status=publication_status,
                report_kind=report_kind,
                artifact_sha256=joint["joint_stage_report_sha256"],
                predecessor_publication_sha256=(
                    predecessor_publication_sha256
                ),
            )
            elapsed_at_intent = (
                self._require_before_deadline(
                    governance_deadlines.intent_prepare,
                    "before scored publication intent",
                )
                - attempt_start
            )
            intent_material = _build_publication_intent(
                implementation_manifest=self._implementation,
                store=self._store,
                attempt_plan=attempt_plan,
                terminal_status=terminal_status,
                report_kind=report_kind,
                artifact_sha256=joint["joint_stage_report_sha256"],
                artifact_receipt=artifact_receipt,
                reconstruction=reconstruction,
                report_material=report_material,
                elapsed_seconds=elapsed_at_intent,
                predecessor_publication_sha256=(
                    predecessor_publication_sha256
                ),
                prepared_publication=prepared,
            )
            publication_intent = self._store.commit_publication_intent(
                capability, intent_material
            )
            publication_started_at = self._require_before_deadline(
                governance_deadlines.intent_prepare,
                "after scored publication intent",
            )
            publication_deadline = min(
                governance_deadlines.publication,
                publication_started_at + _NORMAL_PUBLICATION_SECONDS,
            )
            governance_deadlines = _GovernanceDeadlines(
                intent_prepare=governance_deadlines.intent_prepare,
                publication=publication_deadline,
                finalization=min(
                    governance_deadlines.finalization,
                    publication_deadline
                    + _TERMINAL_FINALIZATION_SECONDS,
                ),
            )
            publication_receipt = self._run_normal_publication(
                attempt_id=attempt_id,
                publication_intent=publication_intent,
                prepared=prepared,
                deadline_monotonic=governance_deadlines.publication,
            )
            finalization_started_at = self._require_before_deadline(
                governance_deadlines.publication,
                "after scored publication",
            )
            governance_deadlines = _GovernanceDeadlines(
                intent_prepare=governance_deadlines.intent_prepare,
                publication=governance_deadlines.publication,
                finalization=min(
                    governance_deadlines.finalization,
                    finalization_started_at
                    + _TERMINAL_FINALIZATION_SECONDS,
                ),
            )
            if publication_receipt is None:
                self._require_before_deadline(
                    governance_deadlines.finalization,
                    "before scored pending handoff",
                )
                return _publication_pending_result(
                    command=command,
                    attempt_id=attempt_id,
                    publication_intent=publication_intent,
                )
            publication = publication_receipt.external_publication
            if not is_verified_external_publication(publication):
                raise SecGemmaOnlineRiskOverlayRunnerError(
                    "Scored publication receipt lost its remote authority"
                )
            terminal_evidence = (
                self._deps.issue_scored_terminal_evidence_fn(
                    implementation_manifest=self._implementation,
                    attempt_plan=attempt_plan,
                    joint_stage_report=joint,
                    report_material=report_material,
                    joint_artifact_receipt=artifact_receipt,
                    gate_checks=gate["checks"],
                    publication_intent=publication_intent,
                    publication_receipt=publication_receipt,
                )
            )
            self._require_before_deadline(
                governance_deadlines.finalization,
                "before scored terminal anchor",
            )
            terminalization = (
                self._store.issue_terminalization_capability(
                    attempt_id
                )
            )
            claim = self._store.commit_terminalization_claim(
                terminalization,
                terminal_evidence,
                terminal_status,
            )
            self._store.complete_terminalized_attempt(
                claim, terminal_evidence
            )
            terminal_committed = True
            terminal = self._store.attempt_history(attempt_id)[-1]
            if gate_passed:
                result = _success_result(
                    command=command,
                    attempt_id=attempt_id,
                    terminal_transition=terminal,
                    terminal_evidence=terminal_evidence.evidence,
                    external_publication=publication.publication,
                    terminal_artifact=joint,
                    terminal_artifact_receipt=artifact_receipt,
                )
            else:
                result = _failure_result(
                    command=command,
                    attempt_id=attempt_id,
                    terminal_status=TERMINAL_FAIL,
                    diagnostic_code="gate_failure_sealed",
                    terminal_transition=terminal,
                    terminal_evidence=terminal_evidence.evidence,
                    external_publication=publication.publication,
                    terminal_artifact=joint,
                    terminal_artifact_receipt=artifact_receipt,
                )
            self._require_before_deadline(
                governance_deadlines.finalization,
                "after sealed scored result",
            )
            return result
        except SecGemmaOnlineRiskOverlayRunnerIndeterminate:
            if capability is None or terminal_committed:
                raise
            if publication_intent is None:
                try:
                    publication_intent = (
                        self._store.publication_intent_authority(
                            attempt_id
                        )
                    )
                except Exception:
                    publication_intent = None
            if publication_intent is not None:
                if governance_deadlines is None:
                    raise
                self._require_before_deadline(
                    governance_deadlines.finalization,
                    "before exceptional publication-pending handoff",
                )
                return _publication_pending_result(
                    command=command,
                    attempt_id=attempt_id,
                    publication_intent=publication_intent,
                )
            self._store.finish_attempt(
                capability,
                terminal_status=TERMINAL_INDETERMINATE,
            )
            terminal = self._store.attempt_history(attempt_id)[-1]
            return _failure_result(
                command=command,
                attempt_id=attempt_id,
                terminal_status=TERMINAL_INDETERMINATE,
                diagnostic_code="worker_or_deadline_indeterminate",
                terminal_transition=terminal,
            )
        except Exception:
            if capability is None or terminal_committed:
                raise
            if publication_intent is None:
                try:
                    publication_intent = (
                        self._store.publication_intent_authority(
                            attempt_id
                        )
                    )
                except Exception:
                    publication_intent = None
            if publication_intent is not None:
                if governance_deadlines is None:
                    raise
                self._require_before_deadline(
                    governance_deadlines.finalization,
                    "before exceptional publication-pending handoff",
                )
                return _publication_pending_result(
                    command=command,
                    attempt_id=attempt_id,
                    publication_intent=publication_intent,
                )
            self._store.finish_attempt(
                capability,
                terminal_status=TERMINAL_INDETERMINATE,
            )
            terminal = self._store.attempt_history(attempt_id)[-1]
            return _failure_result(
                command=command,
                attempt_id=attempt_id,
                terminal_status=TERMINAL_INDETERMINATE,
                diagnostic_code="post_consumption_indeterminate",
                terminal_transition=terminal,
            )

    def verify(
        self,
        *,
        sealed_result: Mapping[str, Any],
        attempt_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._verify_live_implementation()
        result = validate_sealed_stage_result(
            sealed_result,
            implementation_manifest=self._implementation,
            attempt_plan=attempt_plan,
        )
        snapshot = _mapping(
            self._store.snapshot(), "verify store snapshot"
        )
        body = {
            "schema_version": VERIFY_RESULT_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "command": VERIFY_COMMAND,
            "sealed_stage_result_sha256": result[
                "sealed_stage_result_sha256"
            ],
            "attempt_id": result["attempt_id"],
            "terminal_status": result["terminal_status"],
            "store_snapshot_sha256": canonical_sha256(snapshot),
            "passed": True,
        }
        return {**body, "verify_result_sha256": canonical_sha256(body)}


__all__ = [
    "AcquisitionAdapter",
    "CONFIRMATION_COMMAND",
    "DETERMINISTIC_EVALUATION_SCHEMA_VERSION",
    "DEVELOPMENT_ACQUISITION_COMMAND",
    "DEVELOPMENT_COMMAND",
    "DefaultDeterministicStageEvaluator",
    "ExternalReportPublisher",
    "FINAL_COMMAND",
    "FinalRegistryAuthority",
    "JOINT_STAGE_REPORT_SCHEMA_VERSION",
    "LOCAL_PREFLIGHT",
    "PHASE_BUDGET_REPORT_SCHEMA_VERSION",
    "PHASE_OUTPUT_SCHEMA_VERSION",
    "PRODUCTION_EFFECT_BLOCKERS",
    "PhaseExecutor",
    "PhasePermit",
    "RUNNER_COMMANDS",
    "RUNNER_PLAN_SCHEMA_VERSION",
    "RunnerDependencies",
    "SEALED_STAGE_RESULT_SCHEMA_VERSION",
    "STAGE_INPUT_BUNDLE_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayRunner",
    "SecGemmaOnlineRiskOverlayRunnerError",
    "SecGemmaOnlineRiskOverlayRunnerIndeterminate",
    "StageStore",
    "SupervisedPublicationRecoveryWorkerAuthority",
    "VERIFY_COMMAND",
    "VERIFY_RESULT_SCHEMA_VERSION",
    "build_phase_output",
    "build_runner_plan",
    "default_runner_dependencies",
    "is_phase_permit",
    "validate_phase_output",
    "validate_runner_plan",
    "validate_sealed_stage_result",
]
