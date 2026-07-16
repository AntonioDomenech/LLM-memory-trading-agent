"""Verified v2.1 final-registry successor authorization.

The frozen predecessor is the exact repository pin file named by the contract.
This module validates those bytes, appends exactly one ``registered_unrun``
successor for the preregistered final attempt, publishes the successor registry
hash through the opaque external publisher, and only then issues an opaque
authorization.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
from pathlib import Path
import time
from typing import Any, Final, Mapping, Protocol

from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    validate_implementation_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BRANCH_NAME,
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    FINAL_ATTEMPT_ID,
    FINAL_REGISTRY_AUTHORIZATION_FIELDS,
    FINAL_REGISTRY_SUCCESSOR_FIELDS,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
    FINAL_REGISTRY_SUCCESSOR,
    REGISTERED_UNRUN,
    SCORED_PASS,
    TERMINAL_PASS,
    VerifiedExternalPublication,
    is_verified_external_publication,
    validate_external_publication,
)


FINAL_REGISTRY_SUCCESSOR_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-final-registry-successor-v1"
)
FINAL_REGISTRY_AUTHORIZATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-final-registry-authorization-v1"
)
FINAL_REGISTRY_VERIFIER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-final-registry-verifier-v1"
)
INITIAL_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-initial-external-registry-pin-v1"
)
PREDECESSOR_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-repository-holdout-pin-v1"
)
PREDECESSOR_CONTRACT_VERSION: Final[str] = "aapl-sec-filing-gemma-v1"
MIGRATION_SOURCE_CANONICAL_SHA256: Final[str] = (
    "c6efc6de1facd1aed2a0709749219e0a82e862432699d11b24a3eb3ec9bebb4b"
)

_VERIFIED_FINAL_REGISTRY_AUTHORIZATION_SENTINEL = object()


class SecGemmaOnlineRiskOverlayRegistryError(RuntimeError):
    """The frozen predecessor or its published successor failed validation."""


class ExternalRegistryPublisher(Protocol):
    def publish_or_recover_final_registry(
        self,
        *,
        attempt_id: str,
        terminal_status: str,
        report_kind: str,
        artifact_sha256: str,
        predecessor_publication_sha256: str,
        deadline_monotonic: float,
    ) -> VerifiedExternalPublication: ...


class TerminalAnchorStore(Protocol):
    def terminal_anchor_binding(
        self,
        attempt_id: str,
    ) -> Mapping[str, Any]: ...


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            f"{location} must be one detached plain mapping"
        )
    return copy.deepcopy(value)


def _sha256(value: Any, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecGemmaOnlineRiskOverlayRegistryError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _strict_int(value: Any, location: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def _frozen_final_access() -> dict[str, Any]:
    return build_contract_manifest()["stage_access"]["final"]


def validate_predecessor_registry_pin(value: Any) -> dict[str, Any]:
    """Validate the exact legacy pin material frozen into v2.1."""

    observed = _mapping(value, "predecessor registry pin file")
    if set(observed) != {
        "contract_version",
        "migration_source_canonical_sha256",
        "registry_pin",
        "schema_version",
    }:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Predecessor registry pin file fields changed"
        )
    registry_pin = _mapping(
        observed["registry_pin"],
        "predecessor registry pin",
    )
    if set(registry_pin) != {
        "historical_final_reveal_count_lower_bound",
        "registered_entry_count",
        "registry_sha256",
        "schema_version",
        "tip_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Predecessor registry pin fields changed"
        )
    access = _frozen_final_access()
    lower_bound = _strict_int(
        registry_pin["historical_final_reveal_count_lower_bound"],
        "historical final reveal count lower bound",
        minimum=0,
    )
    registered_count = _strict_int(
        registry_pin["registered_entry_count"],
        "registered predecessor entry count",
        minimum=0,
    )
    registry_hash = _sha256(
        registry_pin["registry_sha256"],
        "predecessor registry hash",
    )
    tip_hash = _sha256(
        registry_pin["tip_sha256"],
        "predecessor registry tip hash",
    )
    if (
        observed["schema_version"] != INITIAL_PIN_SCHEMA_VERSION
        or observed["contract_version"] != PREDECESSOR_CONTRACT_VERSION
        or observed["migration_source_canonical_sha256"]
        != MIGRATION_SOURCE_CANONICAL_SHA256
        or registry_pin["schema_version"]
        != PREDECESSOR_PIN_SCHEMA_VERSION
        or lower_bound
        != access["historical_final_reveal_count_lower_bound"]
        or registered_count != 0
        or registry_hash != access["predecessor_registry_sha256"]
        or tip_hash != access["predecessor_registry_tip_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Predecessor registry pin differs from the frozen v2.1 migration"
        )
    return {
        "contract_version": PREDECESSOR_CONTRACT_VERSION,
        "migration_source_canonical_sha256": (
            MIGRATION_SOURCE_CANONICAL_SHA256
        ),
        "registry_pin": {
            "historical_final_reveal_count_lower_bound": lower_bound,
            "registered_entry_count": registered_count,
            "registry_sha256": registry_hash,
            "schema_version": PREDECESSOR_PIN_SCHEMA_VERSION,
            "tip_sha256": tip_hash,
        },
        "schema_version": INITIAL_PIN_SCHEMA_VERSION,
    }


def _predecessor_reveal_count(
    predecessor_pin: Mapping[str, Any],
) -> int:
    validated = validate_predecessor_registry_pin(predecessor_pin)
    pin = validated["registry_pin"]
    return (
        pin["historical_final_reveal_count_lower_bound"]
        + pin["registered_entry_count"]
    )


def build_final_registry_successor(
    *,
    implementation_manifest: Mapping[str, Any],
    predecessor_pin: Mapping[str, Any],
) -> dict[str, Any]:
    """Append exactly the preregistered final attempt to the frozen pin."""

    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    pin = validate_predecessor_registry_pin(predecessor_pin)
    registry_pin = pin["registry_pin"]
    predecessor_count = _predecessor_reveal_count(pin)
    successor_ordinal = predecessor_count + 1
    entry_body = {
        "schema_version": FINAL_REGISTRY_SUCCESSOR_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "branch": BRANCH_NAME,
        "implementation_commit": implementation["implementation_commit"],
        "predecessor_registry_sha256": registry_pin["registry_sha256"],
        "predecessor_registry_tip_sha256": registry_pin["tip_sha256"],
        "predecessor_reveal_count": predecessor_count,
        "successor_ordinal": successor_ordinal,
        "final_attempt_id": FINAL_ATTEMPT_ID,
        "status": REGISTERED_UNRUN,
    }
    entry_hash = canonical_sha256(entry_body)
    registry_body = {
        **entry_body,
        "successor_entry_sha256": entry_hash,
    }
    successor = {
        **registry_body,
        "successor_registry_sha256": canonical_sha256(registry_body),
    }
    if tuple(successor) != FINAL_REGISTRY_SUCCESSOR_FIELDS:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry successor fields differ from the frozen contract"
        )
    return successor


def validate_final_registry_successor(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
    predecessor_pin: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact successor without making it authoritative."""

    observed = _mapping(value, "final registry successor")
    if tuple(observed) != FINAL_REGISTRY_SUCCESSOR_FIELDS:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry successor fields changed"
        )
    _sha256(
        observed["successor_entry_sha256"],
        "successor entry hash",
    )
    _sha256(
        observed["successor_registry_sha256"],
        "successor registry hash",
    )
    expected = build_final_registry_successor(
        implementation_manifest=implementation_manifest,
        predecessor_pin=predecessor_pin,
    )
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry successor differs from its canonical reconstruction"
        )
    return expected


def _authorization_body(
    *,
    successor: Mapping[str, Any],
    external_publication: VerifiedExternalPublication,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    if not is_verified_external_publication(external_publication):
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry authorization requires opaque publication proof"
        )
    publication = validate_external_publication(
        external_publication.publication,
        implementation_manifest=implementation,
    )
    if (
        publication["attempt_id"] != FINAL_ATTEMPT_ID
        or publication["terminal_status"] != REGISTERED_UNRUN
        or publication["report_kind"] != FINAL_REGISTRY_SUCCESSOR
        or publication["artifact_sha256"]
        != successor["successor_registry_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry publication does not bind the exact successor"
        )
    return {
        "schema_version": FINAL_REGISTRY_AUTHORIZATION_SCHEMA_VERSION,
        "verifier_id": FINAL_REGISTRY_VERIFIER_ID,
        "predecessor_registry_sha256": successor[
            "predecessor_registry_sha256"
        ],
        "predecessor_registry_tip_sha256": successor[
            "predecessor_registry_tip_sha256"
        ],
        "predecessor_reveal_count": successor["predecessor_reveal_count"],
        "successor_entry_sha256": successor["successor_entry_sha256"],
        "successor_registry_sha256": successor[
            "successor_registry_sha256"
        ],
        "external_publication_sha256": publication["publication_sha256"],
        "final_attempt_id": FINAL_ATTEMPT_ID,
    }


def build_final_registry_authorization(
    *,
    successor: Mapping[str, Any],
    external_publication: VerifiedExternalPublication,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact authorization payload after opaque publication."""

    body = _authorization_body(
        successor=successor,
        external_publication=external_publication,
        implementation_manifest=implementation_manifest,
    )
    authorization = {
        **body,
        "authorization_sha256": canonical_sha256(body),
    }
    if tuple(authorization) != FINAL_REGISTRY_AUTHORIZATION_FIELDS:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry authorization fields differ from the contract"
        )
    return authorization


def validate_final_registry_authorization(
    value: Any,
    *,
    successor: Mapping[str, Any],
    external_publication: VerifiedExternalPublication,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an authorization payload without issuing opaque authority."""

    observed = _mapping(value, "final registry authorization")
    if tuple(observed) != FINAL_REGISTRY_AUTHORIZATION_FIELDS:
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry authorization fields changed"
        )
    digest = _sha256(
        observed["authorization_sha256"],
        "final registry authorization hash",
    )
    expected = build_final_registry_authorization(
        successor=successor,
        external_publication=external_publication,
        implementation_manifest=implementation_manifest,
    )
    if (
        not hmac.compare_digest(
            digest,
            expected["authorization_sha256"],
        )
        or observed != expected
    ):
        raise SecGemmaOnlineRiskOverlayRegistryError(
            "Final registry authorization differs from its canonical reconstruction"
        )
    return expected


class VerifiedFinalRegistryAuthorization:
    """Opaque proof that the final successor was registered and published."""

    __slots__ = (
        "_authorization",
        "_successor",
        "_external_publication",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        authorization: Mapping[str, Any],
        successor: Mapping[str, Any],
        external_publication: VerifiedExternalPublication,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VERIFIED_FINAL_REGISTRY_AUTHORIZATION_SENTINEL:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Verified final registry authorization requires its verifier"
            )
        self._authorization = copy.deepcopy(dict(authorization))
        self._successor = copy.deepcopy(dict(successor))
        self._external_publication = external_publication
        self._sentinel = _sentinel

    @property
    def authorization(self) -> dict[str, Any]:
        return copy.deepcopy(self._authorization)

    @property
    def authorization_sha256(self) -> str:
        return self._authorization["authorization_sha256"]

    @property
    def successor(self) -> dict[str, Any]:
        return copy.deepcopy(self._successor)

    @property
    def successor_registry_sha256(self) -> str:
        return self._authorization["successor_registry_sha256"]

    @property
    def external_publication(self) -> VerifiedExternalPublication:
        return self._external_publication

    def __repr__(self) -> str:
        return "VerifiedFinalRegistryAuthorization(<redacted>)"


def is_verified_final_registry_authorization(value: Any) -> bool:
    """Return whether ``value`` is authority issued by this module."""

    return (
        type(value) is VerifiedFinalRegistryAuthorization
        and getattr(value, "_sentinel", None)
        is _VERIFIED_FINAL_REGISTRY_AUTHORIZATION_SENTINEL
        and is_verified_external_publication(
            getattr(value, "_external_publication", None)
        )
    )


def _issue_verified_final_registry_authorization(
    *,
    authorization: Mapping[str, Any],
    successor: Mapping[str, Any],
    external_publication: VerifiedExternalPublication,
    implementation_manifest: Mapping[str, Any],
) -> VerifiedFinalRegistryAuthorization:
    validated = validate_final_registry_authorization(
        authorization,
        successor=successor,
        external_publication=external_publication,
        implementation_manifest=implementation_manifest,
    )
    return VerifiedFinalRegistryAuthorization(
        authorization=validated,
        successor=successor,
        external_publication=external_publication,
        _sentinel=_VERIFIED_FINAL_REGISTRY_AUTHORIZATION_SENTINEL,
    )


class FinalRegistryAuthorizer:
    """Validate, append, publish, and authorize the final successor."""

    __slots__ = ("_repo_root", "_publisher", "_store", "_clock")

    def __init__(
        self,
        *,
        repo_root: Path,
        publisher: ExternalRegistryPublisher,
        store: TerminalAnchorStore,
        clock: Any = time.monotonic,
    ) -> None:
        if not isinstance(repo_root, Path) or not repo_root.is_absolute():
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Registry repo_root must be an absolute pathlib.Path"
            )
        try:
            resolved = repo_root.resolve(strict=True)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Registry repository root is unavailable"
            ) from exc
        if not resolved.is_dir():
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Registry repository root is not a directory"
            )
        if not callable(clock):
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Registry clock must be callable"
            )
        self._repo_root = resolved
        self._publisher = publisher
        self._store = store
        self._clock = clock

    def _remaining(self, deadline_monotonic: float) -> float:
        if (
            type(deadline_monotonic) not in {int, float}
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Registry deadline must be one finite monotonic value"
            )
        try:
            now = float(self._clock())
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Registry monotonic clock failed"
            ) from exc
        remaining = float(deadline_monotonic) - now
        if not math.isfinite(now) or now < 0.0 or remaining <= 0.0:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Registry authorization deadline has expired"
            )
        return remaining

    def _load_predecessor_pin(
        self,
        *,
        deadline_monotonic: float,
    ) -> dict[str, Any]:
        self._remaining(deadline_monotonic)
        access = _frozen_final_access()
        relative = Path(access["predecessor_registry_pin_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Frozen predecessor pin path is not repository-relative"
            )
        candidate = self._repo_root.joinpath(relative)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self._repo_root)
        except (OSError, ValueError) as exc:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Frozen predecessor pin file is unavailable or escaped the repository"
            ) from exc
        if candidate.is_symlink() or not resolved.is_file():
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Frozen predecessor pin must be one regular repository file"
            )
        try:
            raw_bytes = resolved.read_bytes()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Frozen predecessor pin bytes could not be read"
            ) from exc
        observed_hash = hashlib.sha256(raw_bytes).hexdigest()
        if not hmac.compare_digest(
            observed_hash,
            access["predecessor_registry_pin_file_sha256"],
        ):
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Frozen predecessor pin file hash changed"
            )
        try:
            parsed = json.loads(raw_bytes.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Frozen predecessor pin is not strict UTF-8 JSON"
            ) from exc
        self._remaining(deadline_monotonic)
        return validate_predecessor_registry_pin(parsed)

    def authorize_final(
        self,
        *,
        implementation_manifest: Mapping[str, Any],
        deadline_monotonic: float,
    ) -> VerifiedFinalRegistryAuthorization:
        """Publish the successor before returning final-registration authority."""

        implementation = validate_implementation_manifest(
            implementation_manifest
        )
        binding = _mapping(
            self._store.terminal_anchor_binding(
                CONFIRMATION_ATTEMPT_ID
            ),
            "confirmation terminal anchor binding",
        )
        if set(binding) != {
            "terminal_evidence",
            "external_publication",
            "artifact_receipt",
        }:
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Confirmation terminal anchor binding fields changed"
            )
        predecessor_publication_payload = validate_external_publication(
            binding["external_publication"],
            implementation_manifest=implementation,
        )
        terminal_evidence = _mapping(
            binding["terminal_evidence"],
            "confirmation terminal evidence",
        )
        if (
            predecessor_publication_payload["attempt_id"]
            != CONFIRMATION_ATTEMPT_ID
            or predecessor_publication_payload["terminal_status"]
            != TERMINAL_PASS
            or predecessor_publication_payload["report_kind"] != SCORED_PASS
            or terminal_evidence.get("attempt_id")
            != CONFIRMATION_ATTEMPT_ID
            or terminal_evidence.get("terminal_status") != TERMINAL_PASS
            or terminal_evidence.get("external_publication_sha256")
            != predecessor_publication_payload["publication_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Confirmation terminal anchor is not the exact published pass"
            )
        predecessor_publication = predecessor_publication_payload[
            "publication_sha256"
        ]
        predecessor_pin = self._load_predecessor_pin(
            deadline_monotonic=deadline_monotonic
        )
        successor = build_final_registry_successor(
            implementation_manifest=implementation,
            predecessor_pin=predecessor_pin,
        )
        publication = self._publisher.publish_or_recover_final_registry(
            attempt_id=FINAL_ATTEMPT_ID,
            terminal_status=REGISTERED_UNRUN,
            report_kind=FINAL_REGISTRY_SUCCESSOR,
            artifact_sha256=successor["successor_registry_sha256"],
            predecessor_publication_sha256=predecessor_publication,
            deadline_monotonic=deadline_monotonic,
        )
        if not is_verified_external_publication(publication):
            raise SecGemmaOnlineRiskOverlayRegistryError(
                "Final registry publisher returned no opaque authority"
            )
        authorization = build_final_registry_authorization(
            successor=successor,
            external_publication=publication,
            implementation_manifest=implementation,
        )
        self._remaining(deadline_monotonic)
        return _issue_verified_final_registry_authorization(
            authorization=authorization,
            successor=successor,
            external_publication=publication,
            implementation_manifest=implementation,
        )


__all__ = [
    "FINAL_REGISTRY_AUTHORIZATION_SCHEMA_VERSION",
    "FINAL_REGISTRY_SUCCESSOR_SCHEMA_VERSION",
    "FINAL_REGISTRY_VERIFIER_ID",
    "FinalRegistryAuthorizer",
    "INITIAL_PIN_SCHEMA_VERSION",
    "MIGRATION_SOURCE_CANONICAL_SHA256",
    "PREDECESSOR_CONTRACT_VERSION",
    "PREDECESSOR_PIN_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayRegistryError",
    "TerminalAnchorStore",
    "VerifiedFinalRegistryAuthorization",
    "build_final_registry_authorization",
    "build_final_registry_successor",
    "is_verified_final_registry_authorization",
    "validate_final_registry_authorization",
    "validate_final_registry_successor",
    "validate_predecessor_registry_pin",
]
