"""Pure holdout-reveal governance for the SEC/Gemma experiment.

The registry is a deterministic JSON value.  This module performs no file,
network, clock, environment, market-data, SEC, or model I/O.  An effectful
caller is responsible for committing each returned registry and its derived
pin *before* allowing the registered candidate to inspect an intermediate or
final outcome.

The candidate manifest binds the registry snapshot that immediately precedes
its entry.  The entry then binds that candidate hash and attempt id into an
append-only hash chain.  This two-step construction avoids a circular hash
while making either side of the binding immutable once the new tip and count
have been pinned outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hmac
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    SecFilingGemmaContractError,
    canonical_sha256,
    validate_candidate_manifest,
)


REGISTRY_SCHEMA_VERSION: Final[str] = "aapl-repository-holdout-registry-v1"
REGISTRY_PIN_SCHEMA_VERSION: Final[str] = "aapl-repository-holdout-pin-v1"
REGISTRY_PIN_TRANSITION_SCHEMA_VERSION: Final[str] = (
    "aapl-repository-holdout-pin-transition-v1"
)
REGISTRY_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-filing-gemma-holdout-attempt-v1"
)
CANDIDATE_DESIGN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-filing-gemma-stable-design-v1"
)
REVEAL_REQUEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-filing-gemma-reveal-request-v1"
)
HISTORICAL_DECLARATION_SCHEMA_VERSION: Final[str] = (
    "aapl-historical-final-reveal-lower-bound-v1"
)
HISTORICAL_REGISTRY_CANONICAL_SHA256: Final[str] = (
    "c6efc6de1facd1aed2a0709749219e0a82e862432699d11b24a3eb3ec9bebb4b"
)
_STAGE_PREREQUISITES: Final[dict[str, str]] = {
    "intermediate": "development",
    "final": "intermediate",
}

# The earlier local registry began at reveal index five.  Its six immutable
# candidate hashes bring the repository-wide, explicitly known lower bound to
# ten before this SEC/Gemma approach.  The first four identities were not
# completely preserved, so this is deliberately a lower bound rather than a
# claim that the historical record is exhaustive.
UNATTRIBUTED_PRE_REGISTRY_REVEAL_COUNT_LOWER_BOUND: Final[int] = 4
_KNOWN_REGISTERED_REVEALS: Final[tuple[tuple[int, str, str], ...]] = (
    (
        5,
        "exhaustion_or_gap_v1",
        "85c6a66e5aceaa4270bfb50f8eea38c7c71279ac2de39b45eb434c6fd99e0908",
    ),
    (
        6,
        "exhaustion_or_gap_v1",
        "3ff207600ccfe24f9ae6ab65cff08d2f1b80cb8eef3953382a20829664213334",
    ),
    (
        7,
        "weak_trend_exhaustion_v1",
        "49650400347112a40ffe46dd309818b719c86140c9456ee15407964b89359aa3",
    ),
    (
        8,
        "sparse_dual_trend_exhaustion_v1",
        "8ada96fba54b9a90cd176a3962605478aa3aecfad51f9fe3428e9c843278267e",
    ),
    (
        9,
        "sparse_dual_trend_exhaustion_v1",
        "a6b0f869ac34e8345d25d58f62f568334cec4a33efc2183addd9a3030b8bbebb",
    ),
    (
        10,
        "hierarchical_empirical_bayes_irrm_h1_v1",
        "2796b55294e748f4fcb562c1faa6bcd377c2f267defb4645bd47cf71ec75c32a",
    ),
)
HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND: Final[int] = (
    UNATTRIBUTED_PRE_REGISTRY_REVEAL_COUNT_LOWER_BOUND
    + len(_KNOWN_REGISTERED_REVEALS)
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT_RE = re.compile(
    rf"{re.escape(CONTRACT_VERSION)}-attempt-(?P<sequence>[0-9]{{3}})\Z"
)


class SecFilingGemmaRevealRegistryError(ValueError):
    """Raised when reveal governance is incomplete, mutable, or inconsistent."""


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecFilingGemmaRevealRegistryError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise SecFilingGemmaRevealRegistryError(
            f"{location} keys must all be strings"
        )
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaRevealRegistryError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaRevealRegistryError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaRevealRegistryError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _same_digest(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def historical_reveal_declaration() -> dict[str, Any]:
    """Declare the known pre-SEC/Gemma final-reveal lower bound."""

    known = [
        {
            "reveal_index": reveal_index,
            "approach_id": approach_id,
            "candidate_sha256": candidate_sha256,
        }
        for reveal_index, approach_id, candidate_sha256 in _KNOWN_REGISTERED_REVEALS
    ]
    return {
        "schema_version": HISTORICAL_DECLARATION_SCHEMA_VERSION,
        "reused_holdout_periods": ["2024", "2025", "2026_ytd"],
        "unattributed_pre_registry_reveal_count_lower_bound": (
            UNATTRIBUTED_PRE_REGISTRY_REVEAL_COUNT_LOWER_BOUND
        ),
        "known_registered_reveals": known,
        "migration_source": (
            "e/bayes_v1/"
            "aapl-unleveraged-hierarchical_empirical_bayes_irrm_h1_v1-"
            "20260710T145708Z-46f8b26a/holdout_registry_snapshot.json"
        ),
        "migration_source_canonical_sha256": HISTORICAL_REGISTRY_CANONICAL_SHA256,
        "historical_final_reveal_count_lower_bound": (
            HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
        ),
        "count_semantics": "lower_bound_not_exhaustive_history",
        "historical_candidate_identities_complete": False,
    }


def _governance() -> dict[str, Any]:
    return {
        "evidence_class": "approach_specific_reused_historical_holdout",
        "append_only": True,
        "candidate_attempt_cardinality": "one_candidate_hash_to_one_attempt_id",
        "stable_candidate_design_cardinality": (
            "one_semantic_design_hash_to_one_attempt_id"
        ),
        "registration_timing": (
            "before_any_intermediate_or_final_outcome_access"
        ),
        "request_scope": "one_current_tip_candidate_only",
        "pure_registry_can_authorize_outcome_access": False,
        "candidate_registration_counts_as_final_reveal": False,
        "actual_final_reveal_count_requires_effectful_consumption_ledger": True,
        "outcome_fields_permitted": False,
        "cross_attempt_ranking_permitted": False,
        "cross_attempt_winner_selection_permitted": False,
        "candidate_replacement_after_reveal_permitted": False,
        "globally_pristine_claim_permitted": False,
    }


def _genesis_tip_sha256() -> str:
    return canonical_sha256(
        {
            "schema_version": "aapl-repository-holdout-genesis-v1",
            "contract_version": CONTRACT_VERSION,
            "historical_reveal_declaration": historical_reveal_declaration(),
            "governance": _governance(),
        }
    )


def _registry_body(
    entries: list[dict[str, Any]], *, tip_sha256: str
) -> dict[str, Any]:
    entry_count = len(entries)
    genesis = _genesis_tip_sha256()
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "historical_reveal_declaration": historical_reveal_declaration(),
        "governance": _governance(),
        "entries": copy.deepcopy(entries),
        "chain": {
            "genesis_tip_sha256": genesis,
            "tip_sha256": tip_sha256,
            "registered_entry_count": entry_count,
            "historical_final_reveal_count_lower_bound": (
                HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
            ),
        },
    }


def _registry_snapshot(
    entries: list[dict[str, Any]], *, tip_sha256: str
) -> dict[str, Any]:
    body = _registry_body(entries, tip_sha256=tip_sha256)
    return {**body, "registry_sha256": canonical_sha256(body)}


def build_initial_reveal_registry() -> dict[str, Any]:
    """Return the deterministic predecessor registry for attempt 001."""

    return _registry_snapshot([], tip_sha256=_genesis_tip_sha256())


def _entry_body(
    *,
    sequence: int,
    attempt_id: str,
    candidate_sha256: str,
    candidate_design_sha256: str,
    candidate_manifest: Mapping[str, Any],
    candidate_bound_predecessor_registry_sha256: str,
    prior_tip_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_ENTRY_SCHEMA_VERSION,
        "sequence": sequence,
        "attempt_id": attempt_id,
        "candidate_sha256": candidate_sha256,
        "candidate_design_sha256": candidate_design_sha256,
        "candidate_manifest": copy.deepcopy(dict(candidate_manifest)),
        "candidate_bound_predecessor_registry_sha256": (
            candidate_bound_predecessor_registry_sha256
        ),
        "prior_tip_sha256": prior_tip_sha256,
        "registration_semantics": (
            "single_precommitted_candidate_before_outcome_access"
        ),
        "outcome_payload_included": False,
        "cross_attempt_winner_selection_permitted": False,
        "candidate_replacement_permitted": False,
    }


def _pin_from_validated_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    chain = registry["chain"]
    return {
        "schema_version": REGISTRY_PIN_SCHEMA_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "tip_sha256": chain["tip_sha256"],
        "registered_entry_count": chain["registered_entry_count"],
        "historical_final_reveal_count_lower_bound": chain[
            "historical_final_reveal_count_lower_bound"
        ],
    }


def _validate_registry_structure(registry: Mapping[str, Any]) -> dict[str, Any]:
    value = _expect_mapping(registry, "reveal registry")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "historical_reveal_declaration",
            "governance",
            "entries",
            "chain",
            "registry_sha256",
        },
        "reveal registry",
    )
    if value["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise SecFilingGemmaRevealRegistryError("Unknown reveal-registry schema")
    if value["contract_version"] != CONTRACT_VERSION:
        raise SecFilingGemmaRevealRegistryError(
            "Reveal registry belongs to another experiment contract"
        )
    expected_history = historical_reveal_declaration()
    if (
        value["historical_reveal_declaration"] != expected_history
        or canonical_sha256(value["historical_reveal_declaration"])
        != canonical_sha256(expected_history)
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Historical final-reveal lower-bound declaration changed"
        )
    expected_governance = _governance()
    if (
        value["governance"] != expected_governance
        or canonical_sha256(value["governance"])
        != canonical_sha256(expected_governance)
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Reveal governance or no-winner-selection semantics changed"
        )
    entries_value = value["entries"]
    if not isinstance(entries_value, list):
        raise SecFilingGemmaRevealRegistryError("Registry entries must be a list")

    validated_entries: list[dict[str, Any]] = []
    seen_attempt_ids: set[str] = set()
    seen_candidate_hashes: set[str] = set()
    seen_candidate_design_hashes: set[str] = set()
    current_tip = _genesis_tip_sha256()

    for expected_sequence, raw_entry in enumerate(entries_value, start=1):
        entry = _expect_mapping(raw_entry, f"registry entry {expected_sequence}")
        entry_keys = {
            "schema_version",
            "sequence",
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "candidate_manifest",
            "candidate_bound_predecessor_registry_sha256",
            "prior_tip_sha256",
            "registration_semantics",
            "outcome_payload_included",
            "cross_attempt_winner_selection_permitted",
            "candidate_replacement_permitted",
            "entry_sha256",
        }
        _expect_keys(entry, entry_keys, f"registry entry {expected_sequence}")
        sequence = _strict_int(
            entry["sequence"], f"registry entry {expected_sequence} sequence", minimum=1
        )
        if sequence != expected_sequence:
            raise SecFilingGemmaRevealRegistryError(
                "Registry entry sequence is not contiguous"
            )
        expected_attempt_id = f"{CONTRACT_VERSION}-attempt-{sequence:03d}"
        attempt_id = entry["attempt_id"]
        match = (
            _ATTEMPT_RE.fullmatch(attempt_id)
            if isinstance(attempt_id, str)
            else None
        )
        if match is None or attempt_id != expected_attempt_id:
            raise SecFilingGemmaRevealRegistryError(
                "Attempt ids must be canonical, contiguous, and registry-unique"
            )
        candidate_hash = _sha256(
            entry["candidate_sha256"],
            f"registry entry {expected_sequence} candidate_sha256",
        )
        candidate_design_hash = _sha256(
            entry["candidate_design_sha256"],
            f"registry entry {expected_sequence} candidate_design_sha256",
        )
        (
            manifest_candidate_hash,
            manifest_design_hash,
            manifest_attempt_id,
            manifest_predecessor_hash,
        ) = _validated_candidate_identity(
            _expect_mapping(
                entry["candidate_manifest"],
                f"registry entry {expected_sequence} candidate_manifest",
            )
        )
        if (
            candidate_hash != manifest_candidate_hash
            or candidate_design_hash != manifest_design_hash
            or attempt_id != manifest_attempt_id
        ):
            raise SecFilingGemmaRevealRegistryError(
                "Registry candidate, design, and attempt are not manifest-bound"
            )
        if attempt_id in seen_attempt_ids:
            raise SecFilingGemmaRevealRegistryError(
                "Duplicate holdout attempt id is forbidden"
            )
        if candidate_hash in seen_candidate_hashes:
            raise SecFilingGemmaRevealRegistryError(
                "A candidate hash may have only one holdout attempt"
            )
        if candidate_design_hash in seen_candidate_design_hashes:
            raise SecFilingGemmaRevealRegistryError(
                "An identical frozen candidate design may have only one holdout attempt"
            )

        predecessor = _registry_snapshot(
            validated_entries,
            tip_sha256=current_tip,
        )
        predecessor_hash = _sha256(
            entry["candidate_bound_predecessor_registry_sha256"],
            (
                f"registry entry {expected_sequence} "
                "candidate_bound_predecessor_registry_sha256"
            ),
        )
        if not _same_digest(predecessor_hash, predecessor["registry_sha256"]):
            raise SecFilingGemmaRevealRegistryError(
                "Candidate does not bind the exact predecessor registry"
            )
        if not _same_digest(manifest_predecessor_hash, predecessor_hash):
            raise SecFilingGemmaRevealRegistryError(
                "Stored candidate manifest does not bind the registry predecessor"
            )
        prior_tip = _sha256(
            entry["prior_tip_sha256"],
            f"registry entry {expected_sequence} prior_tip_sha256",
        )
        if not _same_digest(prior_tip, current_tip):
            raise SecFilingGemmaRevealRegistryError("Registry tip chain is broken")

        expected_body = _entry_body(
            sequence=sequence,
            attempt_id=attempt_id,
            candidate_sha256=candidate_hash,
            candidate_design_sha256=candidate_design_hash,
            candidate_manifest=_expect_mapping(
                entry["candidate_manifest"],
                f"registry entry {expected_sequence} candidate_manifest",
            ),
            candidate_bound_predecessor_registry_sha256=predecessor_hash,
            prior_tip_sha256=prior_tip,
        )
        expected_entry_hash = canonical_sha256(expected_body)
        observed_entry_hash = _sha256(
            entry["entry_sha256"],
            f"registry entry {expected_sequence} entry_sha256",
        )
        observed_body = {
            key: entry[key] for key in entry_keys if key != "entry_sha256"
        }
        expected_entry = {**expected_body, "entry_sha256": expected_entry_hash}
        if (
            entry != expected_entry
            or canonical_sha256(observed_body) != expected_entry_hash
            or not _same_digest(observed_entry_hash, expected_entry_hash)
        ):
            raise SecFilingGemmaRevealRegistryError(
                "Registry entry is not its canonical immutable construction"
            )

        validated_entries.append(copy.deepcopy(expected_entry))
        seen_attempt_ids.add(attempt_id)
        seen_candidate_hashes.add(candidate_hash)
        seen_candidate_design_hashes.add(candidate_design_hash)
        current_tip = expected_entry_hash

    chain = _expect_mapping(value["chain"], "registry chain")
    _expect_keys(
        chain,
        {
            "genesis_tip_sha256",
            "tip_sha256",
            "registered_entry_count",
            "historical_final_reveal_count_lower_bound",
        },
        "registry chain",
    )
    expected_registry = _registry_snapshot(
        validated_entries,
        tip_sha256=current_tip,
    )
    expected_chain = expected_registry["chain"]
    _sha256(chain["genesis_tip_sha256"], "chain genesis_tip_sha256")
    _sha256(chain["tip_sha256"], "chain tip_sha256")
    _strict_int(chain["registered_entry_count"], "chain registered_entry_count")
    _strict_int(
        chain["historical_final_reveal_count_lower_bound"],
        "chain historical_final_reveal_count_lower_bound",
    )
    observed_registry_hash = _sha256(
        value["registry_sha256"], "registry_sha256"
    )
    observed_registry_body = {
        key: value[key] for key in value if key != "registry_sha256"
    }
    if (
        chain != expected_chain
        or value != expected_registry
        or canonical_sha256(observed_registry_body)
        != expected_registry["registry_sha256"]
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Registry count, tip, history, or canonical snapshot hash is inconsistent"
        )
    if not _same_digest(
        observed_registry_hash, expected_registry["registry_sha256"]
    ):
        raise SecFilingGemmaRevealRegistryError("Registry snapshot hash changed")
    return expected_registry


def derive_registry_pin(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a pin that an effectful caller must persist outside the registry.

    Derivation alone is not external authentication.  A later consumer must
    receive the previously persisted value through a separate trust path and
    pass it to :func:`validate_reveal_registry`.
    """

    validated = _validate_registry_structure(registry)
    return _pin_from_validated_registry(validated)


def _validate_pin(pin: Mapping[str, Any]) -> dict[str, Any]:
    value = _expect_mapping(pin, "external registry pin")
    _expect_keys(
        value,
        {
            "schema_version",
            "registry_sha256",
            "tip_sha256",
            "registered_entry_count",
            "historical_final_reveal_count_lower_bound",
        },
        "external registry pin",
    )
    if value["schema_version"] != REGISTRY_PIN_SCHEMA_VERSION:
        raise SecFilingGemmaRevealRegistryError("Unknown registry-pin schema")
    normalized = {
        "schema_version": REGISTRY_PIN_SCHEMA_VERSION,
        "registry_sha256": _sha256(
            value["registry_sha256"], "external pin registry_sha256"
        ),
        "tip_sha256": _sha256(value["tip_sha256"], "external pin tip_sha256"),
        "registered_entry_count": _strict_int(
            value["registered_entry_count"], "external pin registered_entry_count"
        ),
        "historical_final_reveal_count_lower_bound": _strict_int(
            value["historical_final_reveal_count_lower_bound"],
            "external pin historical_final_reveal_count_lower_bound",
        ),
    }
    return normalized


def validate_reveal_registry(
    registry: Mapping[str, Any], *, external_pin: Mapping[str, Any]
) -> str:
    """Validate the complete chain against an independently supplied pin."""

    validated = _validate_registry_structure(registry)
    expected_pin = _pin_from_validated_registry(validated)
    observed_pin = _validate_pin(external_pin)
    if observed_pin != expected_pin:
        raise SecFilingGemmaRevealRegistryError(
            "Registry does not match the externally pinned hash, tip, and counts"
        )
    return validated["registry_sha256"]


def _validated_candidate_identity(
    candidate_manifest: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    try:
        candidate_hash = validate_candidate_manifest(candidate_manifest)
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaRevealRegistryError(
            "Candidate manifest is not a valid frozen SEC/Gemma candidate"
        ) from exc
    manifest = _expect_mapping(candidate_manifest, "candidate manifest")
    candidate_body = {
        key: manifest[key] for key in manifest if key != "candidate_sha256"
    }
    if not _same_digest(canonical_sha256(candidate_body), candidate_hash):
        raise SecFilingGemmaRevealRegistryError(
            "Candidate manifest body is not bound by its candidate hash"
        )
    bindings = _expect_mapping(manifest["bindings"], "candidate bindings")
    design_hash = canonical_sha256(
        {
            "schema_version": CANDIDATE_DESIGN_SCHEMA_VERSION,
            "contract_sha256": manifest["contract_sha256"],
            "model": copy.deepcopy(dict(_expect_mapping(manifest["model"], "model"))),
            "semantic_bindings": {
                "calendar_sessions_sha256": bindings[
                    "calendar_sessions_sha256"
                ],
                "corpus_universe_semantic_sha256": bindings[
                    "corpus_universe_semantic_sha256"
                ],
                "identity_lexicon_sha256": bindings[
                    "identity_lexicon_sha256"
                ],
            },
        }
    )
    attempt_id = bindings["holdout_attempt_id"]
    if not isinstance(attempt_id, str) or _ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise SecFilingGemmaRevealRegistryError(
            "Candidate holdout attempt id is not canonical"
        )
    predecessor_hash = _sha256(
        bindings["predecessor_reveal_registry_sha256"],
        "candidate predecessor_reveal_registry_sha256",
    )
    return candidate_hash, design_hash, attempt_id, predecessor_hash


def candidate_design_sha256(candidate_manifest: Mapping[str, Any]) -> str:
    """Hash semantic design/data bytes without attempt or Git-tree provenance."""

    _, design_hash, _, _ = _validated_candidate_identity(candidate_manifest)
    return design_hash


def append_candidate_attempt(
    registry: Mapping[str, Any],
    *,
    external_prior_pin: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one never-before-registered frozen candidate to the chain.

    The prior registry must already be externally pinned.  The returned value
    is not authorized for outcome access until its new pin is itself stored
    through the caller's independent, immutable evidence path.
    """

    prior_registry_hash = validate_reveal_registry(
        registry, external_pin=external_prior_pin
    )
    validated_registry = _validate_registry_structure(registry)
    candidate_hash, candidate_design_hash, attempt_id, candidate_predecessor_hash = (
        _validated_candidate_identity(candidate_manifest)
    )
    entries = validated_registry["entries"]

    if any(entry["attempt_id"] == attempt_id for entry in entries):
        raise SecFilingGemmaRevealRegistryError(
            "Duplicate holdout attempt id is forbidden; exact reruns reuse its receipt"
        )
    if any(entry["candidate_sha256"] == candidate_hash for entry in entries):
        raise SecFilingGemmaRevealRegistryError(
            "A candidate hash may have only one holdout attempt"
        )
    if any(
        entry["candidate_design_sha256"] == candidate_design_hash
        for entry in entries
    ):
        raise SecFilingGemmaRevealRegistryError(
            "An identical frozen candidate design may have only one holdout attempt"
        )

    sequence = len(entries) + 1
    if sequence > 999:
        raise SecFilingGemmaRevealRegistryError(
            "Three-digit holdout attempt namespace is exhausted"
        )
    expected_attempt_id = f"{CONTRACT_VERSION}-attempt-{sequence:03d}"
    if attempt_id != expected_attempt_id:
        raise SecFilingGemmaRevealRegistryError(
            "Candidate attempt id must be the next contiguous registry id"
        )
    if not _same_digest(candidate_predecessor_hash, prior_registry_hash):
        raise SecFilingGemmaRevealRegistryError(
            "Candidate manifest did not bind the externally pinned predecessor registry"
        )

    prior_tip = validated_registry["chain"]["tip_sha256"]
    entry_body = _entry_body(
        sequence=sequence,
        attempt_id=attempt_id,
        candidate_sha256=candidate_hash,
        candidate_design_sha256=candidate_design_hash,
        candidate_manifest=candidate_manifest,
        candidate_bound_predecessor_registry_sha256=prior_registry_hash,
        prior_tip_sha256=prior_tip,
    )
    entry = {**entry_body, "entry_sha256": canonical_sha256(entry_body)}
    new_entries = copy.deepcopy(entries)
    new_entries.append(entry)
    return _registry_snapshot(new_entries, tip_sha256=entry["entry_sha256"])


def build_registry_pin_transition(
    prior_registry: Mapping[str, Any],
    *,
    external_prior_pin: Mapping[str, Any],
    appended_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact compare-and-swap transition an external store must apply.

    This receipt does not make itself authoritative.  The effectful caller must
    atomically compare its independently retained latest pin with
    ``expected_prior_pin`` and replace it with ``next_pin`` exactly once.  A
    stale or forked transition must lose that compare-and-swap operation.
    """

    validate_reveal_registry(prior_registry, external_pin=external_prior_pin)
    prior = _validate_registry_structure(prior_registry)
    appended = _validate_registry_structure(appended_registry)
    if len(appended["entries"]) != len(prior["entries"]) + 1:
        raise SecFilingGemmaRevealRegistryError(
            "A registry pin transition must append exactly one attempt"
        )
    if (
        appended["entries"][:-1] != prior["entries"]
        or canonical_sha256(appended["entries"][:-1])
        != canonical_sha256(prior["entries"])
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Registry transition rewrites or forks the pinned history"
        )
    appended_entry = appended["entries"][-1]
    if (
        appended_entry["candidate_bound_predecessor_registry_sha256"]
        != prior["registry_sha256"]
        or appended_entry["prior_tip_sha256"] != prior["chain"]["tip_sha256"]
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Appended attempt is not a child of the externally pinned tip"
        )

    expected_prior_pin = _validate_pin(external_prior_pin)
    next_pin = _pin_from_validated_registry(appended)
    body = {
        "schema_version": REGISTRY_PIN_TRANSITION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "expected_prior_pin": expected_prior_pin,
        "next_pin": next_pin,
        "appended_entry_sha256": appended_entry["entry_sha256"],
        "transition_semantics": (
            "external_atomic_compare_and_swap_of_authoritative_latest_pin"
        ),
        "stale_or_forked_prior_permitted": False,
        "derive_external_pin_from_presented_registry_permitted": False,
    }
    return {**body, "transition_sha256": canonical_sha256(body)}


def validate_registry_pin_transition(
    transition: Mapping[str, Any],
    *,
    prior_registry: Mapping[str, Any],
    external_prior_pin: Mapping[str, Any],
    appended_registry: Mapping[str, Any],
) -> str:
    """Validate an exact append transition before an external atomic CAS."""

    observed = _expect_mapping(transition, "registry pin transition")
    expected = build_registry_pin_transition(
        prior_registry,
        external_prior_pin=external_prior_pin,
        appended_registry=appended_registry,
    )
    _expect_keys(observed, set(expected), "registry pin transition")
    observed_hash = _sha256(
        observed["transition_sha256"], "transition_sha256"
    )
    observed_body = {
        key: observed[key] for key in observed if key != "transition_sha256"
    }
    if (
        observed != expected
        or canonical_sha256(observed_body) != expected["transition_sha256"]
        or not _same_digest(observed_hash, expected["transition_sha256"])
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Registry pin transition is stale, forked, or noncanonical"
        )
    return expected["transition_sha256"]


def build_single_candidate_reveal_request(
    registry: Mapping[str, Any],
    *,
    external_pin: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    stage: str,
    stage_access_manifest_sha256: str,
    prerequisite_stage_evidence_sha256: str,
) -> dict[str, Any]:
    """Build one non-authorizing reveal request at the current chain tip.

    The effectful stage gate must validate the prerequisite evidence
    semantically, independently load the latest pin, and atomically consume the
    request before opening any outcome.  This pure receipt never authorizes
    stage access by itself.
    """

    if stage not in _STAGE_PREREQUISITES:
        raise SecFilingGemmaRevealRegistryError(
            "Reveal request stage must be intermediate or final"
        )
    access_manifest_hash = _sha256(
        stage_access_manifest_sha256, "stage_access_manifest_sha256"
    )
    prerequisite_hash = _sha256(
        prerequisite_stage_evidence_sha256,
        "prerequisite_stage_evidence_sha256",
    )

    registry_hash = validate_reveal_registry(registry, external_pin=external_pin)
    validated_registry = _validate_registry_structure(registry)
    candidate_hash, candidate_design_hash, attempt_id, candidate_predecessor_hash = (
        _validated_candidate_identity(candidate_manifest)
    )
    entries = validated_registry["entries"]
    if not entries:
        raise SecFilingGemmaRevealRegistryError(
            "No candidate is registered for holdout access"
        )
    matching = [
        entry
        for entry in entries
        if entry["attempt_id"] == attempt_id
        or entry["candidate_sha256"] == candidate_hash
    ]
    if len(matching) != 1:
        raise SecFilingGemmaRevealRegistryError(
            "Candidate and attempt do not form one immutable registry entry"
        )
    entry = matching[0]
    if (
        entry["entry_sha256"] != entries[-1]["entry_sha256"]
        or entry["attempt_id"] != attempt_id
        or entry["candidate_sha256"] != candidate_hash
        or entry["candidate_design_sha256"] != candidate_design_hash
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Only the single precommitted candidate at the current tip may be revealed"
        )
    if not _same_digest(
        entry["candidate_bound_predecessor_registry_sha256"],
        candidate_predecessor_hash,
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Candidate-to-registry predecessor binding changed"
        )

    chain = validated_registry["chain"]
    body = {
        "schema_version": REVEAL_REQUEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "registry_sha256": registry_hash,
        "registry_tip_sha256": chain["tip_sha256"],
        "registered_entry_count": chain["registered_entry_count"],
        "historical_final_reveal_count_lower_bound": chain[
            "historical_final_reveal_count_lower_bound"
        ],
        "stage": stage,
        "stage_access_manifest_sha256": access_manifest_hash,
        "prerequisite_stage": _STAGE_PREREQUISITES[stage],
        "prerequisite_stage_evidence_sha256": prerequisite_hash,
        "attempt_id": attempt_id,
        "candidate_sha256": candidate_hash,
        "candidate_design_sha256": candidate_design_hash,
        "registry_entry_sha256": entry["entry_sha256"],
        "request_scope": "one_current_tip_candidate_and_one_stage_only",
        "authorizes_outcome_access": False,
        "effectful_atomic_single_use_consumption_required": True,
        "cross_attempt_comparison_permitted": False,
        "cross_attempt_winner_selection_permitted": False,
        "globally_pristine_claim": False,
    }
    return {**body, "request_sha256": canonical_sha256(body)}


def validate_single_candidate_reveal_request(
    request: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    external_pin: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    stage: str,
    stage_access_manifest_sha256: str,
    prerequisite_stage_evidence_sha256: str,
) -> str:
    """Reject altered, stale, multi-candidate, ranking, or winner requests."""

    observed = _expect_mapping(request, "reveal request")
    expected = build_single_candidate_reveal_request(
        registry,
        external_pin=external_pin,
        candidate_manifest=candidate_manifest,
        stage=stage,
        stage_access_manifest_sha256=stage_access_manifest_sha256,
        prerequisite_stage_evidence_sha256=prerequisite_stage_evidence_sha256,
    )
    _expect_keys(observed, set(expected), "reveal request")
    observed_hash = _sha256(
        observed["request_sha256"], "request_sha256"
    )
    observed_body = {
        key: observed[key] for key in observed if key != "request_sha256"
    }
    if (
        observed != expected
        or canonical_sha256(observed_body) != expected["request_sha256"]
        or not _same_digest(observed_hash, expected["request_sha256"])
    ):
        raise SecFilingGemmaRevealRegistryError(
            "Reveal request is stale, altered, or contains winner-picking semantics"
        )
    return expected["request_sha256"]


__all__ = [
    "CANDIDATE_DESIGN_SCHEMA_VERSION",
    "HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND",
    "HISTORICAL_REGISTRY_CANONICAL_SHA256",
    "REGISTRY_ENTRY_SCHEMA_VERSION",
    "REGISTRY_PIN_SCHEMA_VERSION",
    "REGISTRY_PIN_TRANSITION_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "REVEAL_REQUEST_SCHEMA_VERSION",
    "SecFilingGemmaRevealRegistryError",
    "append_candidate_attempt",
    "build_initial_reveal_registry",
    "build_registry_pin_transition",
    "build_single_candidate_reveal_request",
    "candidate_design_sha256",
    "derive_registry_pin",
    "historical_reveal_declaration",
    "validate_registry_pin_transition",
    "validate_reveal_registry",
    "validate_single_candidate_reveal_request",
]
