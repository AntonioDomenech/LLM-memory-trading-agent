"""Pure attempt and implementation bindings for the SEC/Gemma overlay v2.1.

This module deliberately performs no filesystem, Git, clock, network, market,
SEC, model, or database I/O.  It defines the only four effectful attempts,
binds an implementation to the frozen preregistration and source tree, and
constructs a one-shot transition chain that cannot return to an effect-capable
state after consumption.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from typing import Any, Final, Mapping, Sequence

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    ACQUISITION_TERMINAL_EVIDENCE_FIELDS,
    ACQUISITION_VALIDATION_CHECKS,
    ACQUISITION_VALIDATION_FIELDS,
    BRANCH_NAME,
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    EXTERNAL_PUBLICATION_FIELDS,
    FINAL_ATTEMPT_ID,
    FINAL_REGISTRY_AUTHORIZATION_FIELDS,
    SCORED_TERMINAL_EVIDENCE_FIELDS,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    build_contract_manifest,
    canonical_json_bytes,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    CONTRACT_SOURCE_PATH,
    EXPECTED_ORIGIN_URL,
    NUMERICAL_TIME_IMPORT_ROOTS,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    SOURCE_VERIFICATION_SCHEMA_VERSION,
    VerifiedSourceTree,
    is_verified_source_tree,
    source_verification_material,
)


IMPLEMENTATION_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-implementation-manifest-v2"
)
ATTEMPT_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-attempt-plan-v1"
)
ATTEMPT_TRANSITION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-attempt-transition-v1"
)
ACQUISITION_TERMINAL_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-acquisition-terminal-evidence-v1"
)
ACQUISITION_TERMINAL_EVIDENCE_VERIFIER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-acquisition-terminal-verifier-v1"
)
SCORED_TERMINAL_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-scored-terminal-evidence-v1"
)
SCORED_TERMINAL_EVIDENCE_VERIFIER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-scored-terminal-verifier-v1"
)
DETERMINISTIC_EVALUATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-deterministic-evaluation-v1"
)
JOINT_STAGE_REPORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-joint-stage-report-v1"
)

DEVELOPMENT_ACQUISITION: Final[str] = "development_acquisition"
DEVELOPMENT_SCORING: Final[str] = "development_scoring"
CONFIRMATION_SCORING: Final[str] = "confirmation_scoring"
FINAL_SCORING: Final[str] = "final_scoring"

ATTEMPT_KINDS: Final[tuple[str, ...]] = (
    DEVELOPMENT_ACQUISITION,
    DEVELOPMENT_SCORING,
    CONFIRMATION_SCORING,
    FINAL_SCORING,
)
ATTEMPT_IDS: Final[tuple[str, ...]] = (
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    CONFIRMATION_ATTEMPT_ID,
    FINAL_ATTEMPT_ID,
)
ATTEMPT_ID_BY_KIND: Final[dict[str, str]] = dict(
    zip(ATTEMPT_KINDS, ATTEMPT_IDS, strict=True)
)
ATTEMPT_KIND_BY_ID: Final[dict[str, str]] = {
    attempt_id: kind for kind, attempt_id in ATTEMPT_ID_BY_KIND.items()
}
ATTEMPT_ORDINAL_BY_ID: Final[dict[str, int]] = {
    attempt_id: index
    for index, attempt_id in enumerate(ATTEMPT_IDS, start=1)
}
PREREQUISITE_ATTEMPT_BY_ID: Final[dict[str, str | None]] = {
    DEVELOPMENT_ACQUISITION_ID: None,
    DEVELOPMENT_ATTEMPT_ID: DEVELOPMENT_ACQUISITION_ID,
    CONFIRMATION_ATTEMPT_ID: DEVELOPMENT_ATTEMPT_ID,
    FINAL_ATTEMPT_ID: CONFIRMATION_ATTEMPT_ID,
}

PLANNED: Final[str] = "planned"
CONSUMED: Final[str] = "consumed"
TERMINAL_PASS: Final[str] = "terminal_pass"
TERMINAL_FAIL: Final[str] = "terminal_fail"
TERMINAL_INDETERMINATE: Final[str] = "terminal_indeterminate"
ATTEMPT_STATUSES: Final[tuple[str, ...]] = (
    PLANNED,
    CONSUMED,
    TERMINAL_PASS,
    TERMINAL_FAIL,
    TERMINAL_INDETERMINATE,
)
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {TERMINAL_PASS, TERMINAL_FAIL, TERMINAL_INDETERMINATE}
)

REPORT_RECORD_TABLES: Final[tuple[str, ...]] = (
    "evidence",
    "features",
    "predictions",
    "lessons",
    "ledgers",
    "artifacts",
)

MINIMUM_PASS_RECORD_COUNTS: Final[dict[str, dict[str, int]]] = {
    DEVELOPMENT_ACQUISITION_ID: {
        "evidence": 1,
        "features": 0,
        "predictions": 0,
        "lessons": 0,
        "ledgers": 0,
        "artifacts": 1,
    },
    DEVELOPMENT_ATTEMPT_ID: {
        "evidence": 1,
        "features": 1,
        "predictions": 1,
        "lessons": 1,
        "ledgers": 1,
        "artifacts": 1,
    },
    CONFIRMATION_ATTEMPT_ID: {
        "evidence": 1,
        "features": 1,
        "predictions": 1,
        "lessons": 1,
        "ledgers": 1,
        "artifacts": 1,
    },
    FINAL_ATTEMPT_ID: {
        "evidence": 1,
        "features": 1,
        "predictions": 1,
        "lessons": 1,
        "ledgers": 1,
        "artifacts": 1,
    },
}

_ALLOWED_EFFECTS: Final[dict[str, tuple[str, ...]]] = {
    DEVELOPMENT_ACQUISITION_ID: (
        "official_sec_network",
        "market_network",
        "deterministic_private_quarantine",
    ),
    DEVELOPMENT_ATTEMPT_ID: (
        "ollama_runtime_identity",
        "gemma_batch",
        "canonical_market_value_read",
        "chronological_replay",
        "joint_report_seal",
    ),
    CONFIRMATION_ATTEMPT_ID: (
        "official_sec_network",
        "market_network",
        "ollama_runtime_identity",
        "gemma_batch",
        "canonical_market_value_read",
        "chronological_replay",
        "joint_report_seal",
    ),
    FINAL_ATTEMPT_ID: (
        "official_sec_network",
        "market_network",
        "ollama_runtime_identity",
        "gemma_batch",
        "canonical_market_value_read",
        "chronological_replay",
        "joint_report_seal",
    ),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_ROLE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SOURCE_PATH_RE = re.compile(
    r"agent_benchmark/[a-z0-9_]+(?:/[a-z0-9_]+)*\.py\Z"
)
_ATTEMPT_TRANSITION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "implementation_manifest_sha256",
        "attempt_plan_sha256",
        "attempt_id",
        "attempt_kind",
        "attempt_ordinal",
        "sequence_number",
        "prior_status",
        "status",
        "prior_transition_sha256",
        "terminal",
        "passed",
        "effects_permitted",
        "retry_permitted",
        "transition_sha256",
    }
)


class SecGemmaOnlineRiskOverlayAttemptError(ValueError):
    """An implementation, plan, or attempt transition is not canonical."""


def _plain_json(value: Any, location: str) -> Any:
    """Detach exact built-in finite JSON without invoking custom hooks."""

    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or key in result:
                raise SecGemmaOnlineRiskOverlayAttemptError(
                    f"{location} must have unique built-in string keys"
                )
            result[key] = _plain_json(child, f"{location}.{key}")
        return result
    if type(value) is list:
        return [
            _plain_json(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int, float}:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                f"{location} must contain only finite JSON"
            ) from exc
        return value
    raise SecGemmaOnlineRiskOverlayAttemptError(
        f"{location} must contain only exact built-in JSON values"
    )


def _mapping(value: Any, location: str) -> dict[str, Any]:
    detached = _plain_json(value, location)
    if type(detached) is not dict:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"{location} must be a mapping"
        )
    return detached


def _expect_keys(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _git_commit(value: Any, location: str) -> str:
    if type(value) is not str or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"{location} must be a full lowercase Git commit id"
        )
    return value


def _source_role(value: Any, location: str) -> str:
    if type(value) is not str or _SOURCE_ROLE_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"{location} must be a canonical source role"
        )
    return value


def _source_path(value: Any, location: str) -> str:
    if type(value) is not str or _SOURCE_PATH_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"{location} must be a canonical agent_benchmark Python path"
        )
    return value


def _self_hashed_body(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    location: str,
) -> dict[str, Any]:
    observed = _mapping(value, location)
    digest = _sha256(observed.get(hash_field), f"{location}.{hash_field}")
    body = {key: observed[key] for key in observed if key != hash_field}
    if not hmac.compare_digest(canonical_sha256(body), digest):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"{location} self-hash is inconsistent"
        )
    return observed


def _normalized_new_sources(
    value: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "new_sources must be a detached plain mapping"
        )
    normalized: list[dict[str, str]] = []
    paths: set[str] = set()
    for raw_role in sorted(value):
        role = _source_role(raw_role, "new source role")
        raw_spec = value[raw_role]
        if type(raw_spec) is not dict:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                f"new source {role} must be a detached mapping"
            )
        _expect_keys(raw_spec, {"path", "sha256"}, f"new source {role}")
        path = _source_path(raw_spec["path"], f"new source {role}.path")
        digest = _sha256(raw_spec["sha256"], f"new source {role}.sha256")
        if path in paths:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Two new source roles cannot alias one path"
            )
        paths.add(path)
        normalized.append({"role": role, "path": path, "sha256": digest})

    observed_by_role = {item["role"]: item["path"] for item in normalized}
    if observed_by_role != dict(REQUIRED_NEW_SOURCE_PATHS):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "new_sources must bind the exact production source inventory"
        )
    return normalized


def _reused_sources() -> list[dict[str, str]]:
    if set(SOURCE_PINS) != set(SOURCE_PIN_FILES):
        raise RuntimeError("Frozen reused source roles are inconsistent")
    return [
        {
            "role": role,
            "path": SOURCE_PIN_FILES[role],
            "sha256": SOURCE_PINS[role],
        }
        for role in sorted(SOURCE_PINS)
    ]


def _dependency_sources(value: Any) -> list[dict[str, str]]:
    if type(value) is not list:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "dependency_sources must be one exact list"
        )
    result: list[dict[str, str]] = []
    paths: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"dependency source {index}")
        _expect_keys(
            item, {"path", "sha256"}, f"dependency source {index}"
        )
        path = _source_path(item["path"], f"dependency source {index}.path")
        if path in paths:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Dependency source paths must be unique"
            )
        paths.add(path)
        result.append(
            {
                "path": path,
                "sha256": _sha256(
                    item["sha256"],
                    f"dependency source {index}.sha256",
                ),
            }
        )
    expected = sorted(result, key=lambda item: item["path"])
    if result != expected:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Dependency sources must be canonically ordered"
        )
    return result


def _external_distributions(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "external_distributions must be one exact list"
        )
    expected_keys = {
        "import_name",
        "distribution",
        "version",
        "module_files",
        "module_files_sha256",
        "direct_url_sha256",
    }
    result: list[dict[str, Any]] = []
    import_names: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"external distribution {index}")
        _expect_keys(item, expected_keys, f"external distribution {index}")
        for field in (
            "import_name",
            "distribution",
            "version",
        ):
            if type(item[field]) is not str or not item[field]:
                raise SecGemmaOnlineRiskOverlayAttemptError(
                    f"external distribution {index}.{field} is invalid"
                )
        if item["import_name"] in import_names:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "External distribution import names must be unique"
            )
        import_names.add(item["import_name"])
        raw_files = item["module_files"]
        if type(raw_files) is not list or not raw_files:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                f"external distribution {index}.module_files is invalid"
            )
        module_files: list[dict[str, str]] = []
        module_paths: set[str] = set()
        for file_index, raw_file in enumerate(raw_files):
            module_file = _mapping(
                raw_file,
                f"external distribution {index}.module_files[{file_index}]",
            )
            _expect_keys(
                module_file,
                {"path", "sha256"},
                (
                    f"external distribution {index}."
                    f"module_files[{file_index}]"
                ),
            )
            path = module_file["path"]
            if (
                type(path) is not str
                or not path
                or "\\" in path
                or ":" in path
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or path in module_paths
            ):
                raise SecGemmaOnlineRiskOverlayAttemptError(
                    f"external distribution {index}.module_files path is invalid"
                )
            module_paths.add(path)
            module_files.append(
                {
                    "path": path,
                    "sha256": _sha256(
                        module_file["sha256"],
                        (
                            f"external distribution {index}."
                            f"module_files[{file_index}].sha256"
                        ),
                    ),
                }
            )
        expected_files = sorted(module_files, key=lambda entry: entry["path"])
        if module_files != expected_files:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "External distribution module files must be canonically ordered"
            )
        module_files_sha256 = _sha256(
            item["module_files_sha256"],
            f"external distribution {index}.module_files_sha256",
        )
        if module_files_sha256 != canonical_sha256(module_files):
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "External distribution module file inventory hash changed"
            )
        direct_url = item["direct_url_sha256"]
        if direct_url is not None:
            direct_url = _sha256(
                direct_url,
                f"external distribution {index}.direct_url_sha256",
            )
        result.append(
            {
                "import_name": item["import_name"],
                "distribution": item["distribution"],
                "version": item["version"],
                "module_files": module_files,
                "module_files_sha256": module_files_sha256,
                "direct_url_sha256": direct_url,
            }
        )
    expected = sorted(result, key=lambda item: item["import_name"])
    if result != expected:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "External distributions must be canonically ordered"
        )
    return result


def _numerical_time_distributions(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "numerical_time_distributions must be one exact list"
        )
    expected_keys = {
        "import_name",
        "distribution",
        "version",
        "module_origin",
        "runtime_files",
        "runtime_files_sha256",
        "direct_url_sha256",
    }
    result: list[dict[str, Any]] = []
    import_names: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"numerical/time distribution {index}")
        _expect_keys(
            item,
            expected_keys,
            f"numerical/time distribution {index}",
        )
        for field in ("import_name", "distribution", "version"):
            if type(item[field]) is not str or not item[field]:
                raise SecGemmaOnlineRiskOverlayAttemptError(
                    f"numerical/time distribution {index}.{field} is invalid"
                )
        import_name = item["import_name"]
        if import_name in import_names:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Numerical/time distribution import names must be unique"
            )
        import_names.add(import_name)
        raw_files = item["runtime_files"]
        if type(raw_files) is not list or not raw_files:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                f"numerical/time distribution {index}.runtime_files is invalid"
            )
        runtime_files: list[dict[str, str]] = []
        runtime_paths: set[str] = set()
        for file_index, raw_file in enumerate(raw_files):
            runtime_file = _mapping(
                raw_file,
                (
                    f"numerical/time distribution {index}."
                    f"runtime_files[{file_index}]"
                ),
            )
            _expect_keys(
                runtime_file,
                {"path", "sha256"},
                (
                    f"numerical/time distribution {index}."
                    f"runtime_files[{file_index}]"
                ),
            )
            path = runtime_file["path"]
            if (
                type(path) is not str
                or not path
                or "\\" in path
                or ":" in path
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or path in runtime_paths
            ):
                raise SecGemmaOnlineRiskOverlayAttemptError(
                    "Numerical/time runtime file path is invalid"
                )
            runtime_paths.add(path)
            runtime_files.append(
                {
                    "path": path,
                    "sha256": _sha256(
                        runtime_file["sha256"],
                        (
                            f"numerical/time distribution {index}."
                            f"runtime_files[{file_index}].sha256"
                        ),
                    ),
                }
            )
        expected_files = sorted(
            runtime_files, key=lambda entry: entry["path"]
        )
        if runtime_files != expected_files:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Numerical/time runtime files must be canonically ordered"
            )
        runtime_files_sha256 = _sha256(
            item["runtime_files_sha256"],
            (
                f"numerical/time distribution {index}."
                "runtime_files_sha256"
            ),
        )
        if runtime_files_sha256 != canonical_sha256(runtime_files):
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Numerical/time runtime file inventory hash changed"
            )
        module_origin = _mapping(
            item["module_origin"],
            f"numerical/time distribution {index}.module_origin",
        )
        _expect_keys(
            module_origin,
            {"path", "sha256"},
            f"numerical/time distribution {index}.module_origin",
        )
        origin_path = module_origin["path"]
        origin_sha256 = _sha256(
            module_origin["sha256"],
            f"numerical/time distribution {index}.module_origin.sha256",
        )
        if (
            type(origin_path) is not str
            or origin_path not in runtime_paths
            or next(
                entry["sha256"]
                for entry in runtime_files
                if entry["path"] == origin_path
            )
            != origin_sha256
        ):
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Numerical/time active module origin is outside its runtime "
                "file inventory"
            )
        direct_url = item["direct_url_sha256"]
        if direct_url is not None:
            direct_url = _sha256(
                direct_url,
                (
                    f"numerical/time distribution {index}."
                    "direct_url_sha256"
                ),
            )
        result.append(
            {
                "import_name": import_name,
                "distribution": item["distribution"],
                "version": item["version"],
                "module_origin": {
                    "path": origin_path,
                    "sha256": origin_sha256,
                },
                "runtime_files": runtime_files,
                "runtime_files_sha256": runtime_files_sha256,
                "direct_url_sha256": direct_url,
            }
        )
    expected = sorted(result, key=lambda item: item["import_name"])
    if result != expected:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Numerical/time distributions must be canonically ordered"
        )
    if import_names != set(NUMERICAL_TIME_IMPORT_ROOTS):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Numerical/time distribution roots differ from the verified "
            "runtime closure"
        )
    return result


def build_implementation_manifest(
    *,
    verified_sources: VerifiedSourceTree,
) -> dict[str, Any]:
    """Bind a manifest only from opaque, live-verified repository evidence."""

    if not is_verified_source_tree(verified_sources):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation manifest requires live-verified source evidence"
        )
    evidence = source_verification_material(verified_sources)
    if (
        evidence["schema_version"] != SOURCE_VERIFICATION_SCHEMA_VERSION
        or evidence["branch"] != BRANCH_NAME
        or evidence["origin_url"] != EXPECTED_ORIGIN_URL
        or evidence["preregistration_commit"] != PREREGISTRATION_COMMIT
        or evidence["head_commit"] != evidence["upstream_commit"]
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Verified source evidence differs from the preregistration"
        )
    implementation = _git_commit(
        evidence["head_commit"], "verified source head_commit"
    )
    upstream = _git_commit(
        evidence["upstream_commit"], "verified source upstream_commit"
    )
    contract_source = _mapping(
        evidence["contract_source"], "verified contract source"
    )
    if contract_source.get("path") != CONTRACT_SOURCE_PATH:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Verified contract source path changed"
        )
    contract_source_hash = _sha256(
        contract_source.get("sha256"), "verified contract source sha256"
    )
    supplied_new = {
        item["role"]: {"path": item["path"], "sha256": item["sha256"]}
        for item in evidence["new_sources"]
    }
    normalized_new = _normalized_new_sources(supplied_new)
    reused = evidence["reused_sources"]
    if reused != _reused_sources():
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Verified reused sources differ from the frozen source pins"
        )
    dependencies = _dependency_sources(evidence["dependency_sources"])
    external = _external_distributions(evidence["external_distributions"])
    numerical_time = _numerical_time_distributions(
        evidence["numerical_time_distributions"]
    )
    dependency_material = {
        "dependency_sources": dependencies,
        "external_distributions": external,
        "numerical_time_distributions": numerical_time,
    }
    dependency_closure_sha256 = _sha256(
        evidence["dependency_closure_sha256"],
        "verified dependency closure sha256",
    )
    if dependency_closure_sha256 != canonical_sha256(dependency_material):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Verified dependency closure hash is inconsistent"
        )
    source_tree = {
        "contract_source": {
            "path": CONTRACT_SOURCE_PATH,
            "sha256": contract_source_hash,
        },
        "reused_sources": reused,
        "new_sources": normalized_new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure_sha256,
    }
    body = {
        "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "contract_source": copy.deepcopy(source_tree["contract_source"]),
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "implementation_commit": implementation,
        "branch": BRANCH_NAME,
        "upstream_ref": f"origin/{BRANCH_NAME}",
        "upstream_commit": upstream,
        "origin_url": EXPECTED_ORIGIN_URL,
        "clean_tracked_tree": True,
        "head_matches_upstream": True,
        "preregistration_is_ancestor": True,
        "reused_sources": copy.deepcopy(reused),
        "new_sources": copy.deepcopy(normalized_new),
        "dependency_sources": copy.deepcopy(dependencies),
        "external_distributions": copy.deepcopy(external),
        "numerical_time_distributions": copy.deepcopy(numerical_time),
        "dependency_closure_sha256": dependency_closure_sha256,
        "source_tree_sha256": canonical_sha256(source_tree),
        "source_verification_sha256": verified_sources.verification_sha256,
        "effects_permitted": False,
    }
    return {
        **body,
        "implementation_manifest_sha256": canonical_sha256(body),
    }


def validate_implementation_manifest(value: Any) -> dict[str, Any]:
    """Accept only a complete, exact implementation binding."""

    observed = _self_hashed_body(
        value,
        hash_field="implementation_manifest_sha256",
        location="implementation manifest",
    )
    _expect_keys(
        observed,
        {
            "schema_version",
            "contract_version",
            "contract_sha256",
            "contract_source",
            "preregistration_commit",
            "implementation_commit",
            "branch",
            "upstream_ref",
            "upstream_commit",
            "origin_url",
            "clean_tracked_tree",
            "head_matches_upstream",
            "preregistration_is_ancestor",
            "reused_sources",
            "new_sources",
            "dependency_sources",
            "external_distributions",
            "numerical_time_distributions",
            "dependency_closure_sha256",
            "source_tree_sha256",
            "source_verification_sha256",
            "effects_permitted",
            "implementation_manifest_sha256",
        },
        "implementation manifest",
    )
    if (
        observed["schema_version"] != IMPLEMENTATION_MANIFEST_SCHEMA_VERSION
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["preregistration_commit"] != PREREGISTRATION_COMMIT
        or observed["branch"] != BRANCH_NAME
        or observed["upstream_ref"] != f"origin/{BRANCH_NAME}"
        or observed["origin_url"] != EXPECTED_ORIGIN_URL
        or observed["clean_tracked_tree"] is not True
        or observed["head_matches_upstream"] is not True
        or observed["preregistration_is_ancestor"] is not True
        or observed["effects_permitted"] is not False
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation manifest is not bound to the preregistered clean branch"
        )
    implementation = _git_commit(
        observed["implementation_commit"],
        "implementation manifest implementation_commit",
    )
    upstream = _git_commit(
        observed["upstream_commit"],
        "implementation manifest upstream_commit",
    )
    if implementation != upstream:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation manifest HEAD differs from upstream"
        )
    contract_source = _mapping(
        observed["contract_source"], "implementation contract source"
    )
    _expect_keys(
        contract_source,
        {"path", "sha256"},
        "implementation contract source",
    )
    if contract_source["path"] != CONTRACT_SOURCE_PATH:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation contract source path changed"
        )
    _sha256(
        contract_source["sha256"],
        "implementation contract source sha256",
    )

    reused = observed["reused_sources"]
    if type(reused) is not list or reused != _reused_sources():
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation reused source pins differ from the contract"
        )
    raw_new = observed["new_sources"]
    if type(raw_new) is not list:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation new_sources must be a list"
        )
    supplied_new: dict[str, dict[str, str]] = {}
    for index, raw_spec in enumerate(raw_new):
        spec = _mapping(raw_spec, f"implementation new source {index}")
        _expect_keys(
            spec,
            {"role", "path", "sha256"},
            f"implementation new source {index}",
        )
        role = _source_role(spec["role"], f"implementation new source {index}.role")
        if role in supplied_new:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Implementation duplicates a new source role"
            )
        supplied_new[role] = {
            "path": spec["path"],
            "sha256": spec["sha256"],
        }
    normalized_new = _normalized_new_sources(supplied_new)
    if raw_new != normalized_new:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation new sources are not canonically ordered"
        )
    dependencies = _dependency_sources(observed["dependency_sources"])
    external = _external_distributions(observed["external_distributions"])
    numerical_time = _numerical_time_distributions(
        observed["numerical_time_distributions"]
    )
    dependency_material = {
        "dependency_sources": dependencies,
        "external_distributions": external,
        "numerical_time_distributions": numerical_time,
    }
    dependency_closure_sha256 = _sha256(
        observed["dependency_closure_sha256"],
        "implementation dependency closure sha256",
    )
    if dependency_closure_sha256 != canonical_sha256(dependency_material):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation dependency closure hash is inconsistent"
        )
    source_tree = {
        "contract_source": contract_source,
        "reused_sources": reused,
        "new_sources": normalized_new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure_sha256,
    }
    if not hmac.compare_digest(
        _sha256(
            observed["source_tree_sha256"],
            "implementation source_tree_sha256",
        ),
        canonical_sha256(source_tree),
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation source tree hash is inconsistent"
        )
    verification_hash = _sha256(
        observed["source_verification_sha256"],
        "implementation source_verification_sha256",
    )
    verification_material = {
        "schema_version": SOURCE_VERIFICATION_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "branch": BRANCH_NAME,
        "origin_url": EXPECTED_ORIGIN_URL,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "head_commit": implementation,
        "upstream_commit": upstream,
        "contract_source": copy.deepcopy(contract_source),
        "reused_sources": copy.deepcopy(reused),
        "new_sources": copy.deepcopy(normalized_new),
        "dependency_sources": copy.deepcopy(dependencies),
        "external_distributions": copy.deepcopy(external),
        "numerical_time_distributions": copy.deepcopy(numerical_time),
        "dependency_closure_sha256": dependency_closure_sha256,
    }
    if not hmac.compare_digest(
        verification_hash, canonical_sha256(verification_material)
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation source verification hash is inconsistent"
        )
    body = {
        "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "contract_source": copy.deepcopy(contract_source),
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "implementation_commit": implementation,
        "branch": BRANCH_NAME,
        "upstream_ref": f"origin/{BRANCH_NAME}",
        "upstream_commit": upstream,
        "origin_url": EXPECTED_ORIGIN_URL,
        "clean_tracked_tree": True,
        "head_matches_upstream": True,
        "preregistration_is_ancestor": True,
        "reused_sources": copy.deepcopy(reused),
        "new_sources": copy.deepcopy(normalized_new),
        "dependency_sources": copy.deepcopy(dependencies),
        "external_distributions": copy.deepcopy(external),
        "numerical_time_distributions": copy.deepcopy(numerical_time),
        "dependency_closure_sha256": dependency_closure_sha256,
        "source_tree_sha256": canonical_sha256(source_tree),
        "source_verification_sha256": observed[
            "source_verification_sha256"
        ],
        "effects_permitted": False,
    }
    expected = {
        **body,
        "implementation_manifest_sha256": canonical_sha256(body),
    }
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Implementation manifest differs from its canonical reconstruction"
        )
    return copy.deepcopy(expected)


def _attempt_id(value: Any) -> str:
    if type(value) is not str or value not in ATTEMPT_KIND_BY_ID:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "attempt_id is not one of the four preregistered attempts"
        )
    return value


def _optional_sha256(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, location)


def _final_registry_authorization_material(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from agent_benchmark.sec_gemma_online_risk_overlay_registry import (
            FINAL_REGISTRY_AUTHORIZATION_SCHEMA_VERSION,
            FINAL_REGISTRY_VERIFIER_ID,
            is_verified_final_registry_authorization,
            validate_final_registry_authorization,
        )
    except ImportError as exc:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Final registry authorization authority is unavailable"
        ) from exc
    if not is_verified_final_registry_authorization(value):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Final attempt requires opaque verified registry authorization"
        )
    observed = _mapping(
        getattr(value, "authorization", None),
        "final registry authorization",
    )
    _expect_keys(
        observed,
        set(FINAL_REGISTRY_AUTHORIZATION_FIELDS),
        "final registry authorization",
    )
    for field in (
        "predecessor_registry_sha256",
        "predecessor_registry_tip_sha256",
        "successor_entry_sha256",
        "successor_registry_sha256",
        "external_publication_sha256",
        "authorization_sha256",
    ):
        _sha256(observed[field], f"final registry authorization {field}")
    if (
        observed["schema_version"]
        != FINAL_REGISTRY_AUTHORIZATION_SCHEMA_VERSION
        or not observed["schema_version"].startswith(f"{CONTRACT_VERSION}-")
        or observed["verifier_id"] != FINAL_REGISTRY_VERIFIER_ID
        or not observed["verifier_id"].startswith(f"{CONTRACT_VERSION}-")
        or type(observed["predecessor_reveal_count"]) is not int
        or observed["predecessor_reveal_count"] < 0
        or observed["final_attempt_id"] != FINAL_ATTEMPT_ID
        or observed["authorization_sha256"]
        != canonical_sha256(
            {
                key: observed[key]
                for key in observed
                if key != "authorization_sha256"
            }
        )
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Final registry authorization is not canonical v2.1 evidence"
        )
    try:
        return validate_final_registry_authorization(
            observed,
            successor=value.successor,
            external_publication=value.external_publication,
            implementation_manifest=implementation_manifest,
        )
    except Exception as exc:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Final registry authorization failed exact canonical validation"
        ) from exc


def build_attempt_plan(
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    prerequisite_terminal_transition: Mapping[str, Any] | None = None,
    final_registry_authorization: Any | None = None,
) -> dict[str, Any]:
    """Build the only effect plan for one of the four fixed attempts."""

    implementation = validate_implementation_manifest(implementation_manifest)
    fixed_attempt_id = _attempt_id(attempt_id)
    prerequisite_attempt_id = PREREQUISITE_ATTEMPT_BY_ID[fixed_attempt_id]
    prerequisite_transition_hash: str | None = None
    if prerequisite_attempt_id is None:
        if prerequisite_terminal_transition is not None:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Development acquisition cannot claim a prerequisite transition"
            )
    else:
        if prerequisite_terminal_transition is None:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Later attempts require the exact predecessor terminal pass"
            )
        predecessor = _self_hashed_body(
            prerequisite_terminal_transition,
            hash_field="transition_sha256",
            location="prerequisite terminal transition",
        )
        _expect_keys(
            predecessor,
            set(_ATTEMPT_TRANSITION_KEYS),
            "prerequisite terminal transition",
        )
        predecessor_hash = _sha256(
            predecessor.get("transition_sha256"),
            "prerequisite terminal transition hash",
        )
        _sha256(
            predecessor.get("attempt_plan_sha256"),
            "prerequisite attempt plan hash",
        )
        _sha256(
            predecessor.get("prior_transition_sha256"),
            "prerequisite prior transition hash",
        )
        if (
            predecessor.get("schema_version")
            != ATTEMPT_TRANSITION_SCHEMA_VERSION
            or predecessor.get("contract_version") != CONTRACT_VERSION
            or predecessor.get("contract_sha256") != CONTRACT_SHA256
            or predecessor.get("implementation_manifest_sha256")
            != implementation["implementation_manifest_sha256"]
            or predecessor.get("attempt_id") != prerequisite_attempt_id
            or predecessor.get("attempt_kind")
            != ATTEMPT_KIND_BY_ID[prerequisite_attempt_id]
            or predecessor.get("attempt_ordinal")
            != ATTEMPT_ORDINAL_BY_ID[prerequisite_attempt_id]
            or predecessor.get("sequence_number") != 3
            or predecessor.get("prior_status") != CONSUMED
            or predecessor.get("status") != TERMINAL_PASS
            or predecessor.get("terminal") is not True
            or predecessor.get("passed") is not True
            or predecessor.get("effects_permitted") is not False
            or predecessor.get("retry_permitted") is not False
        ):
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Attempt prerequisite is not the exact predecessor terminal pass"
            )
        prerequisite_transition_hash = predecessor_hash

    final_authorization: dict[str, Any] | None = None
    if fixed_attempt_id == FINAL_ATTEMPT_ID:
        if final_registry_authorization is None:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Final scoring requires opaque registry authorization"
            )
        final_authorization = _final_registry_authorization_material(
            final_registry_authorization,
            implementation_manifest=implementation,
        )
    elif final_registry_authorization is not None:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Only final scoring may bind registry authorization"
        )

    ordinal = ATTEMPT_ORDINAL_BY_ID[fixed_attempt_id]
    body = {
        "schema_version": ATTEMPT_PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation[
            "implementation_manifest_sha256"
        ],
        "attempt_id": fixed_attempt_id,
        "attempt_kind": ATTEMPT_KIND_BY_ID[fixed_attempt_id],
        "attempt_ordinal": ordinal,
        "prerequisite_attempt_id": prerequisite_attempt_id,
        "prerequisite_terminal_transition_sha256": (
            prerequisite_transition_hash
        ),
        "final_registry_authorization_sha256": (
            None
            if final_authorization is None
            else final_authorization["authorization_sha256"]
        ),
        "final_registry_successor_sha256": (
            None
            if final_authorization is None
            else final_authorization["successor_registry_sha256"]
        ),
        "final_registry_external_publication_sha256": (
            None
            if final_authorization is None
            else final_authorization["external_publication_sha256"]
        ),
        "allowed_effects": list(_ALLOWED_EFFECTS[fixed_attempt_id]),
        "one_shot": True,
        "retry_permitted": False,
        "cross_attempt_winner_selection_permitted": False,
        "effects_permitted": False,
    }
    return {**body, "attempt_plan_sha256": canonical_sha256(body)}


def validate_attempt_plan(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _self_hashed_body(
        value,
        hash_field="attempt_plan_sha256",
        location="attempt plan",
    )
    _expect_keys(
        observed,
        {
            "schema_version",
            "contract_version",
            "contract_sha256",
            "implementation_manifest_sha256",
            "attempt_id",
            "attempt_kind",
            "attempt_ordinal",
            "prerequisite_attempt_id",
            "prerequisite_terminal_transition_sha256",
            "final_registry_authorization_sha256",
            "final_registry_successor_sha256",
            "final_registry_external_publication_sha256",
            "allowed_effects",
            "one_shot",
            "retry_permitted",
            "cross_attempt_winner_selection_permitted",
            "effects_permitted",
            "attempt_plan_sha256",
        },
        "attempt plan",
    )
    implementation = validate_implementation_manifest(implementation_manifest)
    attempt_id = _attempt_id(observed["attempt_id"])
    ordinal = ATTEMPT_ORDINAL_BY_ID[attempt_id]
    prerequisite_id = PREREQUISITE_ATTEMPT_BY_ID[attempt_id]
    prerequisite_hash = _optional_sha256(
        observed["prerequisite_terminal_transition_sha256"],
        "attempt plan prerequisite transition hash",
    )
    final_authorization_hash = _optional_sha256(
        observed["final_registry_authorization_sha256"],
        "attempt plan final registry authorization",
    )
    final_successor_hash = _optional_sha256(
        observed["final_registry_successor_sha256"],
        "attempt plan final registry successor",
    )
    final_publication_hash = _optional_sha256(
        observed["final_registry_external_publication_sha256"],
        "attempt plan final registry external publication",
    )
    if (
        observed["schema_version"] != ATTEMPT_PLAN_SCHEMA_VERSION
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["implementation_manifest_sha256"]
        != implementation["implementation_manifest_sha256"]
        or observed["attempt_kind"] != ATTEMPT_KIND_BY_ID[attempt_id]
        or type(observed["attempt_ordinal"]) is not int
        or observed["attempt_ordinal"] != ordinal
        or observed["prerequisite_attempt_id"] != prerequisite_id
        or observed["allowed_effects"] != list(_ALLOWED_EFFECTS[attempt_id])
        or observed["one_shot"] is not True
        or observed["retry_permitted"] is not False
        or observed["cross_attempt_winner_selection_permitted"] is not False
        or observed["effects_permitted"] is not False
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Attempt plan differs from the fixed stage contract"
        )
    if (prerequisite_id is None) != (prerequisite_hash is None):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Attempt plan prerequisite binding is incomplete"
        )
    final_hashes = (
        final_authorization_hash,
        final_successor_hash,
        final_publication_hash,
    )
    if (attempt_id == FINAL_ATTEMPT_ID) != all(
        value is not None for value in final_hashes
    ) or (
        attempt_id != FINAL_ATTEMPT_ID
        and any(value is not None for value in final_hashes)
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Attempt plan final registry binding is invalid"
        )
    body = {
        key: observed[key]
        for key in observed
        if key != "attempt_plan_sha256"
    }
    if canonical_sha256(body) != observed["attempt_plan_sha256"]:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Attempt plan hash changed"
        )
    return copy.deepcopy(observed)


def build_attempt_transition(
    *,
    attempt_plan: Mapping[str, Any],
    implementation_manifest: Mapping[str, Any],
    status: str,
    prior_transition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance exactly once through planned, consumed, and one terminal state."""

    plan = validate_attempt_plan(
        attempt_plan,
        implementation_manifest=implementation_manifest,
    )
    if type(status) is not str or status not in ATTEMPT_STATUSES:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Attempt transition status is invalid"
        )
    if prior_transition is None:
        if status != PLANNED:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "The first attempt transition must be planned"
            )
        sequence = 1
        prior_status: str | None = None
        prior_hash: str | None = None
    else:
        prior = _self_hashed_body(
            prior_transition,
            hash_field="transition_sha256",
            location="prior attempt transition",
        )
        _expect_keys(
            prior,
            set(_ATTEMPT_TRANSITION_KEYS),
            "prior attempt transition",
        )
        if (
            prior.get("attempt_id") != plan["attempt_id"]
            or prior.get("attempt_plan_sha256") != plan["attempt_plan_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Attempt transition crossed its plan or identity"
            )
        prior_status = prior.get("status")
        prior_hash = _sha256(
            prior.get("transition_sha256"),
            "prior attempt transition hash",
        )
        prior_sequence = prior.get("sequence_number")
        if type(prior_sequence) is not int or prior_sequence < 1:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Prior attempt transition sequence is invalid"
            )
        if prior_status == PLANNED:
            expected_prior = build_attempt_transition(
                attempt_plan=plan,
                implementation_manifest=implementation_manifest,
                status=PLANNED,
            )
        elif prior_status == CONSUMED:
            expected_planned = build_attempt_transition(
                attempt_plan=plan,
                implementation_manifest=implementation_manifest,
                status=PLANNED,
            )
            expected_prior = build_attempt_transition(
                attempt_plan=plan,
                implementation_manifest=implementation_manifest,
                status=CONSUMED,
                prior_transition=expected_planned,
            )
        else:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "A terminal attempt can never transition or retry"
            )
        if prior != expected_prior:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Prior attempt transition differs from the one-shot chain"
            )
        sequence = prior_sequence + 1
        if prior_status == PLANNED:
            if status != CONSUMED:
                raise SecGemmaOnlineRiskOverlayAttemptError(
                    "A planned attempt may only become consumed"
                )
        elif prior_status == CONSUMED:
            if status not in TERMINAL_STATUSES:
                raise SecGemmaOnlineRiskOverlayAttemptError(
                    "A consumed attempt may only become terminal"
                )
        else:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "A terminal attempt can never transition or retry"
            )
    terminal = status in TERMINAL_STATUSES
    passed: bool | None = True if status == TERMINAL_PASS else (
        False if terminal else None
    )
    body = {
        "schema_version": ATTEMPT_TRANSITION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": plan[
            "implementation_manifest_sha256"
        ],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_ordinal": plan["attempt_ordinal"],
        "sequence_number": sequence,
        "prior_status": prior_status,
        "status": status,
        "prior_transition_sha256": prior_hash,
        "terminal": terminal,
        "passed": passed,
        "effects_permitted": status == CONSUMED,
        "retry_permitted": False,
    }
    return {**body, "transition_sha256": canonical_sha256(body)}


def validate_attempt_history(
    *,
    attempt_plan: Mapping[str, Any],
    implementation_manifest: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay one complete or in-progress attempt transition chain."""

    if isinstance(transitions, (str, bytes)) or not isinstance(
        transitions, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Attempt history must be a sequence"
        )
    if not transitions or len(transitions) > 3:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Attempt history must contain one to three transitions"
        )
    expected_prior: dict[str, Any] | None = None
    validated: list[dict[str, Any]] = []
    for index, raw_transition in enumerate(transitions):
        observed = _self_hashed_body(
            raw_transition,
            hash_field="transition_sha256",
            location=f"attempt transition {index + 1}",
        )
        _expect_keys(
            observed,
            set(_ATTEMPT_TRANSITION_KEYS),
            f"attempt transition {index + 1}",
        )
        expected = build_attempt_transition(
            attempt_plan=attempt_plan,
            implementation_manifest=implementation_manifest,
            status=observed["status"],
            prior_transition=expected_prior,
        )
        if observed != expected:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Attempt transition differs from the one-shot chain"
            )
        validated.append(copy.deepcopy(expected))
        expected_prior = expected
    return validated


def current_attempt_status(
    *,
    attempt_plan: Mapping[str, Any],
    implementation_manifest: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
) -> str:
    return validate_attempt_history(
        attempt_plan=attempt_plan,
        implementation_manifest=implementation_manifest,
        transitions=transitions,
    )[-1]["status"]


_STORE_RECORD_RECEIPT_FIELDS: Final[tuple[str, ...]] = (
    "table",
    "identity",
    "attempt_id",
    "payload_sha256",
    "journal_sequence",
    "journal_entry_sha256",
)
_RECEIPT_IDENTITY_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z"
)
_SCORING_STAGE_BY_KIND: Final[dict[str, str]] = {
    DEVELOPMENT_SCORING: "development",
    CONFIRMATION_SCORING: "confirmation",
    FINAL_SCORING: "final",
}
_CONTRACT_GATES: Final[dict[str, tuple[str, ...]]] = {
    stage: tuple(build_contract_manifest()["gates"][stage])
    for stage in ("development", "confirmation", "final")
}
_VERIFIED_ACQUISITION_TERMINAL_EVIDENCE_SENTINEL = object()
_VERIFIED_SCORED_TERMINAL_EVIDENCE_SENTINEL = object()


def _record_counts(value: Any, location: str) -> dict[str, int]:
    observed = _mapping(value, location)
    _expect_keys(observed, set(REPORT_RECORD_TABLES), location)
    result: dict[str, int] = {}
    for table in REPORT_RECORD_TABLES:
        count = observed[table]
        if type(count) is not int or count < 0:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                f"{location}.{table} must be a nonnegative integer"
            )
        result[table] = count
    return result


def _report_material(value: Any) -> dict[str, Any]:
    observed = _mapping(value, "terminal evidence material")
    _expect_keys(
        observed,
        {
            "attempt_id",
            "attempt_plan_sha256",
            "record_counts",
            "record_commitment_sha256",
        },
        "terminal evidence material",
    )
    return {
        "attempt_id": _attempt_id(observed["attempt_id"]),
        "attempt_plan_sha256": _sha256(
            observed["attempt_plan_sha256"],
            "terminal evidence material attempt_plan_sha256",
        ),
        "record_counts": _record_counts(
            observed["record_counts"],
            "terminal evidence material record_counts",
        ),
        "record_commitment_sha256": _sha256(
            observed["record_commitment_sha256"],
            "terminal evidence material record commitment",
        ),
    }


def store_record_receipt_material(value: Any) -> dict[str, Any]:
    """Detach one exact store-issued receipt; mappings and subclasses fail."""

    try:
        from agent_benchmark.sec_gemma_online_risk_overlay_store import (
            StoreRecordReceipt,
        )
    except ImportError as exc:  # pragma: no cover - import cycle guard
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Store receipt authority is unavailable"
        ) from exc
    if type(value) is not StoreRecordReceipt:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Terminal evidence requires an exact store-issued record receipt"
        )
    material = {
        field: getattr(value, field) for field in _STORE_RECORD_RECEIPT_FIELDS
    }
    if (
        type(material["table"]) is not str
        or type(material["identity"]) is not str
        or _RECEIPT_IDENTITY_RE.fullmatch(material["identity"]) is None
        or type(material["attempt_id"]) is not str
        or material["attempt_id"] not in ATTEMPT_KIND_BY_ID
        or type(material["journal_sequence"]) is not int
        or material["journal_sequence"] < 1
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Store record receipt identity or sequence is invalid"
        )
    _sha256(
        material["payload_sha256"],
        "store record receipt payload_sha256",
    )
    _sha256(
        material["journal_entry_sha256"],
        "store record receipt journal_entry_sha256",
    )
    return material


def store_record_receipt_sha256(value: Any) -> str:
    return canonical_sha256(store_record_receipt_material(value))


def _publication_material(value: Any) -> dict[str, Any]:
    """Require the publisher's exact opaque v2.1 receipt."""

    try:
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            EXTERNAL_PUBLICATION_SCHEMA_VERSION,
            EXTERNAL_PUBLISHER_ID,
            is_verified_external_publication,
        )
    except ImportError as exc:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "External publication authority is unavailable"
        ) from exc
    if not is_verified_external_publication(value):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Terminal evidence requires an opaque verified external publication"
        )
    raw = getattr(value, "publication", None)
    observed = _mapping(raw, "external publication")
    _expect_keys(
        observed,
        set(EXTERNAL_PUBLICATION_FIELDS),
        "external publication",
    )
    publication_hash = _sha256(
        observed["publication_sha256"],
        "external publication publication_sha256",
    )
    body = {
        key: observed[key]
        for key in observed
        if key != "publication_sha256"
    }
    if (
        observed["schema_version"] != EXTERNAL_PUBLICATION_SCHEMA_VERSION
        or observed["publisher_id"] != EXTERNAL_PUBLISHER_ID
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or canonical_sha256(body) != publication_hash
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "External publication is not the canonical v2.1 receipt"
        )
    _git_commit(
        observed["implementation_commit"],
        "external publication implementation_commit",
    )
    _git_commit(
        observed["tag_target_commit"],
        "external publication tag_target_commit",
    )
    _git_commit(
        observed["remote_peeled_commit"],
        "external publication remote_peeled_commit",
    )
    _git_commit(
        observed["remote_tag_object_sha1"],
        "external publication remote_tag_object_sha1",
    )
    _sha256(
        observed["artifact_sha256"],
        "external publication artifact_sha256",
    )
    predecessor = observed["predecessor_publication_sha256"]
    if predecessor is not None:
        _sha256(
            predecessor,
            "external publication predecessor_publication_sha256",
        )
    for field in ("tag_ref", "remote_name", "remote_url"):
        if type(observed[field]) is not str or not observed[field]:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                f"External publication {field} is invalid"
            )
    _sha256(
        observed["tag_message_sha256"],
        "external publication tag_message_sha256",
    )
    cost = observed["external_cost_usd"]
    if type(cost) not in {int, float} or cost != 0:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "External publication must report exactly zero external cost"
        )
    return observed


def _validate_publication_binding(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    publication = _publication_material(value)
    try:
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            validate_external_publication,
        )

        publication = validate_external_publication(
            publication,
            implementation_manifest=implementation_manifest,
        )
    except Exception as exc:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "External publication failed exact canonical validation"
        ) from exc
    if (
        publication["implementation_commit"]
        != implementation_manifest["implementation_commit"]
        or publication["tag_target_commit"]
        != implementation_manifest["implementation_commit"]
        or publication["remote_peeled_commit"]
        != implementation_manifest["implementation_commit"]
        or publication["attempt_id"] != attempt_id
        or publication["terminal_status"] != terminal_status
        or publication["report_kind"] != report_kind
        or publication["artifact_sha256"] != artifact_sha256
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "External publication crossed its implementation, attempt, or artifact"
        )
    return publication


def _acquisition_report_payload(
    value: Any, *, require_opaque: bool
) -> dict[str, Any]:
    try:
        from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
            ACQUISITION_VALIDATION_SCHEMA_VERSION,
            ACQUISITION_VALIDATION_VERIFIER_ID,
            is_verified_acquisition_report,
        )
    except ImportError as exc:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition validation authority is unavailable"
        ) from exc
    if require_opaque:
        if not is_verified_acquisition_report(value):
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Acquisition terminal evidence requires an opaque verified report"
            )
        raw = value.as_dict()
    else:
        raw = value
    observed = _mapping(raw, "verified acquisition report")
    _expect_keys(
        observed,
        set(ACQUISITION_VALIDATION_FIELDS),
        "verified acquisition report",
    )
    checks = _mapping(
        observed["checks"], "verified acquisition report checks"
    )
    if set(checks) != set(ACQUISITION_VALIDATION_CHECKS):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition validation lacks the exact seven frozen checks"
        )
    for name in ACQUISITION_VALIDATION_CHECKS:
        _sha256(
            checks[name],
            f"verified acquisition report checks.{name}",
        )
    predecessors = observed["predecessor_chain_bundle_sha256s"]
    if type(predecessors) is not list:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition predecessor chain must be one exact list"
        )
    for index, digest in enumerate(predecessors):
        _sha256(
            digest,
            f"verified acquisition predecessor bundle {index}",
        )
    for field in (
        "acquisition_plan_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "check_set_sha256",
        "validation_sha256",
    ):
        _sha256(observed[field], f"verified acquisition report {field}")
    body = {
        key: observed[key]
        for key in observed
        if key != "validation_sha256"
    }
    if (
        observed["schema_version"] != ACQUISITION_VALIDATION_SCHEMA_VERSION
        or not observed["schema_version"].startswith(
            f"{CONTRACT_VERSION}-"
        )
        or observed["verifier_id"] != ACQUISITION_VALIDATION_VERIFIER_ID
        or not observed["verifier_id"].startswith(f"{CONTRACT_VERSION}-")
        or observed["verdict"] != "pass"
        or observed["stage"]
        not in {"development", "confirmation", "final"}
        or observed["attempt_id"] not in ATTEMPT_KIND_BY_ID
        or observed["attempt_kind"]
        != ATTEMPT_KIND_BY_ID[observed["attempt_id"]]
        or observed["check_set_sha256"] != canonical_sha256(checks)
        or observed["validation_sha256"] != canonical_sha256(body)
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition validation is not an exact v2.1 pass"
        )
    return observed


def _stage_gate_checks(stage: str, value: Any) -> dict[str, bool]:
    observed = _mapping(value, f"{stage} gate checks")
    expected_names = _CONTRACT_GATES.get(stage)
    if expected_names is None or set(observed) != set(expected_names):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            f"{stage} gate checks must use the exact literal contract key set"
        )
    result: dict[str, bool] = {}
    for name in expected_names:
        passed = observed[name]
        if type(passed) is not bool:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                f"{stage} gate check {name} must be an exact Boolean"
            )
        result[name] = passed
    return result


def _scored_joint_material(
    value: Any,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    joint = _self_hashed_body(
        value,
        hash_field="joint_stage_report_sha256",
        location="joint stage report",
    )
    stage = _SCORING_STAGE_BY_KIND.get(plan["attempt_kind"])
    if stage is None:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Scored terminal evidence cannot seal an acquisition attempt"
        )
    if (
        joint.get("schema_version") != JOINT_STAGE_REPORT_SCHEMA_VERSION
        or joint.get("contract_version") != CONTRACT_VERSION
        or joint.get("contract_sha256") != CONTRACT_SHA256
        or joint.get("implementation_manifest_sha256")
        != plan["implementation_manifest_sha256"]
        or joint.get("attempt_id") != plan["attempt_id"]
        or joint.get("attempt_plan_sha256") != plan["attempt_plan_sha256"]
        or joint.get("metric_stage") != stage
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Joint report crossed its scored attempt or stage"
        )
    evaluation = _self_hashed_body(
        joint.get("deterministic_evaluation"),
        hash_field="deterministic_evaluation_sha256",
        location="joint deterministic evaluation",
    )
    if (
        evaluation.get("schema_version")
        != DETERMINISTIC_EVALUATION_SCHEMA_VERSION
        or evaluation.get("contract_version") != CONTRACT_VERSION
        or evaluation.get("contract_sha256") != CONTRACT_SHA256
        or evaluation.get("stage") != stage
        or joint.get("deterministic_evaluation_sha256")
        != evaluation["deterministic_evaluation_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Joint report lost its deterministic evaluation binding"
        )
    stage_input_hash = _sha256(
        evaluation.get("stage_input_bundle_sha256"),
        "deterministic evaluation stage_input_bundle_sha256",
    )
    metrics_input = _mapping(
        evaluation.get("metrics_input"),
        "deterministic evaluation metrics input",
    )
    metrics_input_hash = _sha256(
        metrics_input.get("stage_metrics_input_sha256"),
        "deterministic evaluation stage_metrics_input_sha256",
    )
    if evaluation.get("metrics_input_sha256") != metrics_input_hash:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Deterministic evaluation lost its stage-metrics input binding"
        )
    metrics = _mapping(
        evaluation.get("stage_metrics"),
        "deterministic evaluation stage metrics",
    )
    metrics_hash = _sha256(
        evaluation.get("stage_metrics_sha256"),
        "deterministic evaluation stage_metrics_sha256",
    )
    if metrics.get("stage_metrics_sha256") != metrics_hash:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Deterministic evaluation lost its stage metrics binding"
        )
    gate = _self_hashed_body(
        evaluation.get("gate_report"),
        hash_field="gate_report_sha256",
        location="deterministic evaluation gate report",
    )
    gate_hash = _sha256(
        evaluation.get("gate_report_sha256"),
        "deterministic evaluation gate_report_sha256",
    )
    checks = _stage_gate_checks(stage, gate.get("checks"))
    failed = [name for name in _CONTRACT_GATES[stage] if not checks[name]]
    if (
        gate["gate_report_sha256"] != gate_hash
        or gate.get("passed") is not (not failed)
        or gate.get("failed_checks") != failed
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Gate report verdict differs from the exact literal gate checks"
        )
    proofs = _mapping(
        evaluation.get("no_leverage_proofs"),
        "deterministic evaluation no-leverage proofs",
    )
    proofs_hash = _sha256(
        evaluation.get("no_leverage_proofs_sha256"),
        "deterministic evaluation no_leverage_proofs_sha256",
    )
    if proofs_hash != canonical_sha256(proofs):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "No-leverage proof commitment changed"
        )
    return {
        "stage": stage,
        "stage_input_bundle_sha256": stage_input_hash,
        "deterministic_evaluation_sha256": evaluation[
            "deterministic_evaluation_sha256"
        ],
        "stage_metrics_input_sha256": metrics_input_hash,
        "stage_metrics_sha256": metrics_hash,
        "gate_report_sha256": gate_hash,
        "no_leverage_proofs_sha256": proofs_hash,
        "joint_stage_report_sha256": joint[
            "joint_stage_report_sha256"
        ],
        "gate_checks": checks,
        "failed_gate_names": failed,
        "joint_stage_report": joint,
    }


class VerifiedAcquisitionTerminalEvidence:
    """Opaque authority for one complete acquisition terminal pass."""

    __slots__ = (
        "_acquisition_report",
        "_canonical_bytes",
        "_artifact_payload_bytes",
        "_artifact_receipt",
        "_external_publication",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        evidence: Mapping[str, Any],
        acquisition_report: Any,
        artifact_payload: Mapping[str, Any],
        artifact_receipt: Any,
        external_publication: Any,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VERIFIED_ACQUISITION_TERMINAL_EVIDENCE_SENTINEL:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Acquisition terminal evidence can only be verifier-issued"
            )
        self._canonical_bytes = canonical_json_bytes(
            _mapping(evidence, "acquisition terminal evidence")
        )
        self._acquisition_report = acquisition_report
        self._artifact_payload_bytes = canonical_json_bytes(
            _mapping(artifact_payload, "acquisition terminal artifact")
        )
        self._artifact_receipt = artifact_receipt
        self._external_publication = external_publication
        self._sentinel = _sentinel

    @property
    def evidence(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes.decode("utf-8"))

    @property
    def artifact_payload(self) -> dict[str, Any]:
        return json.loads(self._artifact_payload_bytes.decode("utf-8"))

    @property
    def acquisition_report(self) -> Any:
        return self._acquisition_report

    @property
    def artifact_receipt(self) -> Any:
        return self._artifact_receipt

    @property
    def external_publication(self) -> Any:
        return self._external_publication

    @property
    def terminal_evidence_sha256(self) -> str:
        return self.evidence["terminal_evidence_sha256"]

    def __repr__(self) -> str:
        return "VerifiedAcquisitionTerminalEvidence(<redacted>)"


class VerifiedScoredTerminalEvidence:
    """Opaque authority for a scored pass or complete failed-gate diagnostic."""

    __slots__ = (
        "_canonical_bytes",
        "_artifact_payload_bytes",
        "_artifact_receipt",
        "_external_publication",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        evidence: Mapping[str, Any],
        artifact_payload: Mapping[str, Any],
        artifact_receipt: Any,
        external_publication: Any,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VERIFIED_SCORED_TERMINAL_EVIDENCE_SENTINEL:
            raise SecGemmaOnlineRiskOverlayAttemptError(
                "Scored terminal evidence can only be verifier-issued"
            )
        self._canonical_bytes = canonical_json_bytes(
            _mapping(evidence, "scored terminal evidence")
        )
        self._artifact_payload_bytes = canonical_json_bytes(
            _mapping(artifact_payload, "scored terminal artifact")
        )
        self._artifact_receipt = artifact_receipt
        self._external_publication = external_publication
        self._sentinel = _sentinel

    @property
    def evidence(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes.decode("utf-8"))

    @property
    def artifact_payload(self) -> dict[str, Any]:
        return json.loads(self._artifact_payload_bytes.decode("utf-8"))

    @property
    def artifact_receipt(self) -> Any:
        return self._artifact_receipt

    @property
    def external_publication(self) -> Any:
        return self._external_publication

    @property
    def terminal_evidence_sha256(self) -> str:
        return self.evidence["terminal_evidence_sha256"]

    def __repr__(self) -> str:
        return "VerifiedScoredTerminalEvidence(<redacted>)"


def issue_verified_acquisition_terminal_evidence(
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_plan: Mapping[str, Any],
    acquisition_report: Any,
    report_material: Mapping[str, Any],
    acquisition_artifact_receipt: Any,
    external_publication: Any,
) -> VerifiedAcquisitionTerminalEvidence:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    plan = validate_attempt_plan(
        attempt_plan,
        implementation_manifest=implementation,
    )
    if plan["attempt_kind"] != DEVELOPMENT_ACQUISITION:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition terminal evidence is only valid for acquisition"
        )
    report = _acquisition_report_payload(
        acquisition_report, require_opaque=True
    )
    material = _report_material(report_material)
    receipt = store_record_receipt_material(
        acquisition_artifact_receipt
    )
    if (
        report["stage"] != "development"
        or report["attempt_id"] != plan["attempt_id"]
        or report["attempt_kind"] != plan["attempt_kind"]
        or material["attempt_id"] != plan["attempt_id"]
        or material["attempt_plan_sha256"] != plan["attempt_plan_sha256"]
        or receipt["table"] != "artifacts"
        or receipt["attempt_id"] != plan["attempt_id"]
        or receipt["payload_sha256"] != canonical_sha256(report)
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition terminal material crossed its plan or stored artifact"
        )
    publication = _validate_publication_binding(
        external_publication,
        implementation_manifest=implementation,
        attempt_id=plan["attempt_id"],
        terminal_status=TERMINAL_PASS,
        report_kind="acquisition_pass",
        artifact_sha256=report["validation_sha256"],
    )
    body = {
        "schema_version": ACQUISITION_TERMINAL_EVIDENCE_SCHEMA_VERSION,
        "verifier_id": ACQUISITION_TERMINAL_EVIDENCE_VERIFIER_ID,
        "verdict": "pass",
        "terminal_status": TERMINAL_PASS,
        "stage": "development",
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "acquisition_validation_sha256": report["validation_sha256"],
        "bundle_sha256": report["bundle_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "private_index_sha256": report["private_index_sha256"],
        "check_set_sha256": report["check_set_sha256"],
        "record_commitment_sha256": material[
            "record_commitment_sha256"
        ],
        "acquisition_artifact_receipt_sha256": canonical_sha256(receipt),
        "external_publication_sha256": publication["publication_sha256"],
    }
    evidence = {
        **body,
        "terminal_evidence_sha256": canonical_sha256(body),
    }
    _expect_keys(
        evidence,
        set(ACQUISITION_TERMINAL_EVIDENCE_FIELDS),
        "acquisition terminal evidence",
    )
    return VerifiedAcquisitionTerminalEvidence(
        evidence=evidence,
        acquisition_report=acquisition_report,
        artifact_payload=report,
        artifact_receipt=acquisition_artifact_receipt,
        external_publication=external_publication,
        _sentinel=_VERIFIED_ACQUISITION_TERMINAL_EVIDENCE_SENTINEL,
    )


def issue_verified_scored_terminal_evidence(
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_plan: Mapping[str, Any],
    joint_stage_report: Mapping[str, Any],
    report_material: Mapping[str, Any],
    joint_artifact_receipt: Any,
    gate_checks: Mapping[str, bool],
    external_publication: Any,
) -> VerifiedScoredTerminalEvidence:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    plan = validate_attempt_plan(
        attempt_plan,
        implementation_manifest=implementation,
    )
    joint = _scored_joint_material(joint_stage_report, plan=plan)
    supplied_checks = _stage_gate_checks(
        joint["stage"], gate_checks
    )
    if supplied_checks != joint["gate_checks"]:
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Caller gate checks differ from the sealed deterministic evaluation"
        )
    material = _report_material(report_material)
    receipt = store_record_receipt_material(joint_artifact_receipt)
    if (
        material["attempt_id"] != plan["attempt_id"]
        or material["attempt_plan_sha256"] != plan["attempt_plan_sha256"]
        or receipt["table"] != "artifacts"
        or receipt["attempt_id"] != plan["attempt_id"]
        or receipt["payload_sha256"]
        != canonical_sha256(joint["joint_stage_report"])
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Scored terminal material crossed its plan or stored artifact"
        )
    failed = joint["failed_gate_names"]
    terminal_status = TERMINAL_PASS if not failed else TERMINAL_FAIL
    verdict = "pass" if not failed else "failed_gate"
    report_kind = "scored_pass" if not failed else "scored_failed_gate"
    publication = _validate_publication_binding(
        external_publication,
        implementation_manifest=implementation,
        attempt_id=plan["attempt_id"],
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=joint["joint_stage_report_sha256"],
    )
    body = {
        "schema_version": SCORED_TERMINAL_EVIDENCE_SCHEMA_VERSION,
        "verifier_id": SCORED_TERMINAL_EVIDENCE_VERIFIER_ID,
        "verdict": verdict,
        "terminal_status": terminal_status,
        "stage": joint["stage"],
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "stage_input_bundle_sha256": joint[
            "stage_input_bundle_sha256"
        ],
        "deterministic_evaluation_sha256": joint[
            "deterministic_evaluation_sha256"
        ],
        "stage_metrics_input_sha256": joint[
            "stage_metrics_input_sha256"
        ],
        "stage_metrics_sha256": joint["stage_metrics_sha256"],
        "gate_report_sha256": joint["gate_report_sha256"],
        "no_leverage_proofs_sha256": joint[
            "no_leverage_proofs_sha256"
        ],
        "joint_stage_report_sha256": joint[
            "joint_stage_report_sha256"
        ],
        "record_counts": material["record_counts"],
        "record_commitment_sha256": material[
            "record_commitment_sha256"
        ],
        "joint_artifact_receipt_sha256": canonical_sha256(receipt),
        "gate_checks": supplied_checks,
        "gate_check_set_sha256": canonical_sha256(supplied_checks),
        "failed_gate_names": failed,
        "external_publication_sha256": publication["publication_sha256"],
    }
    evidence = {
        **body,
        "terminal_evidence_sha256": canonical_sha256(body),
    }
    _expect_keys(
        evidence,
        set(SCORED_TERMINAL_EVIDENCE_FIELDS),
        "scored terminal evidence",
    )
    return VerifiedScoredTerminalEvidence(
        evidence=evidence,
        artifact_payload=joint["joint_stage_report"],
        artifact_receipt=joint_artifact_receipt,
        external_publication=external_publication,
        _sentinel=_VERIFIED_SCORED_TERMINAL_EVIDENCE_SENTINEL,
    )


def validate_acquisition_terminal_evidence(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not is_verified_acquisition_terminal_evidence(value):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition terminal evidence must be opaque and verifier-issued"
        )
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    plan = validate_attempt_plan(
        attempt_plan,
        implementation_manifest=implementation,
    )
    assert type(value) is VerifiedAcquisitionTerminalEvidence
    observed = _self_hashed_body(
        value.evidence,
        hash_field="terminal_evidence_sha256",
        location="acquisition terminal evidence",
    )
    _expect_keys(
        observed,
        set(ACQUISITION_TERMINAL_EVIDENCE_FIELDS),
        "acquisition terminal evidence",
    )
    current_report = _acquisition_report_payload(
        value.acquisition_report, require_opaque=True
    )
    artifact = _acquisition_report_payload(
        value.artifact_payload, require_opaque=False
    )
    receipt = store_record_receipt_material(value.artifact_receipt)
    publication = _validate_publication_binding(
        value.external_publication,
        implementation_manifest=implementation,
        attempt_id=plan["attempt_id"],
        terminal_status=TERMINAL_PASS,
        report_kind="acquisition_pass",
        artifact_sha256=artifact["validation_sha256"],
    )
    if (
        plan["attempt_kind"] != DEVELOPMENT_ACQUISITION
        or observed["schema_version"]
        != ACQUISITION_TERMINAL_EVIDENCE_SCHEMA_VERSION
        or observed["verifier_id"]
        != ACQUISITION_TERMINAL_EVIDENCE_VERIFIER_ID
        or observed["verdict"] != "pass"
        or observed["terminal_status"] != TERMINAL_PASS
        or observed["stage"] != "development"
        or observed["attempt_id"] != plan["attempt_id"]
        or observed["attempt_kind"] != plan["attempt_kind"]
        or observed["attempt_plan_sha256"] != plan["attempt_plan_sha256"]
        or observed["acquisition_validation_sha256"]
        != artifact["validation_sha256"]
        or observed["bundle_sha256"] != artifact["bundle_sha256"]
        or observed["manifest_sha256"] != artifact["manifest_sha256"]
        or observed["private_index_sha256"]
        != artifact["private_index_sha256"]
        or observed["check_set_sha256"] != artifact["check_set_sha256"]
        or current_report != artifact
        or receipt["table"] != "artifacts"
        or receipt["attempt_id"] != plan["attempt_id"]
        or receipt["payload_sha256"] != canonical_sha256(artifact)
        or observed["acquisition_artifact_receipt_sha256"]
        != canonical_sha256(receipt)
        or observed["external_publication_sha256"]
        != publication["publication_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Acquisition terminal evidence lost an exact artifact binding"
        )
    _sha256(
        observed["record_commitment_sha256"],
        "acquisition terminal record commitment",
    )
    return observed


def validate_scored_terminal_evidence(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not is_verified_scored_terminal_evidence(value):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Scored terminal evidence must be opaque and verifier-issued"
        )
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    plan = validate_attempt_plan(
        attempt_plan,
        implementation_manifest=implementation,
    )
    assert type(value) is VerifiedScoredTerminalEvidence
    observed = _self_hashed_body(
        value.evidence,
        hash_field="terminal_evidence_sha256",
        location="scored terminal evidence",
    )
    _expect_keys(
        observed,
        set(SCORED_TERMINAL_EVIDENCE_FIELDS),
        "scored terminal evidence",
    )
    joint = _scored_joint_material(value.artifact_payload, plan=plan)
    receipt = store_record_receipt_material(value.artifact_receipt)
    checks = _stage_gate_checks(observed["stage"], observed["gate_checks"])
    failed = [name for name in _CONTRACT_GATES[observed["stage"]] if not checks[name]]
    terminal_status = TERMINAL_PASS if not failed else TERMINAL_FAIL
    verdict = "pass" if not failed else "failed_gate"
    report_kind = "scored_pass" if not failed else "scored_failed_gate"
    publication = _validate_publication_binding(
        value.external_publication,
        implementation_manifest=implementation,
        attempt_id=plan["attempt_id"],
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=joint["joint_stage_report_sha256"],
    )
    if (
        observed["schema_version"]
        != SCORED_TERMINAL_EVIDENCE_SCHEMA_VERSION
        or observed["verifier_id"]
        != SCORED_TERMINAL_EVIDENCE_VERIFIER_ID
        or observed["verdict"] != verdict
        or observed["terminal_status"] != terminal_status
        or observed["stage"] != joint["stage"]
        or observed["attempt_id"] != plan["attempt_id"]
        or observed["attempt_kind"] != plan["attempt_kind"]
        or observed["attempt_plan_sha256"] != plan["attempt_plan_sha256"]
        or observed["stage_input_bundle_sha256"]
        != joint["stage_input_bundle_sha256"]
        or observed["deterministic_evaluation_sha256"]
        != joint["deterministic_evaluation_sha256"]
        or observed["stage_metrics_input_sha256"]
        != joint["stage_metrics_input_sha256"]
        or observed["stage_metrics_sha256"]
        != joint["stage_metrics_sha256"]
        or observed["gate_report_sha256"] != joint["gate_report_sha256"]
        or observed["no_leverage_proofs_sha256"]
        != joint["no_leverage_proofs_sha256"]
        or observed["joint_stage_report_sha256"]
        != joint["joint_stage_report_sha256"]
        or observed["record_counts"]
        != _record_counts(observed["record_counts"], "scored record counts")
        or receipt["table"] != "artifacts"
        or receipt["attempt_id"] != plan["attempt_id"]
        or receipt["payload_sha256"]
        != canonical_sha256(joint["joint_stage_report"])
        or observed["joint_artifact_receipt_sha256"]
        != canonical_sha256(receipt)
        or checks != joint["gate_checks"]
        or observed["gate_check_set_sha256"] != canonical_sha256(checks)
        or observed["failed_gate_names"] != failed
        or observed["external_publication_sha256"]
        != publication["publication_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayAttemptError(
            "Scored terminal evidence lost an exact artifact or gate binding"
        )
    _sha256(
        observed["record_commitment_sha256"],
        "scored terminal record commitment",
    )
    return observed


def is_verified_acquisition_terminal_evidence(value: Any) -> bool:
    return (
        type(value) is VerifiedAcquisitionTerminalEvidence
        and getattr(value, "_sentinel", None)
        is _VERIFIED_ACQUISITION_TERMINAL_EVIDENCE_SENTINEL
    )


def is_verified_scored_terminal_evidence(value: Any) -> bool:
    return (
        type(value) is VerifiedScoredTerminalEvidence
        and getattr(value, "_sentinel", None)
        is _VERIFIED_SCORED_TERMINAL_EVIDENCE_SENTINEL
    )


def payload_sha256(value: Any) -> str:
    """Hash one detached finite JSON payload using the experiment encoding."""

    return hashlib.sha256(
        canonical_json_bytes(_plain_json(value, "payload"))
    ).hexdigest()


__all__ = [
    "ATTEMPT_IDS",
    "ATTEMPT_ID_BY_KIND",
    "ATTEMPT_KIND_BY_ID",
    "ATTEMPT_KINDS",
    "ATTEMPT_PLAN_SCHEMA_VERSION",
    "ATTEMPT_STATUSES",
    "ATTEMPT_TRANSITION_SCHEMA_VERSION",
    "ACQUISITION_TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "ACQUISITION_TERMINAL_EVIDENCE_VERIFIER_ID",
    "CONFIRMATION_SCORING",
    "CONSUMED",
    "CONTRACT_SOURCE_PATH",
    "DEVELOPMENT_ACQUISITION",
    "DEVELOPMENT_SCORING",
    "DETERMINISTIC_EVALUATION_SCHEMA_VERSION",
    "FINAL_SCORING",
    "IMPLEMENTATION_MANIFEST_SCHEMA_VERSION",
    "JOINT_STAGE_REPORT_SCHEMA_VERSION",
    "MINIMUM_PASS_RECORD_COUNTS",
    "PLANNED",
    "PREREGISTRATION_COMMIT",
    "PREREQUISITE_ATTEMPT_BY_ID",
    "REPORT_RECORD_TABLES",
    "SCORED_TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "SCORED_TERMINAL_EVIDENCE_VERIFIER_ID",
    "SecGemmaOnlineRiskOverlayAttemptError",
    "TERMINAL_FAIL",
    "TERMINAL_INDETERMINATE",
    "TERMINAL_PASS",
    "TERMINAL_STATUSES",
    "VerifiedAcquisitionTerminalEvidence",
    "VerifiedScoredTerminalEvidence",
    "build_attempt_plan",
    "build_attempt_transition",
    "build_implementation_manifest",
    "current_attempt_status",
    "is_verified_acquisition_terminal_evidence",
    "is_verified_scored_terminal_evidence",
    "issue_verified_acquisition_terminal_evidence",
    "issue_verified_scored_terminal_evidence",
    "payload_sha256",
    "store_record_receipt_material",
    "store_record_receipt_sha256",
    "validate_acquisition_terminal_evidence",
    "validate_attempt_history",
    "validate_attempt_plan",
    "validate_implementation_manifest",
    "validate_scored_terminal_evidence",
]
