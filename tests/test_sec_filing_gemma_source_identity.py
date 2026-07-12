from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    REQUIRED_SOURCE_HASHES,
    build_candidate_manifest,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_source_identity import (
    CANONICAL_SOURCE_ROLE_PATHS,
    SOURCE_IDENTITY_RECEIPT_SCHEMA_VERSION,
    SOURCE_TREE_SCHEMA_VERSION,
    UNRESOLVED_SOURCE_ROLES,
    SecFilingGemmaSourceIdentityError,
    SecFilingGemmaSourceIdentityIncompleteError,
    audit_candidate_source_identity,
    audit_runtime_candidate_source_identity,
    canonical_source_tree_sha256,
    validate_complete_candidate_source_identity,
    validate_complete_runtime_candidate_source_identity,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UNRESOLVED_ROLES = (
    "extractor_prompt",
    "extractor_schema",
    "ledger",
    "market_acquirer",
)
EXPECTED_TRANSITIVE_SOURCE_PATHS = {
    "package_init": "agent_benchmark/__init__.py",
    "cftc_cot_policy": "agent_benchmark/cftc_cot_policy.py",
    "deterministic_aapl": "agent_benchmark/deterministic_aapl.py",
    "direct_edge_features": "agent_benchmark/direct_edge_features.py",
    "downside_features": "agent_benchmark/downside_features.py",
    "sec_audit_artifact": "agent_benchmark/sec_audit_artifact.py",
    "sec_audit_evaluation": "agent_benchmark/sec_audit_evaluation.py",
    "sec_audit_plan": "agent_benchmark/sec_audit_plan.py",
    "sec_audit_selection": "agent_benchmark/sec_audit_selection.py",
    "sec_point_in_time": "agent_benchmark/sec_point_in_time.py",
    "unleveraged_aapl": "agent_benchmark/unleveraged_aapl.py",
}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate(source_hashes: dict[str, str]) -> dict[str, object]:
    return build_candidate_manifest(
        model_digest=_hash("model"),
        ollama_runtime_fingerprint_sha256=_hash("runtime"),
        sec_audit_checksums_json_sha256=_hash("audit-checksums"),
        sec_catalog_artifact_sha256=_hash("catalog"),
        sec_audit_source_commit="a" * 40,
        calendar_source_evidence_sha256=_hash("calendar-source"),
        calendar_sessions_sha256=session_calendar_sha256(EXPECTED_SESSIONS),
        corpus_universe_sha256=_hash("universe"),
        corpus_universe_semantic_sha256=_hash("universe-semantic"),
        identity_lexicon_sha256=_hash("identity-lexicon"),
        predecessor_reveal_registry_sha256=_hash("predecessor-registry"),
        holdout_attempt_id="aapl-sec-filing-gemma-v1-attempt-001",
        experiment_source_commit="b" * 40,
        source_tree_sha256=canonical_source_tree_sha256(source_hashes),
        source_hashes=source_hashes,
    )


def _load_declared_runtime_modules() -> None:
    for path in CANONICAL_SOURCE_ROLE_PATHS.values():
        if path is not None:
            module_name = (
                "agent_benchmark"
                if path == "agent_benchmark/__init__.py"
                else path[:-3].replace("/", ".")
            )
            importlib.import_module(module_name)


@pytest.fixture()
def evidence() -> dict[str, object]:
    paths = dict(CANONICAL_SOURCE_ROLE_PATHS)
    payloads: dict[str, bytes | None] = {}
    source_hashes: dict[str, str] = {}
    for role in REQUIRED_SOURCE_HASHES:
        path = paths[role]
        if path is None:
            payloads[role] = None
            source_hashes[role] = _hash(f"unresolved:{role}")
            continue
        payload = (REPOSITORY_ROOT / path).read_bytes()
        payloads[role] = payload
        source_hashes[role] = hashlib.sha256(payload).hexdigest()
    candidate = _candidate(source_hashes)
    return {
        "paths": paths,
        "payloads": payloads,
        "source_hashes": source_hashes,
        "candidate": candidate,
        "candidate_hash": candidate["candidate_sha256"],
    }


def _audit(evidence: dict[str, object], **overrides) -> dict[str, object]:
    arguments = {
        "candidate_manifest": evidence["candidate"],
        "expected_candidate_sha256": evidence["candidate_hash"],
        "source_paths_by_role": evidence["paths"],
        "source_bytes_by_role": evidence["payloads"],
    }
    arguments.update(overrides)
    return audit_candidate_source_identity(**arguments)


def test_frozen_role_mapping_covers_every_role_without_resolved_aliases() -> None:
    assert SOURCE_IDENTITY_RECEIPT_SCHEMA_VERSION == (
        "aapl-sec-gemma-source-identity-audit-v4"
    )
    assert tuple(CANONICAL_SOURCE_ROLE_PATHS) == REQUIRED_SOURCE_HASHES
    assert UNRESOLVED_SOURCE_ROLES == EXPECTED_UNRESOLVED_ROLES
    resolved = [
        path for path in CANONICAL_SOURCE_ROLE_PATHS.values() if path is not None
    ]
    assert len(resolved) == len({path.casefold() for path in resolved})
    for path in resolved:
        assert path.startswith("agent_benchmark/")
        assert (REPOSITORY_ROOT / path).is_file()
    assert CANONICAL_SOURCE_ROLE_PATHS["extractor_prompt"] is None
    assert CANONICAL_SOURCE_ROLE_PATHS["extractor_schema"] is None
    assert CANONICAL_SOURCE_ROLE_PATHS["runner"] == (
        "agent_benchmark/sec_filing_gemma_stage_runner.py"
    )
    assert {
        role: CANONICAL_SOURCE_ROLE_PATHS[role]
        for role in EXPECTED_TRANSITIVE_SOURCE_PATHS
    } == EXPECTED_TRANSITIVE_SOURCE_PATHS


def test_audit_hashes_exact_detached_bytes_and_is_canonical_nonauthorizing(
    evidence,
) -> None:
    receipt = _audit(evidence)
    assert receipt["schema_version"] == SOURCE_IDENTITY_RECEIPT_SCHEMA_VERSION
    assert receipt["candidate_sha256"] == evidence["candidate_hash"]
    assert receipt["required_source_roles"] == list(REQUIRED_SOURCE_HASHES)
    assert receipt["source_role_paths"] == evidence["paths"]
    assert receipt["declared_static_local_import_closure_complete"] is True
    assert receipt[
        "declared_static_local_import_closure_sha256"
    ] == canonical_sha256(
        receipt["declared_static_local_imports_by_source"]
    )
    assert set(receipt["declared_static_local_imports_by_source"]) == {
        path for path in evidence["paths"].values() if path is not None
    }
    assert "agent_benchmark/sec_point_in_time.py" in receipt[
        "declared_static_local_imports_by_source"
    ]["agent_benchmark/sec_filing_gemma_corpus.py"]
    assert "agent_benchmark/unleveraged_aapl.py" in receipt[
        "declared_static_local_imports_by_source"
    ]["agent_benchmark/sec_filing_gemma_no_leverage.py"]
    runner_imports = receipt["declared_static_local_imports_by_source"][
        "agent_benchmark/sec_filing_gemma_stage_runner.py"
    ]
    assert "agent_benchmark/sec_filing_gemma_corpus.py" in runner_imports
    assert "agent_benchmark/sec_filing_gemma_reveal_store.py" in runner_imports
    assert receipt["candidate_source_hashes"] == evidence["source_hashes"]
    assert receipt["unresolved_roles"] == list(EXPECTED_UNRESOLVED_ROLES)
    assert receipt["candidate_source_tree_sha256"] == canonical_source_tree_sha256(
        evidence["source_hashes"]
    )
    assert receipt["computed_source_tree_sha256"] == receipt[
        "candidate_source_tree_sha256"
    ]
    assert receipt["role_ownership_complete"] is False
    assert receipt["runtime_source_files_verified"] is False
    assert receipt["runtime_module_paths_verified"] is False
    assert receipt["complete"] is False
    assert receipt["authorizes"] is False
    assert receipt["authorization_scope"] == "none"
    assert receipt["resolved_role_count"] + receipt["unresolved_role_count"] == len(
        REQUIRED_SOURCE_HASHES
    )
    expected_total = sum(
        len(payload)
        for payload in evidence["payloads"].values()
        if payload is not None
    )
    assert receipt["source_byte_count_total"] == expected_total
    for item in receipt["resolved_sources"]:
        payload = evidence["payloads"][item["role"]]
        assert item["repository_path"] == evidence["paths"][item["role"]]
        assert item["byte_count"] == len(payload)
        assert item["source_sha256"] == hashlib.sha256(payload).hexdigest()
        assert item["candidate_source_sha256"] == evidence["source_hashes"][
            item["role"]
        ]
    body = {
        key: value
        for key, value in receipt.items()
        if key != "source_identity_receipt_sha256"
    }
    assert receipt["source_identity_receipt_sha256"] == canonical_sha256(body)
    json.dumps(receipt, allow_nan=False)


def test_audit_is_independent_of_input_mapping_insertion_order(evidence) -> None:
    reversed_paths = dict(reversed(list(evidence["paths"].items())))
    reversed_payloads = dict(reversed(list(evidence["payloads"].items())))
    assert _audit(evidence) == _audit(
        evidence,
        source_paths_by_role=reversed_paths,
        source_bytes_by_role=reversed_payloads,
    )


def test_static_local_import_omission_is_rejected_even_when_hashes_are_rebuilt(
    evidence,
) -> None:
    payloads = dict(evidence["payloads"])
    source_hashes = dict(evidence["source_hashes"])
    injected = (
        payloads["contract"]
        + b"\nfrom agent_benchmark.local_gemma_loop import main\n"
    )
    payloads["contract"] = injected
    source_hashes["contract"] = hashlib.sha256(injected).hexdigest()
    candidate = _candidate(source_hashes)

    with pytest.raises(
        SecFilingGemmaSourceIdentityError,
        match="imports undeclared local sources.*local_gemma_loop",
    ):
        audit_candidate_source_identity(
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
            source_paths_by_role=evidence["paths"],
            source_bytes_by_role=payloads,
        )


def test_complete_validator_names_real_unresolved_implementation_roles(evidence) -> None:
    with pytest.raises(
        SecFilingGemmaSourceIdentityIncompleteError,
        match="extractor_prompt.*extractor_schema.*ledger.*market_acquirer",
    ):
        validate_complete_candidate_source_identity(
            candidate_manifest=evidence["candidate"],
            expected_candidate_sha256=evidence["candidate_hash"],
            source_paths_by_role=evidence["paths"],
            source_bytes_by_role=evidence["payloads"],
        )


def test_runtime_audit_reads_actual_loaded_module_files_and_remains_blocked(
    evidence,
) -> None:
    _load_declared_runtime_modules()
    receipt = audit_runtime_candidate_source_identity(
        candidate_manifest=evidence["candidate"],
        expected_candidate_sha256=evidence["candidate_hash"],
    )
    assert receipt["schema_version"] == SOURCE_IDENTITY_RECEIPT_SCHEMA_VERSION
    assert SOURCE_TREE_SCHEMA_VERSION == "aapl-sec-gemma-source-tree-v1"
    assert receipt["receipt_kind"] == (
        "runtime_non_authorizing_source_identity_audit"
    )
    assert receipt["runtime_source_files_verified"] is True
    assert receipt["runtime_module_paths_verified"] is True
    assert receipt["runtime_module_files_attested"] is True
    assert receipt["runtime_executing_code_bytes_attested"] is False
    assert receipt["caller_supplied_paths_or_bytes_accepted"] is False
    assert receipt["complete"] is False
    assert receipt["authorizes"] is False
    assert receipt["unresolved_roles"] == list(EXPECTED_UNRESOLVED_ROLES)
    assert {item["role"] for item in receipt["runtime_modules"]} == {
        role
        for role, path in CANONICAL_SOURCE_ROLE_PATHS.items()
        if path is not None
    }

    with pytest.raises(
        SecFilingGemmaSourceIdentityIncompleteError,
        match="Runtime source identity remains incomplete",
    ):
        validate_complete_runtime_candidate_source_identity(
            candidate_manifest=evidence["candidate"],
            expected_candidate_sha256=evidence["candidate_hash"],
        )


def test_runtime_audit_rejects_another_checkout_and_loaded_module_path(
    evidence,
    tmp_path,
    monkeypatch,
) -> None:
    _load_declared_runtime_modules()
    import agent_benchmark.sec_filing_gemma_source_identity as source_module

    with monkeypatch.context() as context:
        context.setattr(source_module, "__file__", str(tmp_path / "missing.py"))
        with pytest.raises(SecFilingGemmaSourceIdentityError, match="cannot be resolved"):
            audit_runtime_candidate_source_identity(
                candidate_manifest=evidence["candidate"],
                expected_candidate_sha256=evidence["candidate_hash"],
            )

    import agent_benchmark.sec_session_calendar as calendar_module

    fake_source = tmp_path / "sec_session_calendar.py"
    fake_source.write_bytes(
        (REPOSITORY_ROOT / CANONICAL_SOURCE_ROLE_PATHS["calendar"]).read_bytes()
    )
    monkeypatch.setattr(calendar_module, "__file__", str(fake_source))
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="another path"):
        audit_runtime_candidate_source_identity(
            candidate_manifest=evidence["candidate"],
            expected_candidate_sha256=evidence["candidate_hash"],
        )


def test_runtime_audit_rejects_a_relative_loaded_module_alias(
    evidence,
    monkeypatch,
) -> None:
    _load_declared_runtime_modules()
    import agent_benchmark.sec_session_calendar as calendar_module

    monkeypatch.setattr(
        calendar_module,
        "__file__",
        "agent_benchmark/sec_session_calendar.py",
    )
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="relative path alias"):
        audit_runtime_candidate_source_identity(
            candidate_manifest=evidence["candidate"],
            expected_candidate_sha256=evidence["candidate_hash"],
        )


def test_runtime_audit_never_imports_an_absent_declared_module(
    evidence,
    monkeypatch,
) -> None:
    _load_declared_runtime_modules()
    path = CANONICAL_SOURCE_ROLE_PATHS["sec_audit_artifact"]
    assert path is not None
    module_name = path[:-3].replace("/", ".")
    assert module_name in sys.modules
    monkeypatch.delitem(sys.modules, module_name)
    import_attempts: list[str] = []

    def forbidden_import(name: str, *args, **kwargs):
        import_attempts.append(name)
        raise AssertionError("runtime source audit must not import candidate code")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="not already loaded"):
        audit_runtime_candidate_source_identity(
            candidate_manifest=evidence["candidate"],
            expected_candidate_sha256=evidence["candidate_hash"],
        )
    assert import_attempts == []


def test_runtime_audit_rejects_repository_root_alias(
    evidence, tmp_path, monkeypatch
) -> None:
    _load_declared_runtime_modules()
    alias = tmp_path / "repo-alias"
    try:
        alias.symlink_to(REPOSITORY_ROOT, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    import agent_benchmark.sec_filing_gemma_source_identity as source_module

    aliased_module = alias / "agent_benchmark" / "sec_filing_gemma_source_identity.py"
    with monkeypatch.context() as context:
        context.setattr(source_module, "__file__", str(aliased_module))
        with pytest.raises(SecFilingGemmaSourceIdentityError, match="alias"):
            audit_runtime_candidate_source_identity(
                candidate_manifest=evidence["candidate"],
                expected_candidate_sha256=evidence["candidate_hash"],
            )


def test_source_tree_pin_cannot_be_an_arbitrary_digest(evidence) -> None:
    bad_candidate = build_candidate_manifest(
        model_digest=_hash("model"),
        ollama_runtime_fingerprint_sha256=_hash("runtime"),
        sec_audit_checksums_json_sha256=_hash("audit-checksums"),
        sec_catalog_artifact_sha256=_hash("catalog"),
        sec_audit_source_commit="a" * 40,
        calendar_source_evidence_sha256=_hash("calendar-source"),
        calendar_sessions_sha256=session_calendar_sha256(EXPECTED_SESSIONS),
        corpus_universe_sha256=_hash("universe"),
        corpus_universe_semantic_sha256=_hash("universe-semantic"),
        identity_lexicon_sha256=_hash("identity-lexicon"),
        predecessor_reveal_registry_sha256=_hash("predecessor-registry"),
        holdout_attempt_id="aapl-sec-filing-gemma-v1-attempt-001",
        experiment_source_commit="b" * 40,
        source_tree_sha256=_hash("arbitrary-source-tree"),
        source_hashes=evidence["source_hashes"],
    )
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="source_tree"):
        audit_candidate_source_identity(
            candidate_manifest=bad_candidate,
            expected_candidate_sha256=bad_candidate["candidate_sha256"],
            source_paths_by_role=evidence["paths"],
            source_bytes_by_role=evidence["payloads"],
        )


@pytest.mark.parametrize("mapping_name", ["paths", "payloads"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_role_mapping_omissions_and_extras_fail(evidence, mapping_name, mutation) -> None:
    changed = dict(evidence[mapping_name])
    if mutation == "missing":
        changed.pop("calendar")
    else:
        changed["invented_role"] = None
    override = (
        {"source_paths_by_role": changed}
        if mapping_name == "paths"
        else {"source_bytes_by_role": changed}
    )
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="missing=.*extra="):
        _audit(evidence, **override)


@pytest.mark.parametrize(
    "bad_path",
    [
        "../agent_benchmark/sec_session_calendar.py",
        "/agent_benchmark/sec_session_calendar.py",
        "agent_benchmark\\sec_session_calendar.py",
        "agent_benchmark/./sec_session_calendar.py",
        "agent_benchmark//sec_session_calendar.py",
        "agent_benchmark/SEC_SESSION_CALENDAR.py",
    ],
)
def test_path_traversal_and_spelling_aliases_fail(evidence, bad_path) -> None:
    paths = dict(evidence["paths"])
    paths["calendar"] = bad_path
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="path|alias"):
        _audit(evidence, source_paths_by_role=paths)


def test_role_cannot_alias_another_roles_canonical_path(evidence) -> None:
    paths = dict(evidence["paths"])
    paths["calendar"] = paths["contract"]
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="another role's path"):
        _audit(evidence, source_paths_by_role=paths)


def test_unresolved_role_cannot_invent_a_path_or_supply_bytes(evidence) -> None:
    paths = dict(evidence["paths"])
    paths["market_acquirer"] = "agent_benchmark/market_data.py"
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="invented path"):
        _audit(evidence, source_paths_by_role=paths)

    payloads = dict(evidence["payloads"])
    payloads["market_acquirer"] = b"invented acquisition source"
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="cannot claim detached"):
        _audit(evidence, source_bytes_by_role=payloads)


def test_one_byte_source_substitution_fails_candidate_pin(evidence) -> None:
    payloads = dict(evidence["payloads"])
    payloads["calendar"] = payloads["calendar"] + b"\n"
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="calendar.*candidate"):
        _audit(evidence, source_bytes_by_role=payloads)


def test_coherently_rebuilt_wrong_candidate_source_pin_still_fails_bytes(evidence) -> None:
    source_hashes = dict(evidence["source_hashes"])
    source_hashes["calendar"] = _hash("wrong-calendar-pin")
    candidate = _candidate(source_hashes)
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="calendar.*candidate"):
        _audit(
            evidence,
            candidate_manifest=candidate,
            expected_candidate_sha256=candidate["candidate_sha256"],
        )


def test_inputs_must_be_detached_builtin_mappings_and_exact_bytes(evidence) -> None:
    class MappingSubclass(dict):
        pass

    with pytest.raises(SecFilingGemmaSourceIdentityError, match="detached built-in"):
        _audit(evidence, source_paths_by_role=MappingSubclass(evidence["paths"]))

    payloads = dict(evidence["payloads"])
    payloads["calendar"] = bytearray(payloads["calendar"])
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="detached bytes"):
        _audit(evidence, source_bytes_by_role=payloads)

    with pytest.raises(SecFilingGemmaSourceIdentityError, match="candidate"):
        _audit(
            evidence,
            candidate_manifest=MappingSubclass(evidence["candidate"]),
        )


def test_wrong_external_candidate_pin_fails(evidence) -> None:
    with pytest.raises(SecFilingGemmaSourceIdentityError, match="Candidate"):
        _audit(evidence, expected_candidate_sha256=_hash("wrong-candidate"))
