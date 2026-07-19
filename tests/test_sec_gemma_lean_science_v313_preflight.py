from __future__ import annotations

import copy
import builtins
import hashlib
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import time
import typing
import types
import weakref
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from agent_benchmark import sec_gemma_lean_science_v313_contract as contract
from agent_benchmark import sec_gemma_lean_science_v313_bridge as bridge
from agent_benchmark import sec_gemma_lean_science_v313_preflight as preflight
from agent_benchmark import sec_gemma_lean_science_v313_runner as runner
from agent_benchmark import sec_gemma_lean_science_v313_store as store_module
from tests import test_sec_gemma_lean_science_v313_bridge as bridge_fixtures


SHA1_A = "1" * 40
SHA1_B = "2" * 40
SHA64 = {
    name: hashlib.sha256(name.encode("ascii")).hexdigest()
    for name in (
        "legacy",
        "documents",
        "records",
        "source_order",
        "events",
        "prior_links",
        "primary",
        "preprocessed",
        "requests",
        "pilot",
        "remaining",
        "compatibility",
        "universe",
        "content",
        "proofs",
        "model_slice",
        "request_index",
        "model_plan",
    )
}


def _assert_code(expected_code: str, function, *args, **kwargs) -> None:
    with pytest.raises(preflight.V313PreflightError) as captured:
        function(*args, **kwargs)
    assert captured.value.code == expected_code
    assert str(captured.value) == expected_code


def _assert_failed_preflight_artifact(root: Path, code: str) -> dict[str, object]:
    payload = (root / contract.PREFLIGHT_ARTIFACT_PATH).read_bytes()
    value = json.loads(payload)
    assert payload == contract.canonical_json_bytes(value) + b"\n"
    assert preflight.validate_public_failed_preflight_artifact(value) == value
    assert value["status"] == "failed"
    assert value["failure_code"] == code
    assert value["preflight_consumed"] is True
    assert value["failure_preserved"] is True
    assert value["rerun_authorized"] is False
    assert value["development_authorized"] is False
    return value


def _source_inventory() -> list[dict[str, object]]:
    rows = []
    for index, path in enumerate(contract.IMPLEMENTATION_ALLOWED_PATHS, start=1):
        rows.append(
            {
                "path": path,
                "git_blob_sha1": f"{index:x}" * 40,
                "literal_sha256": f"{index:x}" * 64,
                "byte_count": index,
            }
        )
    return rows


def _repository_snapshot() -> dict[str, object]:
    rows = _source_inventory()
    predecessor = "e" * 64
    local_import_paths = sorted(
        {
            contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
            *contract.IMPLEMENTATION_PRODUCTION_PATHS,
        },
        key=lambda value: value.encode("utf-8"),
    )
    return {
        "ancestry": {
            "branch": contract.BRANCH_NAME,
            "commit": SHA1_A,
            "tree": SHA1_B,
            "parent": contract.PREREGISTRATION_COMMIT,
            "local_head": SHA1_A,
            "remote_head": SHA1_A,
            "changed_paths": {
                path: "A" for path in contract.IMPLEMENTATION_ALLOWED_PATHS
            },
            "clean_worktree": True,
            "preregistration_authenticated": True,
            "predecessor_blobs_unchanged": True,
        },
        "source_inventory": rows,
        "source_inventory_sha256": contract.canonical_sha256(rows),
        "production_source_inventory_sha256": contract.canonical_sha256(rows[:6]),
        "test_source_inventory_sha256": contract.canonical_sha256(rows[6:]),
        "predecessor_inventory_sha256": predecessor,
        "unchanged_predecessor_inventory_sha256": predecessor,
        "local_production_closure_manifest_sha256": (
            contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256
        ),
        "local_production_closure_paths_sha256": (
            contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256
        ),
        "local_import_paths": local_import_paths,
        "local_import_paths_sha256": contract.canonical_sha256(
            local_import_paths
        ),
        "package_initializer_predecessor_equal": True,
        "preregistration_authenticated": True,
    }


def _bridge_manifest() -> dict[str, object]:
    body = {
        "schema_version": "aapl-sec-gemma-lean-science-v3-13-streaming-bridge-v1",
        "stage": "development",
        "document_count": 75,
        "filename_present_count": 73,
        "filename_missing_count": 2,
        "source_authority_pins_sha256": contract.canonical_sha256(
            contract.build_source_authority_pins()
        ),
        "source_authority_base_commit": contract.V38_SOURCE_COMMIT,
        "source_authority_base_tree": contract.V38_SOURCE_TREE,
        "source_inventory_sha256": contract.V38_INVENTORY_SHA256,
        "stage_source_seal_sha256": contract.V38_STAGE_SOURCE_SEAL_SHA256,
        "checkpoint_sha256": contract.V38_LOGICAL_CHECKPOINT_SHA256,
        "compact_replay_sha256": contract.V38_COMPACT_REPLAY_SHA256,
        "role_manifests_sha256": contract.V38_ROLE_MANIFEST_INVENTORY_SHA256,
        "role_plan_sha256": contract.V38_ROLE_PLAN_SHA256,
        "science_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
        "legacy_projection_sha256": SHA64["legacy"],
        "compatibility_manifest_sha256": SHA64["compatibility"],
        "universe_sha256": SHA64["universe"],
        "content_manifest_sha256": SHA64["content"],
        "calendar_sessions_sha256": contract.CALENDAR_SESSIONS_SHA256,
        "universe_event_proofs_sha256": SHA64["proofs"],
        "documents_sha256": SHA64["documents"],
        "records_sha256": SHA64["records"],
        "source_order_sha256": SHA64["source_order"],
        "event_order_sha256": SHA64["events"],
        "prior_links_sha256": SHA64["prior_links"],
        "primary_documents_sha256": SHA64["primary"],
        "set_parity": True,
        "source_sequence_parity": True,
        "legacy_projection_parity": True,
        "typed_identity_parity": True,
        "nullable_filename_parity": True,
        "no_fabricated_primary_url": True,
        "prior_links_are_internal_only": True,
        "first_10k_and_10q_have_no_prior": True,
        "peak_live_complete_submission_blob_count": 1,
        "sec_request_count": 0,
        "confirmation_or_final_opened": False,
        "contains_private_rows": False,
        "contains_accessions_urls_filenames_or_bodies": False,
    }
    return {**body, "bridge_sha256": contract.canonical_sha256(body)}


def _commitments() -> dict[str, object]:
    return {
        "schema_version": preflight.REQUEST_COMMITMENTS_SCHEMA_VERSION,
        "stage": "development",
        "document_count": 75,
        "record_count": 75,
        "event_count": 75,
        "request_count": 75,
        "pilot_count": 5,
        "remaining_count": 70,
        "filename_present_count": 73,
        "filename_missing_count": 2,
        "documents_sha256": SHA64["documents"],
        "records_sha256": SHA64["records"],
        "events_sha256": SHA64["events"],
        "compatibility_manifest_sha256": SHA64["compatibility"],
        "universe_sha256": SHA64["universe"],
        "content_manifest_sha256": SHA64["content"],
        "calendar_sessions_sha256": contract.CALENDAR_SESSIONS_SHA256,
        "universe_event_proofs_sha256": SHA64["proofs"],
        "preprocessed_events_sha256": SHA64["preprocessed"],
        "canonical_requests_sha256": SHA64["requests"],
        "model_slice_sha256": SHA64["model_slice"],
        "canonical_request_index_sha256": SHA64["request_index"],
        "model_plan_sha256": SHA64["model_plan"],
        "pilot_order_sha256": SHA64["pilot"],
        "remaining_order_sha256": SHA64["remaining"],
        "minimum_request_byte_count": 1_000,
        "maximum_request_byte_count": 2_000,
        "source_order_sha256": SHA64["source_order"],
        "prior_links_sha256": SHA64["prior_links"],
        "set_parity": True,
        "sequence_parity": True,
        "prior_link_parity": True,
        "confirmation_or_final_opened": False,
        "contains_private_rows": False,
    }


def _qualification() -> dict[str, object]:
    phases = []
    for index, phase in enumerate(contract.QUALIFICATION_PHASES, start=1):
        if phase == contract.QUALIFICATION_PHASE_V313:
            node_count = contract.QUALIFICATION_LATEST_MIN_NODE_COUNT
            node_hash = "a" * 64
        elif phase == contract.QUALIFICATION_PHASE_DEPENDENCIES:
            node_count = contract.QUALIFICATION_SHARED_NODE_COUNT
            node_hash = contract.QUALIFICATION_SHARED_NODE_LIST_SHA256
        else:
            node_count = contract.QUALIFICATION_SENTINEL_NODE_COUNT
            node_hash = contract.QUALIFICATION_SENTINEL_NODE_LIST_SHA256
        phases.append(
            {
                "phase": phase,
                "node_count": node_count,
                "node_list_sha256": node_hash,
                "collection_duration_ns": index,
                "execution_duration_ns": index + 10,
                "collection_exit_code": 0,
                "execution_exit_code": 0,
                "passed_count": node_count,
                "failed_count": 0,
                "error_count": 0,
                "skipped_count": 0,
                "xfailed_count": 0,
                "xpassed_count": 0,
                "collection_result_sha256": "1" * 64,
                "execution_result_sha256": "2" * 64,
                "collection_log_sha256": "3" * 64,
                "execution_log_sha256": "4" * 64,
                "collection_xml_sha256": "5" * 64,
                "execution_xml_sha256": "6" * 64,
                "collection_manifest_sha256": "7" * 64,
            }
        )
    body = {
        "schema_version": preflight.QUALIFICATION_REPORT_SCHEMA_VERSION,
        "suite_identity": contract.QUALIFICATION_SUITE_ID,
        "status": "passed",
        "phase_count": len(phases),
        "deadline_seconds": contract.QUALIFICATION_TIMEOUT_SECONDS,
        "durability_mode": contract.QUALIFICATION_DURABILITY_MODE,
        "command_profile_sha256": preflight.QUALIFICATION_COMMAND_PROFILE_SHA256,
        "phases": phases,
        "private_aggregate_sha256": "f" * 64,
    }
    return {**body, "qualification_sha256": contract.canonical_sha256(body)}


_SHARED_NODE_IDS_CACHE: list[str] | None = None


def _shared_node_ids() -> list[str]:
    global _SHARED_NODE_IDS_CACHE
    if _SHARED_NODE_IDS_CACHE is None:
        root = Path(__file__).resolve().parents[1]
        runtime = _qualification_runtime_fixture()
        environment, _profile = preflight._qualification_environment(runtime)
        dispatch = [
            str(root)
            if token == contract.QUALIFICATION_CANONICAL_ROOT_TOKEN
            else token
            for token in contract.QUALIFICATION_DISPATCH_PREFIX
        ]
        completed = subprocess.run(
            [
                sys.executable,
                *contract.QUALIFICATION_PYTHON_FLAGS,
                "-c",
                contract.QUALIFICATION_BOOTSTRAP_LITERAL,
                *dispatch,
                *contract.QUALIFICATION_PYTEST_ARGS,
                "--collect-only",
                *contract.QUALIFICATION_SHARED_TEST_PATHS,
            ],
            cwd=root,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _SHARED_NODE_IDS_CACHE = preflight._qualification_collection_node_ids(
            completed.stdout
        )
        assert len(_SHARED_NODE_IDS_CACHE) == (
            contract.QUALIFICATION_SHARED_NODE_COUNT
        )
        assert hashlib.sha256(
            "\n".join(_SHARED_NODE_IDS_CACHE).encode("utf-8")
        ).hexdigest() == contract.QUALIFICATION_SHARED_NODE_LIST_SHA256
        multiplicities = Counter(_SHARED_NODE_IDS_CACHE)
        assert len(multiplicities) == contract.QUALIFICATION_SHARED_UNIQUE_NODE_COUNT
        assert {
            node_id: count
            for node_id, count in multiplicities.items()
            if count != 1
        } == {}
    return list(_SHARED_NODE_IDS_CACHE)


def _qualification_runtime_fixture() -> dict[str, object]:
    return {
        "python_version": contract.QUALIFICATION_PYTHON_VERSION,
        "python_cache_tag": contract.QUALIFICATION_PYTHON_CACHE_TAG,
        "os_name": contract.QUALIFICATION_OS_NAME,
        "sys_platform": contract.QUALIFICATION_SYS_PLATFORM,
        "executable_path": str(Path(sys.executable).resolve()),
        "executable_basename": contract.QUALIFICATION_EXECUTABLE_BASENAME,
        "executable_bytes": contract.QUALIFICATION_EXECUTABLE_BYTES,
        "executable_sha256": contract.QUALIFICATION_EXECUTABLE_SHA256,
        "pytest_version": contract.QUALIFICATION_PYTEST_VERSION,
        "pytest_init_path": str(Path(pytest.__file__).resolve()),
        "pytest_init_bytes": contract.QUALIFICATION_PYTEST_INIT_BYTES,
        "pytest_init_sha256": contract.QUALIFICATION_PYTEST_INIT_SHA256,
    }


def _freeze_qualification_environment_for_process_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, profile = preflight._qualification_environment(
        _qualification_runtime_fixture()
    )
    monkeypatch.setattr(
        preflight,
        "_qualification_environment",
        lambda _runtime: (dict(environment), dict(profile)),
    )


def _runtime_authority_fixture() -> dict[str, object]:
    return preflight.build_runtime_binding_authority(
        repository_runtime_manifest_sha256="a" * 64,
        execution_dependency_manifest_sha256="b" * 64,
        checkpoint_loaded_code_manifest_sha256s={
            name: hashlib.sha256(name.encode("utf-8")).hexdigest()
            for name in preflight.RUNTIME_CHECKPOINT_NAMES
        },
        process_environment_sha256="c" * 64,
        final_sys_path_tokens=["d" * 64],
    )


def _fingerprint_context() -> preflight._RuntimeFingerprintContext:
    context = object.__new__(preflight._RuntimeFingerprintContext)
    context.repo_rows = {}
    context.repo_origins = {}
    context.dependency_origins = {
        name: hashlib.sha256(name.encode("ascii")).hexdigest()
        for name in ("builtins", "collections.abc", "typing", "weakref")
    }
    context.dependency_files = {}
    context.binary_by_path = {}
    context.python_dll_sha256 = "a" * 64
    context.timezones = {}
    context.code_cache = {}
    context.value_cache = {}
    context.active = set()
    context.identity_sentinels = {}
    context.lock_owners = {}
    context.weak_registry_owners = {}
    context.weak_self_refs = {}
    context.relative_path_bindings = {}
    context.typing_alias_bindings = {}
    context.owner_slots = {}
    context.owner_slot_hashes = {}
    context.referenced_callables = {}
    return context


def _minimal_loaded_code_manifest(checkpoint_name: str) -> dict[str, object]:
    module_names = sorted(
        (
            path[: -len("/__init__.py")].replace("/", ".")
            if path.endswith("/__init__.py")
            else path[:-3].replace("/", ".")
        )
        for path in (
            *contract.IMPLEMENTATION_PRODUCTION_PATHS,
            *contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
            contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
        )
    )
    module_rows = [
        {
            "module_name": name,
            "origin_kind": "repository_source",
            "origin_identity_sha256": hashlib.sha256(
                f"origin:{name}".encode("utf-8")
            ).hexdigest(),
            "namespace_sha256": hashlib.sha256(
                f"namespace:{name}".encode("utf-8")
            ).hexdigest(),
        }
        for name in module_names
    ]
    callable_rows = [
        {
            "qualified_name": "synthetic_function",
            "kind": "function",
            "owner_module": module_names[0],
            "owner_file_sha256": "1" * 64,
            "object_type": "builtins.function",
            "code_sha256": "2" * 64,
            "defaults_sha256": "3" * 64,
            "kwdefaults_sha256": "4" * 64,
            "annotations_sha256": "5" * 64,
            "closure_sha256": "6" * 64,
            "referenced_globals_sha256": "7" * 64,
            "descriptor_members_sha256": None,
            "owning_binary_sha256": None,
            "owner_slots_sha256": "0" * 64,
        }
    ]
    body = {
        "schema_version": preflight.LOADED_CODE_MANIFEST_SCHEMA_VERSION,
        "checkpoint_name": checkpoint_name,
        "repository_runtime_manifest_sha256": "8" * 64,
        "execution_dependency_manifest_sha256": "9" * 64,
        "module_count": len(module_rows),
        "module_rows": module_rows,
        "callable_count": len(callable_rows),
        "callable_rows": callable_rows,
    }
    return {
        **body,
        "loaded_code_manifest_sha256": contract.canonical_sha256(body),
    }


def test_runtime_fingerprint_relative_paths_require_a_sealed_exact_adapter() -> None:
    value = Path("authority") / "manifest.json"
    _assert_code(
        "runtime_fingerprint_relative_path_invalid",
        _fingerprint_context().value,
        value,
    )
    context = _fingerprint_context()
    context.relative_path_bindings[id(value)] = (
        value,
        (("synthetic.module", "MANIFEST"),),
        "authority/manifest.json",
    )
    assert context.value(value).startswith(b"relative_path[")


def test_runtime_fingerprint_timedelta_uses_all_exact_integer_fields() -> None:
    context = _fingerprint_context()
    one = context.value(preflight._datetime.timedelta(days=1, seconds=2, microseconds=3))
    two = _fingerprint_context().value(
        preflight._datetime.timedelta(days=1, seconds=2, microseconds=4)
    )
    assert one != two
    assert all(token in one for token in (b"1", b"2", b"3"))


def test_runtime_fingerprint_value_cache_keeps_a_strong_object_reference() -> None:
    context = _fingerprint_context()
    value = ["sealed"]
    encoded = context.value(value)
    cached_object, cached_bytes = context.value_cache[id(value)]
    assert cached_object is value
    assert cached_bytes == encoded
    assert context.value(value) == encoded


def test_runtime_fingerprint_typing_semantic_and_stateful_callable_rules() -> None:
    assert preflight._runtime_is_typing_semantic(typing.Mapping)
    _assert_code(
        "runtime_fingerprint_typing_alias_invalid",
        _fingerprint_context().value,
        typing.Mapping,
    )

    class StatefulCallable:
        def __call__(self) -> None:
            return None

    _assert_code(
        "runtime_fingerprint_stateful_callable_unsupported",
        _fingerprint_context().value,
        StatefulCallable(),
    )


def test_runtime_fingerprint_weak_registry_rejects_live_entries() -> None:
    class Key:
        pass

    registry: weakref.WeakKeyDictionary[object, object] = weakref.WeakKeyDictionary()
    key = Key()
    registry[key] = "temporary capability state"
    _assert_code(
        "runtime_weak_registry_invalid",
        preflight._runtime_validate_weak_registry,
        registry,
    )


def test_runtime_weak_registry_rejects_dirty_pending_foreign_ref_without_laundering() -> None:
    class Key:
        pass

    registry: weakref.WeakKeyDictionary[object, object] = weakref.WeakKeyDictionary()
    foreign_key = Key()
    foreign_ref = weakref.ref(foreign_key)
    raw = vars(registry)
    raw["_pending_removals"].append(foreign_ref)
    raw["_dirty_len"] = True
    pending_before = list(raw["_pending_removals"])
    _assert_code(
        "runtime_weak_registry_invalid",
        preflight._runtime_validate_weak_registry,
        registry,
    )
    assert raw["_pending_removals"] == pending_before
    assert raw["_pending_removals"][0] is foreign_ref
    assert raw["_dirty_len"] is True


def test_runtime_callable_qualification_distinguishes_same_named_objects() -> None:
    def first() -> None:
        return None

    def second() -> None:
        return None

    first.__qualname__ = "generated"
    second.__qualname__ = "generated"
    context = _fingerprint_context()
    context.dependency_origins[__name__] = hashlib.sha256(
        __name__.encode("utf-8")
    ).hexdigest()
    context.owner_slot_hashes[id(first)] = (first, "1" * 64)
    context.owner_slot_hashes[id(second)] = (second, "2" * 64)
    assert context.callable_reference(first) != context.callable_reference(second)


def test_runtime_pseudo_module_normalization_requires_all_four_once() -> None:
    import pyexpat

    aliases = {
        "pyexpat.errors": pyexpat.errors,
        "pyexpat.model": pyexpat.model,
        "typing.io": typing.io,
        "typing.re": typing.re,
    }
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in aliases}
    process_state = preflight._runtime_process_state()
    saved_marker = process_state["pseudo_module_normalization"]
    try:
        process_state["pseudo_module_normalization"] = None
        sys.modules.update(aliases)
        sealed = preflight._runtime_normalize_pseudo_module_aliases()
        assert set(sealed) == set(aliases)
        assert all(name not in sys.modules for name in aliases)
        rows = preflight._runtime_pseudo_module_alias_rows(
            sealed,
            owner_origins={"pyexpat": "1" * 64, "typing": "2" * 64},
        )
        assert [row["alias_name"] for row in rows] == sorted(aliases)
        assert all(len(row["row_sha256"]) == 64 for row in rows)
        _assert_code(
            "runtime_pseudo_module_alias_invalid",
            preflight._runtime_normalize_pseudo_module_aliases,
        )
        sys.modules.update(aliases)
        _assert_code(
            "runtime_pseudo_module_alias_invalid",
            preflight._runtime_normalize_pseudo_module_aliases,
        )
        assert all(sys.modules[name] is value for name, value in aliases.items())
        sys.modules.pop("typing.io")
        _assert_code(
            "runtime_pseudo_module_alias_invalid",
            preflight._runtime_normalize_pseudo_module_aliases,
        )
    finally:
        process_state["pseudo_module_normalization"] = saved_marker
        for name, value in saved.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


_PRODUCTION_OWNER_GRAPH_CACHE: dict[str, object] | None = None


def _production_owner_graph_fixture() -> dict[str, object]:
    global _PRODUCTION_OWNER_GRAPH_CACHE
    if _PRODUCTION_OWNER_GRAPH_CACHE is None:
        module_names = [
            (
                path[: -len("/__init__.py")].replace("/", ".")
                if path.endswith("/__init__.py")
                else path[:-3].replace("/", ".")
            )
            for path in (
                *contract.IMPLEMENTATION_PRODUCTION_PATHS,
                *contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
                contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
            )
        ]
        repo_modules = {
            name: importlib.import_module(name)
            for name in module_names
        }
        assert len(repo_modules) == 44
        relative_paths = preflight._runtime_bind_relative_path_adapters(repo_modules)
        typing_aliases = preflight._runtime_bind_typing_alias_adapters(repo_modules)
        names_by_module, roots = preflight._runtime_collect_namespace_roots(
            repo_modules
        )
        graph = preflight._runtime_build_owner_graph(
            repo_modules=repo_modules,
            roots=roots,
        )
        closure_state = preflight._runtime_validate_closure_state_allowlist(graph)
        _PRODUCTION_OWNER_GRAPH_CACHE = {
            "repo_modules": repo_modules,
            "relative_paths": relative_paths,
            "typing_aliases": typing_aliases,
            "names_by_module": names_by_module,
            "roots": roots,
            "graph": graph,
            "closure_state": closure_state,
        }
    return _PRODUCTION_OWNER_GRAPH_CACHE


def test_runtime_owner_graph_has_no_discovered_pseudo_slots_or_unrooted_callables() -> None:
    fixture = _production_owner_graph_fixture()
    graph = fixture["graph"]
    assert set(graph["strong_objects"]) == set(graph["canonical_paths"])
    assert set(graph["strong_objects"]) == set(graph["owner_slots"])
    assert set(graph["strong_objects"]) == set(graph["owner_slot_hashes"])
    for identifier, value in graph["strong_objects"].items():
        assert graph["owner_slots"][identifier][0] is value
        assert graph["owner_slot_hashes"][identifier][0] is value
        slots = graph["owner_slots"][identifier][1]
        assert slots
        assert all(
            "discovered:" not in field
            for slot in slots
            for field in slot
        )
        if preflight._runtime_is_supported_callable(value):
            assert len(graph["owner_slot_hashes"][identifier][1]) == 64


def test_runtime_owner_slots_disambiguate_every_same_natural_callable_identity() -> None:
    graph = _production_owner_graph_fixture()["graph"]
    groups: dict[tuple[str, str, str], list[object]] = {}
    for value in graph["strong_objects"].values():
        if not preflight._runtime_is_supported_callable(value):
            continue
        owner = getattr(value, "__module__", type(value).__module__)
        natural = getattr(value, "__qualname__", type(value).__qualname__)
        groups.setdefault(
            (owner, natural, preflight._runtime_callable_kind(value)), []
        ).append(value)
    collisions = [values for values in groups.values() if len(values) > 1]
    assert collisions
    for values in collisions:
        hashes = {
            graph["owner_slot_hashes"][id(value)][1]
            for value in values
        }
        assert len(hashes) == len(values)


def test_runtime_closure_state_allowlist_binds_exact_five_objects_and_slots() -> None:
    fixture = _production_owner_graph_fixture()
    graph = fixture["graph"]
    state = fixture["closure_state"]
    labels = {
        "issuer",
        "missing",
        "acquisition_results",
        "capability_bindings",
        "registry_lock",
    }
    assert len({id(state[label]) for label in labels}) == 5
    expected_closure_counts = {
        "issuer": 3,
        "missing": 3,
        "acquisition_results": 2,
        "capability_bindings": 3,
        "registry_lock": 4,
    }
    for label, count in expected_closure_counts.items():
        slots = graph["owner_slots"][id(state[label])][1]
        assert len(
            [slot for slot in slots if slot[2] == "function_closure"]
        ) == count


def test_runtime_weak_registries_are_empty_callback_bound_and_identity_continuous() -> None:
    fixture = _production_owner_graph_fixture()
    graph = fixture["graph"]
    state = fixture["closure_state"]
    callbacks = []
    self_refs = []
    for label in ("acquisition_results", "capability_bindings"):
        registry = state[label]
        first = preflight._runtime_validate_weak_registry(registry)
        second = preflight._runtime_validate_weak_registry(registry)
        assert first["callback"] is second["callback"]
        assert first["self_ref"] is second["self_ref"]
        assert graph["strong_objects"][id(first["callback"])] is first["callback"]
        assert graph["strong_objects"][id(first["self_ref"])] is first["self_ref"]
        callbacks.append(first["callback"])
        self_refs.append(first["self_ref"])
    assert callbacks[0] is not callbacks[1]
    assert self_refs[0] is not self_refs[1]


def test_runtime_relative_path_allowlist_is_exact_and_alias_bound() -> None:
    fixture = _production_owner_graph_fixture()
    adapters = fixture["relative_paths"]
    assert len(adapters) == 4
    assert sum(len(bindings) for _value, bindings, _posix in adapters.values()) == 5
    aliased = [
        (value, bindings)
        for value, bindings, _posix in adapters.values()
        if len(bindings) == 2
    ]
    assert len(aliased) == 1
    assert {binding[1] for binding in aliased[0][1]} == {
        "STATE_RELATIVE_DIRECTORY"
    }
    for value, _bindings, posix_value in adapters.values():
        assert type(value).__qualname__ == "WindowsPath"
        assert value.as_posix() == posix_value


def test_runtime_typing_alias_allowlist_is_exact_and_identity_bound() -> None:
    fixture = _production_owner_graph_fixture()
    graph = fixture["graph"]
    adapters = fixture["typing_aliases"]
    assert len(adapters) == 2
    by_name = {alias_name: (value, bindings) for value, bindings, alias_name in adapters.values()}
    assert by_name["Mapping"][0] is typing.Mapping
    assert by_name["Sequence"][0] is typing.Sequence
    assert len(by_name["Mapping"][1]) == 8
    assert len(by_name["Sequence"][1]) == 5
    preflight._runtime_validate_typing_alias(typing.Mapping, alias_name="Mapping")
    preflight._runtime_validate_typing_alias(typing.Sequence, alias_name="Sequence")
    context = _fingerprint_context()
    context.typing_alias_bindings = dict(adapters)
    context.owner_slot_hashes = dict(graph["owner_slot_hashes"])
    owner_hash = graph["owner_slot_hashes"][id(typing.Mapping)][1]
    encoded = context.value(typing.Mapping)
    assert owner_hash.encode("ascii") in encoded

    missing = _fingerprint_context()
    missing.typing_alias_bindings = dict(adapters)
    missing.owner_slot_hashes = {
        identifier: row
        for identifier, row in graph["owner_slot_hashes"].items()
        if identifier != id(typing.Mapping)
    }
    _assert_code(
        "runtime_fingerprint_typing_alias_owner_slots_invalid",
        missing.value,
        typing.Mapping,
    )

    changed = _fingerprint_context()
    changed.typing_alias_bindings = dict(adapters)
    changed.owner_slot_hashes = dict(graph["owner_slot_hashes"])
    changed.owner_slot_hashes[id(typing.Mapping)] = (typing.Mapping, "f" * 64)
    assert changed.value(typing.Mapping) != encoded


def _full_production_loaded_code_probe(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    repo_modules = fixture["repo_modules"]
    graph = fixture["graph"]
    canonical_requests = sys.modules["requests"]
    for module_name in (
        "agent_benchmark.sec_gemma_online_risk_overlay_production",
        "agent_benchmark.sec_filing_gemma_ollama",
    ):
        holder = sys.modules.get(module_name)
        if holder is not None:
            monkeypatch.setattr(holder, "requests", canonical_requests)
    path_rows = (
        *contract.IMPLEMENTATION_PRODUCTION_PATHS,
        *contract.LOCAL_PRODUCTION_CLOSURE_PATHS,
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
    )
    path_by_module = {
        (
            path[: -len("/__init__.py")].replace("/", ".")
            if path.endswith("/__init__.py")
            else path[:-3].replace("/", ".")
        ): path
        for path in path_rows
    }
    implementation_commit = "1" * 40
    implementation_tree = "2" * 40
    repository_rows = []
    for path in sorted(path_rows, key=lambda value: value.encode("utf-8")):
        module_name = (
            path[: -len("/__init__.py")].replace("/", ".")
            if path.endswith("/__init__.py")
            else path[:-3].replace("/", ".")
        )
        role = (
            "v313"
            if path in contract.IMPLEMENTATION_PRODUCTION_PATHS
            else "package_bootstrap"
            if path == contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH
            else "shared"
        )
        repository_rows.append(
            {
                "module_name": module_name,
                "relative_path": path,
                "role": role,
                "owning_revision": (
                    implementation_commit if role == "v313" else contract.BASE_COMMIT
                ),
                "git_blob_sha1": hashlib.sha1(path.encode("utf-8")).hexdigest(),
                "literal_sha256": hashlib.sha256(
                    f"repository:{module_name}".encode("utf-8")
                ).hexdigest(),
                "byte_count": 1,
            }
        )
    repository_body = {
        "schema_version": preflight.REPOSITORY_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "branch": contract.BRANCH_NAME,
        "origin_url": preflight.CREATION_PRETRUST_EXPECTED_ORIGIN_URL,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "successor_base_commit": contract.BASE_COMMIT,
        "shared_path_list_sha256": contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256,
        "module_count": 44,
        "modules": repository_rows,
    }
    repository = {
        **repository_body,
        "repository_runtime_manifest_sha256": contract.canonical_sha256(
            repository_body
        ),
    }

    external_names = {
        module.__name__
        for module in sys.modules.values()
        if type(module) is types.ModuleType
        and type(getattr(module, "__name__", None)) is str
    }
    external_names.update(
        (
            owner
            if type(owner := getattr(value, "__module__", None)) is str
            else type(value).__module__
        )
        for value in graph["strong_objects"].values()
        if preflight._runtime_is_supported_callable(value)
    )
    external_names.difference_update(repo_modules)
    builtin_rows = [
        {"module_name": name, "origin_kind": "builtin"}
        for name in sorted(external_names, key=lambda value: value.encode("utf-8"))
    ]
    owner_origins = {
        row["module_name"]: contract.canonical_sha256(row)
        for row in builtin_rows
    }

    import pyexpat

    aliases = {
        "pyexpat.errors": pyexpat.errors,
        "pyexpat.model": pyexpat.model,
        "typing.io": typing.io,
        "typing.re": typing.re,
    }
    missing_alias = object()
    saved_aliases = {
        name: sys.modules.get(name, missing_alias) for name in aliases
    }
    process_state = preflight._runtime_process_state()
    saved_marker = process_state["pseudo_module_normalization"]
    try:
        process_state["pseudo_module_normalization"] = None
        sys.modules.update(aliases)
        sealed = preflight._runtime_normalize_pseudo_module_aliases()
        pseudo_rows = preflight._runtime_pseudo_module_alias_rows(
            sealed,
            owner_origins={
                "pyexpat": owner_origins["pyexpat"],
                "typing": owner_origins["typing"],
            },
        )
        flag_names = (
            "debug", "inspect", "interactive", "optimize", "dont_write_bytecode",
            "no_user_site", "no_site", "ignore_environment", "verbose",
            "bytes_warning", "quiet", "hash_randomization", "isolated",
            "dev_mode", "utf8_mode", "warn_default_encoding", "safe_path",
            "int_max_str_digits",
        )
        flag_values = (
            0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, False, 1, 0, False, 4300
        )
        executable_sha256 = "3" * 64
        dll_path_sha256 = "4" * 64
        dll_sha256 = "5" * 64
        launcher = preflight.build_launcher_profile(
            python_executable_sha256=executable_sha256,
            bootstrap_bytes=preflight.SCIENTIFIC_BOOTSTRAP_BYTES,
        )
        python_runtime = {
            "executable_basename": contract.QUALIFICATION_EXECUTABLE_BASENAME,
            "executable_path_sha256": "6" * 64,
            "executable_bytes": 1,
            "executable_sha256": executable_sha256,
            "python_dll_basename": "python312.dll",
            "python_dll_path_sha256": dll_path_sha256,
            "python_dll_bytes": 1,
            "python_dll_sha256": dll_sha256,
            "python_version": contract.QUALIFICATION_PYTHON_VERSION,
            "cache_tag": contract.QUALIFICATION_PYTHON_CACHE_TAG,
            "os_name": contract.QUALIFICATION_OS_NAME,
            "sys_platform": contract.QUALIFICATION_SYS_PLATFORM,
            "launcher_profile_sha256": launcher["launcher_profile_sha256"],
            "bootstrap_sha256": preflight.SCIENTIFIC_BOOTSTRAP_SHA256,
            "flags_sha256": contract.canonical_sha256(
                dict(zip(flag_names, flag_values, strict=True))
            ),
            "process_environment_sha256": "7" * 64,
        }
        git_runtime = {
            "basename": "git.exe",
            "resolved_path_sha256": "8" * 64,
            "byte_count": 1,
            "literal_sha256": "9" * 64,
            "version_output_sha256": "a" * 64,
        }
        loaded_binaries = [
            {
                "basename": "python312.dll",
                "resolved_path_sha256": dll_path_sha256,
                "byte_count": 1,
                "literal_sha256": dll_sha256,
            }
        ]
        timezone_rows = [
            {
                "zone_key": "America/Chicago",
                "provider": "tzdata:synthetic",
                "resolved_path_sha256": None,
                "byte_count": 1,
                "literal_sha256": "b" * 64,
            },
            {
                "zone_key": "America/New_York",
                "provider": "tzdata:synthetic",
                "resolved_path_sha256": None,
                "byte_count": 1,
                "literal_sha256": "c" * 64,
            },
        ]
        counts = {
            "builtin_or_frozen_modules": len(builtin_rows),
            "module_files": 0,
            "distributions": 0,
            "loaded_binaries": 1,
            "timezone_files": 2,
        }
        dependency_body = {
            "schema_version": preflight.EXECUTION_DEPENDENCY_MANIFEST_SCHEMA_VERSION,
            "python_runtime": python_runtime,
            "git_runtime": git_runtime,
            "builtin_or_frozen_modules": builtin_rows,
            "module_files": [],
            "distributions": [],
            "loaded_binaries": loaded_binaries,
            "timezone_files": timezone_rows,
            "normalized_pseudo_module_aliases": pseudo_rows,
            "sys_path_sha256": "d" * 64,
            "counts": counts,
        }
        dependency = {
            **dependency_body,
            "execution_dependency_manifest_sha256": contract.canonical_sha256(
                dependency_body
            ),
        }
        manifest, baseline = preflight._build_loaded_code_manifest(
            checkpoint_name=preflight.RUNTIME_CHECKPOINT_NAMES[0],
            repository_manifest=repository,
            dependency_manifest=dependency,
            critical_objects=preflight._runtime_live_critical_objects(),
        )
    finally:
        process_state["pseudo_module_normalization"] = saved_marker
        for name, value in saved_aliases.items():
            if value is missing_alias:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return manifest, baseline


def test_loaded_code_v2_callable_rows_bind_owner_slot_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_owner_graph_fixture()
    graph = fixture["graph"]
    repo_modules = fixture["repo_modules"]
    closure_state = fixture["closure_state"]
    context = _fingerprint_context()
    context.repo_rows = {
        name: {
            "relative_path": name.replace(".", "/") + ".py",
            "literal_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        }
        for name in repo_modules
    }
    context.repo_origins = {
        name: hashlib.sha256(f"repo:{name}".encode("utf-8")).hexdigest()
        for name in repo_modules
    }
    external_modules = {
        getattr(value, "__module__", type(value).__module__)
        for value in graph["strong_objects"].values()
        if preflight._runtime_is_supported_callable(value)
    }
    external_modules.update(
        module.__name__
        for module in sys.modules.values()
        if type(module) is types.ModuleType
        and type(getattr(module, "__name__", None)) is str
    )
    context.dependency_origins.update(
        {
            name: hashlib.sha256(f"dependency:{name}".encode("utf-8")).hexdigest()
            for name in external_modules
            if name not in repo_modules
        }
    )
    context.dependency_files = {
        name: {
            "origin_kind": "stdlib_source",
            "resolved_path_sha256": hashlib.sha256(
                f"path:{name}".encode("utf-8")
            ).hexdigest(),
            "member_name": None,
            "literal_sha256": hashlib.sha256(
                f"file:{name}".encode("utf-8")
            ).hexdigest(),
        }
        for name in context.dependency_origins
        if name not in {"builtins", "_thread"}
    }
    context.timezones = {
        "America/Chicago": "a" * 64,
        "America/New_York": "b" * 64,
    }
    context.owner_slots = dict(graph["owner_slots"])
    context.owner_slot_hashes = dict(graph["owner_slot_hashes"])
    context.relative_path_bindings = dict(fixture["relative_paths"])
    context.typing_alias_bindings = dict(fixture["typing_aliases"])
    closure_ids = {
        id(closure_state[label])
        for label in (
            "issuer",
            "missing",
            "acquisition_results",
            "capability_bindings",
            "registry_lock",
        )
    }
    for identifier, value in graph["strong_objects"].items():
        if type(value) not in {
            object,
            weakref.WeakKeyDictionary,
            preflight._RUNTIME_LOCK_TYPE,
            preflight._RUNTIME_RLOCK_TYPE,
        }:
            continue
        slots = graph["owner_slots"][identifier][1]
        if identifier not in closure_ids:
            assert len([slot for slot in slots if slot[0] == "root"]) == 1
        owner_hash = graph["owner_slot_hashes"][identifier][1]
        if type(value) is object:
            context.identity_sentinels[identifier] = (value, owner_hash)
        elif type(value) in {
            preflight._RUNTIME_LOCK_TYPE,
            preflight._RUNTIME_RLOCK_TYPE,
        }:
            context.lock_owners[identifier] = (value, owner_hash)
        else:
            context.weak_registry_owners[identifier] = (value, owner_hash)
    for label, details in closure_state["registry_details"].items():
        context.weak_self_refs[id(details["self_ref"])] = (
            details["self_ref"],
            closure_state[label],
        )
    callable_keys = []
    references = []
    rows = []
    for value in graph["strong_objects"].values():
        if not preflight._runtime_is_supported_callable(value):
            continue
        owner = getattr(value, "__module__", type(value).__module__)
        natural = getattr(value, "__qualname__", type(value).__qualname__)
        owner_hash = graph["owner_slot_hashes"][id(value)][1]
        callable_keys.append(
            (owner, natural, preflight._runtime_callable_kind(value), owner_hash)
        )
        references.append(context.callable_reference(value))
        rows.append(
            preflight._runtime_function_row(value, context=context)
            if owner in repo_modules and inspect.isfunction(value)
            else preflight._runtime_class_row(value, context=context)
            if owner in repo_modules and inspect.isclass(value)
            else preflight._runtime_external_callable_row(value, context=context)
        )
    assert len(callable_keys) == len(set(callable_keys))
    assert len(references) == len(callable_keys)
    assert len(rows) == len(callable_keys)
    assert all(key[3] and len(key[3]) == 64 for key in callable_keys)
    assert all(reference.startswith(b"callable_reference[") for reference in references)
    assert all(row["owner_slots_sha256"] for row in rows)
    manifest, baseline = _full_production_loaded_code_probe(monkeypatch, fixture)
    validated = preflight.validate_loaded_code_manifest(manifest)
    assert validated["schema_version"] == preflight.LOADED_CODE_MANIFEST_SCHEMA_VERSION
    assert validated["module_count"] == 44
    assert validated["callable_count"] == len(callable_keys)
    assert set(baseline["callable_identity"]) == set(callable_keys)
    assert all(row["owner_slots_sha256"] for row in validated["callable_rows"])


@pytest.mark.parametrize(
    "replacement_kind",
    ("requests_session", "requests_adapter", "requests_module", "verifier_module"),
)
def test_runtime_checkpoint_rebuilds_critical_identities_from_live_owners(
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    _production_owner_graph_fixture()
    canonical_requests = sys.modules["requests"]
    for module_name in (
        "agent_benchmark.sec_gemma_online_risk_overlay_production",
        "agent_benchmark.sec_filing_gemma_ollama",
    ):
        holder = sys.modules.get(module_name)
        if holder is not None:
            monkeypatch.setattr(holder, "requests", canonical_requests)
    baseline = preflight._runtime_live_critical_objects()
    requests_module = baseline["requests_module"]
    sessions_module = sys.modules["requests.sessions"]
    adapters_module = sys.modules["requests.adapters"]
    if replacement_kind == "requests_session":
        replacement = type("ReplacementSession", (), {})
        monkeypatch.setattr(requests_module, "Session", replacement)
        monkeypatch.setattr(sessions_module, "Session", replacement)
    elif replacement_kind == "requests_adapter":
        replacement = type("ReplacementAdapter", (), {})
        monkeypatch.setattr(adapters_module, "HTTPAdapter", replacement)
    elif replacement_kind == "requests_module":
        replacement_module = types.ModuleType("requests")
        replacement_module.Session = baseline["requests_session_class"]
        replacement_module.adapters = adapters_module
        monkeypatch.setitem(sys.modules, "requests", replacement_module)
        for module_name in (
            "agent_benchmark.sec_gemma_online_risk_overlay_production",
            "agent_benchmark.sec_filing_gemma_ollama",
        ):
            holder = sys.modules.get(module_name)
            if holder is not None:
                monkeypatch.setattr(holder, "requests", replacement_module)
    else:
        verifier_name = (
            "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier"
        )
        monkeypatch.setitem(sys.modules, verifier_name, types.ModuleType(verifier_name))
    observed = preflight._runtime_live_critical_objects()
    _assert_code(
        "runtime_binding_identity_mismatch",
        preflight._runtime_require_identity_map,
        baseline,
        observed,
    )


def test_runtime_process_state_is_out_of_namespace_and_sentinel_bound() -> None:
    name = preflight._RUNTIME_PROCESS_STATE_NAME
    missing = object()
    saved = getattr(builtins, name, missing)
    original_sentinel = preflight._RUNTIME_PROCESS_STATE_SENTINEL
    try:
        if hasattr(builtins, name):
            delattr(builtins, name)
        state = preflight._runtime_process_state()
        state["active_preflight"]["temporary"] = True
        assert preflight._runtime_process_state() is state
        preflight._RUNTIME_PROCESS_STATE_SENTINEL = object()
        _assert_code(
            "runtime_process_state_invalid",
            preflight._runtime_process_state,
        )
    finally:
        preflight._RUNTIME_PROCESS_STATE_SENTINEL = original_sentinel
        if saved is missing:
            if hasattr(builtins, name):
                delattr(builtins, name)
        else:
            setattr(builtins, name, saved)


def test_loaded_code_checkpoint_manifests_change_only_name_and_self_hash() -> None:
    manifests = {
        name: preflight.validate_loaded_code_manifest(
            _minimal_loaded_code_manifest(name)
        )
        for name in preflight.RUNTIME_CHECKPOINT_NAMES
    }
    assert tuple(manifests) == preflight.RUNTIME_CHECKPOINT_NAMES
    assert len(
        {
            manifest["loaded_code_manifest_sha256"]
            for manifest in manifests.values()
        }
    ) == len(preflight.RUNTIME_CHECKPOINT_NAMES)
    normalized = []
    for manifest in manifests.values():
        body = copy.deepcopy(manifest)
        body.pop("loaded_code_manifest_sha256")
        body.pop("checkpoint_name")
        normalized.append(body)
    assert all(value == normalized[0] for value in normalized[1:])


@pytest.mark.parametrize(
    ("mutation",),
    (
        (lambda item: item["module_rows"].reverse(),),
        (lambda item: item["module_rows"].append(copy.deepcopy(item["module_rows"][0])),),
        (lambda item: item["callable_rows"][0].__setitem__("kind", "callable"),),
        (lambda item: item["callable_rows"][0].__setitem__("extra", True),),
    ),
)
def test_loaded_code_validator_rejects_nested_order_duplicates_kinds_and_fields(
    mutation,
) -> None:
    manifest = _minimal_loaded_code_manifest(preflight.RUNTIME_CHECKPOINT_NAMES[0])
    mutation(manifest)
    body = dict(manifest)
    body.pop("loaded_code_manifest_sha256")
    manifest["loaded_code_manifest_sha256"] = contract.canonical_sha256(body)
    _assert_code(
        "loaded_code_manifest_invalid",
        preflight.validate_loaded_code_manifest,
        manifest,
    )


def _write_runtime_authority_bundle(root: Path) -> dict[str, object]:
    runtime_root = (
        root
        / contract.PRIVATE_PREFLIGHT_NAMESPACE
        / preflight.PRIVATE_RUNTIME_DIRECTORY
    )
    loaded_root = runtime_root / preflight.PRIVATE_RUNTIME_LOADED_CODE_DIRECTORY
    loaded_root.mkdir(parents=True)
    repository = preflight._self_hash(
        {
            "schema_version": preflight.REPOSITORY_RUNTIME_MANIFEST_SCHEMA_VERSION,
            "synthetic_offline_fixture": True,
        },
        "repository_runtime_manifest_sha256",
    )
    dependency = preflight._self_hash(
        {
            "schema_version": preflight.EXECUTION_DEPENDENCY_MANIFEST_SCHEMA_VERSION,
            "synthetic_offline_fixture": True,
        },
        "execution_dependency_manifest_sha256",
    )
    loaded: dict[str, dict[str, object]] = {}
    for name in preflight.RUNTIME_CHECKPOINT_NAMES:
        manifest = preflight._self_hash(
            {
                "schema_version": preflight.LOADED_CODE_MANIFEST_SCHEMA_VERSION,
                "checkpoint_name": name,
                "synthetic_offline_fixture": True,
            },
            "loaded_code_manifest_sha256",
        )
        loaded[name] = manifest
        (loaded_root / f"{name}.json").write_bytes(
            contract.canonical_json_bytes(manifest)
        )
    authority = preflight.build_runtime_binding_authority(
        repository_runtime_manifest_sha256=repository[
            "repository_runtime_manifest_sha256"
        ],
        execution_dependency_manifest_sha256=dependency[
            "execution_dependency_manifest_sha256"
        ],
        checkpoint_loaded_code_manifest_sha256s={
            name: manifest["loaded_code_manifest_sha256"]
            for name, manifest in loaded.items()
        },
        process_environment_sha256="c" * 64,
        final_sys_path_tokens=["d" * 64],
    )
    (runtime_root / preflight.PRIVATE_RUNTIME_REPOSITORY_FILENAME).write_bytes(
        contract.canonical_json_bytes(repository)
    )
    (runtime_root / preflight.PRIVATE_RUNTIME_DEPENDENCY_FILENAME).write_bytes(
        contract.canonical_json_bytes(dependency)
    )
    (runtime_root / preflight.PRIVATE_RUNTIME_AUTHORITY_FILENAME).write_bytes(
        contract.canonical_json_bytes(authority)
    )
    return authority


def _creation_pretrust_attestation(
    root: Path,
    repository: dict[str, object],
) -> dict[str, object]:
    git_executable, git_payload, git_version = (
        preflight._resolve_authenticated_git_runtime()
    )
    python_executable = Path(sys.executable).resolve(strict=True)
    python_payload = python_executable.read_bytes()
    source_rows = [
        {
            "path": row["path"],
            "git_blob_sha1": row["git_blob_sha1"],
            "git_literal_sha256": row["literal_sha256"],
            "git_byte_count": row["byte_count"],
            "working_literal_sha256": row["literal_sha256"],
            "working_byte_count": row["byte_count"],
        }
        for row in repository["source_inventory"]
    ]
    ancestry = repository["ancestry"]
    body = {
        "schema_version": (
            preflight.CREATION_PRETRUST_ATTESTATION_SCHEMA_VERSION
        ),
        "base_commit": contract.BASE_COMMIT,
        "base_tree": contract.BASE_TREE,
        "preregistration_commit": contract.PREREGISTRATION_COMMIT,
        "preregistration_tree": contract.PREREGISTRATION_TREE,
        "preregistration_git_blob_sha1": (
            contract.PREREGISTRATION_GIT_BLOB_SHA1
        ),
        "preregistration_literal_sha256": (
            contract.PREREGISTRATION_LITERAL_SHA256
        ),
        "preregistration_literal_bytes": contract.PREREGISTRATION_LITERAL_BYTES,
        "implementation_commit": ancestry["commit"],
        "implementation_tree": ancestry["tree"],
        "implementation_parent": contract.PREREGISTRATION_COMMIT,
        "branch": contract.BRANCH_NAME,
        "upstream": f"origin/{contract.BRANCH_NAME}",
        "origin_url": preflight.CREATION_PRETRUST_EXPECTED_ORIGIN_URL,
        "repository_root_sha256": preflight.private_windows_path_sha256(root),
        "source_inventory": source_rows,
        "source_inventory_sha256": contract.canonical_sha256(source_rows),
        "clean_worktree": True,
        "git_runtime": {
            "basename": git_executable.name,
            "resolved_path_sha256": preflight.private_windows_path_sha256(
                git_executable
            ),
            "byte_count": len(git_payload),
            "literal_sha256": hashlib.sha256(git_payload).hexdigest(),
            "version_output_sha256": hashlib.sha256(git_version).hexdigest(),
        },
        "python_runtime": {
            "basename": python_executable.name,
            "resolved_path_sha256": preflight.private_windows_path_sha256(
                python_executable
            ),
            "byte_count": len(python_payload),
            "literal_sha256": hashlib.sha256(python_payload).hexdigest(),
            "python_version": sys.version,
            "cache_tag": sys.implementation.cache_tag,
        },
        "bootstrap": {
            "byte_count": len(preflight.SCIENTIFIC_BOOTSTRAP_BYTES),
            "literal_sha256": preflight.SCIENTIFIC_BOOTSTRAP_SHA256,
        },
        "effect_counts": contract.build_effect_budgets()[
            "zero_effect_preflight"
        ],
    }
    return preflight._self_hash(body, "attestation_sha256")


def _install_creation_pretrust(
    root: Path,
    repository: dict[str, object] | None = None,
) -> None:
    snapshot = repository if repository is not None else _repository_snapshot()
    setattr(
        builtins,
        preflight.CREATION_PRETRUST_SENTINEL_NAME,
        _creation_pretrust_attestation(root.resolve(strict=True), snapshot),
    )


def _qualification_xml(node_ids: list[str]) -> bytes:
    root = ET.Element("testsuites", {"name": "pytest tests"})
    suite = ET.SubElement(
        root,
        "testsuite",
        {
            "name": "pytest",
            "errors": "0",
            "failures": "0",
            "skipped": "0",
            "tests": str(len(node_ids)),
        },
    )
    for node_id in node_ids:
        parts = node_id.split("::")
        module_name = parts[0][:-3].replace("/", ".")
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": ".".join((module_name, *parts[1:-1])),
                "name": parts[-1],
            },
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _write_qualification_private_state(
    root: Path,
    repository: dict[str, object],
    *,
    v313_node_path: str = (
        "tests/test_sec_gemma_lean_science_v313_contract.py"
    ),
    duration_ns_override: int | None = None,
    log_injection: bytes = b"",
) -> dict[str, object]:
    root = root.resolve()
    private_root = root / contract.PRIVATE_PREFLIGHT_NAMESPACE
    qualification_root = private_root / "qualification"
    qualification_root.mkdir()
    runtime = _qualification_runtime_fixture()
    deadline_ns = 9_000_000_000_000_000
    clean_hash = hashlib.sha256(b"").hexdigest()
    ancestry = repository["ancestry"]
    phase_nodes = {
        contract.QUALIFICATION_PHASE_V313: [
            *preflight._RUNTIME_REQUIRED_PHASE1_NODE_IDS,
            *(
                f"{v313_node_path}::test_synthetic_{index:03d}"
                for index in range(
                    contract.QUALIFICATION_LATEST_MIN_NODE_COUNT
                    - len(preflight._RUNTIME_REQUIRED_PHASE1_NODE_IDS)
                )
            ),
        ],
        contract.QUALIFICATION_PHASE_DEPENDENCIES: _shared_node_ids(),
        contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY: [
            contract.QUALIFICATION_SENTINEL_NODE_ID
        ],
    }
    private_phases: list[dict[str, object]] = []
    public_phases: list[dict[str, object]] = []
    for phase_index, phase in enumerate(contract.QUALIFICATION_PHASES, start=1):
        phase_root = qualification_root / phase
        phase_root.mkdir()
        node_ids = phase_nodes[phase]
        node_hash = hashlib.sha256(
            "\n".join(node_ids).encode("utf-8")
        ).hexdigest()

        def write_mode(
            mode: str,
            collection_manifest_sha256: str | None,
        ) -> tuple[dict[str, object], dict[str, object]]:
            mode_root = phase_root / mode
            mode_root.mkdir()
            junit_partial = private_root / Path(
                contract.QUALIFICATION_JUNIT_RELATIVE_TEMPLATE.format(
                    phase=phase,
                    mode=mode,
                )
            )
            argv = preflight._qualification_argv(
                root,
                runtime=runtime,
                junit_partial=junit_partial,
                phase=phase,
                mode=mode,
            )
            _environment, environment_profile = (
                preflight._qualification_environment(runtime)
            )
            intent = preflight._self_hash(
                {
                    "schema_version": contract.QUALIFICATION_INTENT_SCHEMA_VERSION,
                    "suite_identity": contract.QUALIFICATION_SUITE_ID,
                    "phase": phase,
                    "mode": mode,
                    "argv": argv,
                    "environment_profile": environment_profile,
                    "repository_commit": ancestry["commit"],
                    "repository_tree": ancestry["tree"],
                    "clean_state_sha256": clean_hash,
                    "deadline_monotonic_ns": deadline_ns,
                    "runtime": copy.deepcopy(runtime),
                    "collection_manifest_sha256": (
                        collection_manifest_sha256
                    ),
                },
                "intent_sha256",
            )
            log_payload = (
                f"synthetic {phase} {mode}\n".encode("utf-8")
                + log_injection
            )
            stdout_payload = (
                (
                    "\n".join(node_ids)
                    + f"\n\n{len(node_ids)} tests collected in 0.01s\n"
                ).encode("utf-8")
                if mode == "collection"
                else f"{len(node_ids)} passed in 0.01s\n".encode("utf-8")
            )
            xml_payload = _qualification_xml(
                [] if mode == "collection" else node_ids
            )
            counts = {
                field: 0 for field in preflight._QUALIFICATION_COUNT_FIELDS
            }
            if mode == "execution":
                counts["passed_count"] = len(node_ids)
            result = preflight._self_hash(
                {
                    "schema_version": contract.QUALIFICATION_RESULT_SCHEMA_VERSION,
                    "suite_identity": contract.QUALIFICATION_SUITE_ID,
                    "phase": phase,
                    "mode": mode,
                    "status": "passed",
                    "intent_sha256": intent["intent_sha256"],
                    "collection_manifest_sha256": (
                        collection_manifest_sha256
                    ),
                    "started_unix_ns": phase_index,
                    "ended_unix_ns": phase_index + 1,
                    "duration_monotonic_ns": (
                        duration_ns_override
                        if duration_ns_override is not None
                        else (
                            phase_index
                            if mode == "collection"
                            else phase_index + 10
                        )
                    ),
                    "node_count": len(node_ids),
                    "node_list_sha256": node_hash,
                    "node_ids": node_ids,
                    **counts,
                    "exit_code": 0,
                    "timed_out": False,
                    "deadline_overrun": False,
                    "terminal_reason": "completed",
                    "exception_code": None,
                    "log_bytes": len(log_payload),
                    "log_sha256": hashlib.sha256(log_payload).hexdigest(),
                    "stdout_bytes": len(stdout_payload),
                    "stdout_sha256": hashlib.sha256(
                        stdout_payload
                    ).hexdigest(),
                    "xml_present": True,
                    "xml_bytes": len(xml_payload),
                    "xml_sha256": hashlib.sha256(xml_payload).hexdigest(),
                },
                "result_sha256",
            )
            intent_payload = contract.canonical_json_bytes(intent)
            result_payload = contract.canonical_json_bytes(result)
            completion = preflight._self_hash(
                {
                    "schema_version": (
                        preflight.QUALIFICATION_COMPLETION_SCHEMA_VERSION
                    ),
                    "suite_identity": contract.QUALIFICATION_SUITE_ID,
                    "phase": phase,
                    "mode": mode,
                    "status": "completed",
                    "terminal_reason": "completed",
                    "deadline_overrun": False,
                    "result_sha256": result["result_sha256"],
                    "result_literal_sha256": hashlib.sha256(
                        result_payload
                    ).hexdigest(),
                    "log_sha256": result["log_sha256"],
                    "stdout_sha256": result["stdout_sha256"],
                    "xml_sha256": result["xml_sha256"],
                },
                "completion_sha256",
            )
            (mode_root / "intent.json").write_bytes(intent_payload)
            (mode_root / "output.log").write_bytes(log_payload)
            (mode_root / "stdout.bin").write_bytes(stdout_payload)
            (mode_root / "junit.xml").write_bytes(xml_payload)
            (mode_root / "result.json").write_bytes(result_payload)
            (mode_root / "complete.json").write_bytes(
                contract.canonical_json_bytes(completion)
            )
            return result, completion

        collection, collection_completion = write_mode("collection", None)
        collection_manifest = preflight._self_hash(
            {
                "schema_version": (
                    contract.QUALIFICATION_COLLECTION_SCHEMA_VERSION
                ),
                "suite_identity": contract.QUALIFICATION_SUITE_ID,
                "phase": phase,
                "status": "passed",
                "ordered_node_ids": node_ids,
                "node_count": len(node_ids),
                "node_list_sha256": node_hash,
                "collection_result_sha256": collection["result_sha256"],
                "collection_completion_sha256": collection_completion[
                    "completion_sha256"
                ],
            },
            "collection_manifest_sha256",
        )
        (phase_root / "collection-manifest.json").write_bytes(
            contract.canonical_json_bytes(collection_manifest)
        )
        execution, execution_completion = write_mode(
            "execution", collection_manifest["collection_manifest_sha256"]
        )
        private_phases.append(
            {
                "phase": phase,
                "node_count": len(node_ids),
                "node_list_sha256": node_hash,
                "collection_manifest_sha256": collection_manifest[
                    "collection_manifest_sha256"
                ],
                "collection_completion_sha256": collection_completion[
                    "completion_sha256"
                ],
                "execution_completion_sha256": execution_completion[
                    "completion_sha256"
                ],
                "collection": collection,
                "execution": execution,
            }
        )
        public_phases.append(
            {
                "phase": phase,
                "node_count": len(node_ids),
                "node_list_sha256": node_hash,
                "collection_duration_ns": collection[
                    "duration_monotonic_ns"
                ],
                "execution_duration_ns": execution["duration_monotonic_ns"],
                "collection_exit_code": 0,
                "execution_exit_code": 0,
                **{
                    field: execution[field]
                    for field in preflight._QUALIFICATION_COUNT_FIELDS
                },
                "collection_result_sha256": collection["result_sha256"],
                "execution_result_sha256": execution["result_sha256"],
                "collection_log_sha256": collection["log_sha256"],
                "execution_log_sha256": execution["log_sha256"],
                "collection_xml_sha256": collection["xml_sha256"],
                "execution_xml_sha256": execution["xml_sha256"],
                "collection_manifest_sha256": collection_manifest[
                    "collection_manifest_sha256"
                ],
            }
        )
    aggregate = preflight._self_hash(
        {
            "schema_version": (
                contract.QUALIFICATION_PRIVATE_AGGREGATE_SCHEMA_VERSION
            ),
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "status": "passed",
            "deadline_monotonic_ns": deadline_ns,
            "deadline_seconds": contract.QUALIFICATION_TIMEOUT_SECONDS,
            "durability_mode": contract.QUALIFICATION_DURABILITY_MODE,
            "repository_commit": ancestry["commit"],
            "repository_tree": ancestry["tree"],
            "clean_state_sha256": clean_hash,
            "runtime": runtime,
            "phases": private_phases,
        },
        "private_aggregate_sha256",
    )
    aggregate_payload = contract.canonical_json_bytes(aggregate)
    (qualification_root / "private-aggregate.json").write_bytes(
        aggregate_payload
    )
    aggregate_completion = preflight._self_hash(
        {
            "schema_version": preflight.QUALIFICATION_COMPLETION_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "phase": "aggregate",
            "mode": "aggregate",
            "status": "completed",
            "terminal_reason": "completed",
            "deadline_overrun": False,
            "result_sha256": aggregate["private_aggregate_sha256"],
            "result_literal_sha256": hashlib.sha256(
                aggregate_payload
            ).hexdigest(),
            "log_sha256": None,
            "stdout_sha256": None,
            "xml_sha256": None,
        },
        "completion_sha256",
    )
    (qualification_root / "private-aggregate.complete.json").write_bytes(
        contract.canonical_json_bytes(aggregate_completion)
    )
    return preflight._self_hash(
        {
            "schema_version": preflight.QUALIFICATION_REPORT_SCHEMA_VERSION,
            "suite_identity": contract.QUALIFICATION_SUITE_ID,
            "status": "passed",
            "phase_count": len(public_phases),
            "deadline_seconds": contract.QUALIFICATION_TIMEOUT_SECONDS,
            "durability_mode": contract.QUALIFICATION_DURABILITY_MODE,
            "command_profile_sha256": (
                preflight.QUALIFICATION_COMMAND_PROFILE_SHA256
            ),
            "phases": public_phases,
            "private_aggregate_sha256": aggregate[
                "private_aggregate_sha256"
            ],
        },
        "qualification_sha256",
    )


def test_qualification_profile_freezes_three_phases_and_one_deadline() -> None:
    assert preflight.QUALIFICATION_TIMEOUT_SECONDS == 25 * 60
    assert preflight.QUALIFICATION_COMMAND_PROFILE["timeout_seconds"] == 25 * 60
    assert preflight.QUALIFICATION_COMMAND_PROFILE["phases"] == list(
        contract.QUALIFICATION_PHASES
    )


def _shared_case_sensitive_node_ids() -> list[str]:
    first = (
        "tests/test_sec_gemma_lean_v38_transport.py::"
        "test_noncanonical_urls_are_rejected[HTTPS://data.sec.gov/submissions/"
        "CIK0000320193.json]"
    )
    second = (
        "tests/test_sec_gemma_lean_v38_transport.py::"
        "test_noncanonical_urls_are_rejected[https://DATA.sec.gov/submissions/"
        "CIK0000320193.json]"
    )
    unique = [
        f"tests/test_shared_{index:03}.py::test_case_{index:03}"
        for index in range(contract.QUALIFICATION_SHARED_NODE_COUNT - 2)
    ]
    return [*unique, first, second]


def test_phase2_case_sensitive_multiplicity_is_655_unique_zero_duplicates() -> None:
    nodes = _shared_case_sensitive_node_ids()
    digest = hashlib.sha256("\n".join(nodes).encode("utf-8")).hexdigest()

    validated = preflight._qualification_validate_node_ids(
        nodes,
        phase=contract.QUALIFICATION_PHASE_DEPENDENCIES,
        count=contract.QUALIFICATION_SHARED_NODE_COUNT,
        digest=digest,
        code="multiplicity_invalid",
    )
    multiplicities = Counter(validated)
    assert len(validated) == 655
    assert len(multiplicities) == 655
    assert [value for value in multiplicities.values() if value != 1] == []


def test_phase2_rejects_case_insensitive_node_id_collapsing() -> None:
    nodes = _shared_case_sensitive_node_ids()
    # Reproduce the V3.11 mistake: one case-distinct URL is collapsed onto the
    # other spelling.  The list still has 655 entries, but exact Counter
    # semantics now see 654 unique IDs and one duplicate, so V3.13 rejects it.
    nodes[-1] = nodes[-2]
    digest = hashlib.sha256("\n".join(nodes).encode("utf-8")).hexdigest()

    _assert_code(
        "multiplicity_invalid",
        preflight._qualification_validate_node_ids,
        nodes,
        phase=contract.QUALIFICATION_PHASE_DEPENDENCIES,
        count=len(nodes),
        digest=digest,
        code="multiplicity_invalid",
    )


def test_isolated_qualifier_bootstrap_matches_frozen_literal_and_imports_pinned_pytest() -> None:
    root = Path(__file__).resolve().parents[1]
    bootstrap = contract.QUALIFICATION_BOOTSTRAP_LITERAL.encode("utf-8")
    assert len(bootstrap) == contract.QUALIFICATION_BOOTSTRAP_BYTES == 3_999
    assert hashlib.sha256(bootstrap).hexdigest() == (
        contract.QUALIFICATION_BOOTSTRAP_SHA256
    )
    assert "pytest" not in preflight.__dict__

    runtime = _qualification_runtime_fixture()
    environment, profile = preflight._qualification_environment(runtime)
    assert environment != profile
    assert tuple(sorted(environment)) == tuple(sorted(profile))
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        assert profile[name] == preflight.private_windows_path_sha256(
            environment[name]
        )
        assert environment[name] not in contract.canonical_json_bytes(profile).decode(
            "ascii"
        )
    assert profile["PATH"] == [
        preflight.private_windows_path_sha256(path)
        for path in environment["PATH"].split(";")
    ]
    assert set(environment) == set(contract.QUALIFICATION_ENVIRONMENT_NAMES)
    dispatch = [
        str(root)
        if token == contract.QUALIFICATION_CANONICAL_ROOT_TOKEN
        else token
        for token in contract.QUALIFICATION_DISPATCH_PREFIX
    ]
    argv = [
        runtime["executable_path"],
        *contract.QUALIFICATION_PYTHON_FLAGS,
        "-c",
        contract.QUALIFICATION_BOOTSTRAP_LITERAL,
        *dispatch,
        *contract.QUALIFICATION_PYTEST_ARGS,
        "--collect-only",
        contract.QUALIFICATION_SENTINEL_NODE_ID,
    ]
    completed = subprocess.run(
        argv,
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert preflight._qualification_collection_node_ids(completed.stdout) == [
        contract.QUALIFICATION_SENTINEL_NODE_ID
    ]


def test_authority_creation_precedes_authority_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    _install_creation_pretrust(root)
    events: list[str] = []
    real_consume = preflight._consume_creation_pretrust_attestation
    real_reserve = preflight._reserve_once

    def consume(path: Path) -> dict[str, object]:
        events.append("outer_creation_pretrust")
        return real_consume(path)

    def reserve(path: Path, **kwargs: object) -> Path:
        events.append("reservation_consumption_boundary")
        return real_reserve(path, **kwargs)

    monkeypatch.setattr(
        preflight, "_consume_creation_pretrust_attestation", consume
    )
    monkeypatch.setattr(preflight, "_reserve_once", reserve)
    dependencies = _dependencies(root)
    dependencies = preflight.PreflightDependencies(
        **{
            **dependencies.__dict__,
            "inspect_repository": lambda _root: (
                events.append("inner_repository_binding")
                or _repository_snapshot()
            ),
            "run_qualification": lambda _root: (_ for _ in ()).throw(
                preflight.V313PreflightError("stop_after_reservation")
            ),
        }
    )
    _assert_code(
        "stop_after_reservation",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    assert events == [
        "outer_creation_pretrust",
        "inner_repository_binding",
        "reservation_consumption_boundary",
    ]
    _assert_failed_preflight_artifact(root, "stop_after_reservation")


def test_missing_or_tampered_creation_pretrust_writes_nothing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    _make_repo(missing)
    _assert_code(
        "creation_pretrust_attestation_missing",
        preflight.run_preflight,
        missing,
        dependencies=_dependencies(missing),
    )
    assert not (missing / contract.PRIVATE_PREFLIGHT_NAMESPACE).exists()
    assert not (missing / contract.PREFLIGHT_ARTIFACT_PATH).exists()

    tampered = tmp_path / "tampered"
    _make_repo(tampered)
    attestation = _creation_pretrust_attestation(
        tampered.resolve(), _repository_snapshot()
    )
    attestation["implementation_tree"] = "f" * 40
    setattr(
        builtins,
        preflight.CREATION_PRETRUST_SENTINEL_NAME,
        attestation,
    )
    _assert_code(
        "creation_pretrust_attestation_invalid",
        preflight.run_preflight,
        tampered,
        dependencies=_dependencies(tampered),
    )
    assert not (tampered / contract.PRIVATE_PREFLIGHT_NAMESPACE).exists()
    assert not (tampered / contract.PREFLIGHT_ARTIFACT_PATH).exists()


def test_authority_creation_never_requires_f_or_pushed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    _install_creation_pretrust(root)
    observed: list[tuple[Path, dict[str, object]]] = []

    def create_runtime(
        received_root: Path,
        repository: dict[str, object],
    ) -> dict[str, object]:
        assert not (received_root / contract.PREFLIGHT_ARTIFACT_PATH).exists()
        ancestry = repository["ancestry"]
        assert ancestry["parent"] == contract.PREREGISTRATION_COMMIT
        assert ancestry["commit"] != contract.PREREGISTRATION_COMMIT
        assert "preflight_commit" not in repository
        assert "pushed_preflight_gate_passed" not in repository
        observed.append((received_root, repository))
        return _runtime_authority_fixture()

    monkeypatch.setattr(
        preflight,
        "_validate_qualification_private_state",
        lambda *_args, **_kwargs: {},
    )
    dependencies = _dependencies(root)
    dependencies = preflight.PreflightDependencies(
        **{
            **dependencies.__dict__,
            "run_qualification": lambda _root: _qualification(),
            "create_runtime_authority": create_runtime,
            "authenticate_source": lambda _root: (_ for _ in ()).throw(
                preflight.V313PreflightError("stop_after_authority_creation")
            ),
        }
    )
    _assert_code(
        "stop_after_authority_creation",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    assert len(observed) == 1
    assert observed[0][0] == root.resolve()


def test_runtime_sys_path_drops_only_the_absent_frozen_zip(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "Python312"
    runtime.mkdir()
    executable = runtime / "python.exe"
    executable.write_bytes(b"python")
    dlls = runtime / "DLLs"
    library = runtime / "Lib"
    repository = tmp_path / "repo"
    purelib = tmp_path / "site-packages"
    for path in (dlls, library, repository, purelib):
        path.mkdir()
    initial = [
        "",
        str(runtime / "python312.zip"),
        str(dlls),
        str(library),
        str(runtime),
    ]

    derived = preflight.derive_final_sys_path(
        initial,
        python_executable=executable,
        repository_root=repository,
        purelib_root=purelib,
        platlib_root=purelib,
    )

    assert len(derived["final_sys_path_tokens"]) == 5
    assert derived["final_sys_path_sha256"] == contract.canonical_sha256(
        derived["final_sys_path_tokens"]
    )
    (runtime / "python312.zip").write_bytes(b"unexpected")
    _assert_code(
        "runtime_initial_sys_path_invalid",
        preflight.derive_final_sys_path,
        initial,
        python_executable=executable,
        repository_root=repository,
        purelib_root=purelib,
        platlib_root=purelib,
    )


def test_runtime_authority_and_receipt_bind_every_checkpoint() -> None:
    checkpoint_hashes = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in preflight.RUNTIME_CHECKPOINT_NAMES
    }
    tokens = [_digest for _digest in ("1" * 64, "2" * 64)]
    authority = preflight.build_runtime_binding_authority(
        repository_runtime_manifest_sha256="3" * 64,
        execution_dependency_manifest_sha256="4" * 64,
        checkpoint_loaded_code_manifest_sha256s=checkpoint_hashes,
        process_environment_sha256="5" * 64,
        final_sys_path_tokens=tokens,
    )
    assert preflight.validate_runtime_binding_authority(authority) == authority
    receipt = preflight.build_runtime_binding_receipt(
        checkpoint_name="evaluation_pre",
        invocation_kind="development",
        head="a" * 40,
        tree="b" * 40,
        clean_state_sha256="6" * 64,
        authority=authority,
        interpreter_identity_sha256="7" * 64,
        previous_runtime_binding_receipt_sha256=None,
    )
    assert receipt["loaded_code_manifest_sha256"] == checkpoint_hashes[
        "evaluation_pre"
    ]
    assert receipt["route_argv_sha256"] == contract.canonical_sha256(
        ["development"]
    )
    tampered = copy.deepcopy(authority)
    tampered["final_sys_path_tokens"].reverse()
    _assert_code(
        "runtime_binding_authority_invalid",
        preflight.validate_runtime_binding_authority,
        tampered,
    )


def test_junit_xpass_uses_one_anchored_terminal_summary() -> None:
    node_ids = [
        "tests/test_example.py::test_passed",
        "tests/test_example.py::test_xpassed",
    ]
    xml_payload = _qualification_xml(node_ids)
    stdout = (
        b"earlier user output says 0 xpassed\n"
        b"1 passed, 1 xpassed in 0.01s\n"
    )
    assert preflight._qualification_junit_counts(xml_payload, stdout) == {
        "passed_count": 1,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 1,
    }
    _assert_code(
        "preflight_qualification_junit_invalid",
        preflight._qualification_junit_counts,
        xml_payload,
        (
            b"0 passed, 2 xpassed in 0.01s\n"
            b"1 passed, 1 xpassed in 0.02s\n"
        ),
    )
    assert preflight.QUALIFICATION_COMMAND_PROFILE["modes"] == [
        "collection",
        "execution",
    ]
    assert preflight.QUALIFICATION_COMMAND_PROFILE["stdout"] == (
        "concurrent_binary_tee"
    )
    assert preflight.validate_qualification_report(_qualification()) == (
        _qualification()
    )
    assert len(contract.QUALIFICATION_SHARED_TEST_PATHS) == 14
    assert contract.QUALIFICATION_SHARED_NODE_COUNT == 655
    assert contract.QUALIFICATION_SHARED_NODE_LIST_SHA256 == (
        "1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c"
    )
    assert preflight._qualification_collection_evidence(
        contract.QUALIFICATION_SENTINEL_NODE_ID.encode("utf-8")
    ) == (
        contract.QUALIFICATION_SENTINEL_NODE_COUNT,
        contract.QUALIFICATION_SENTINEL_NODE_LIST_SHA256,
    )


def test_local_production_closure_reproduces_exact_frozen_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = preflight._derive_local_production_closure(
        root,
        head_commit=contract.BASE_COMMIT,
    )
    encoded = contract.canonical_json_bytes(manifest)
    assert len(manifest["paths"]) == 37
    assert len(encoded) == contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_BYTES
    assert hashlib.sha256(encoded).hexdigest() == (
        contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256
    )


def test_deadline_overrun_completion_fails_without_mutating_self_hashed_result() -> None:
    result = preflight._self_hash(
        {
            "status": "passed",
            "terminal_reason": "completed",
            "deadline_overrun": False,
            "log_sha256": "1" * 64,
            "stdout_sha256": "4" * 64,
            "xml_sha256": "2" * 64,
        },
        "result_sha256",
    )
    frozen_result = copy.deepcopy(result)
    completed = preflight._qualification_completion_receipt(
        phase=contract.QUALIFICATION_PHASE_V313,
        mode="execution",
        result=result,
        result_literal_sha256="3" * 64,
        finalization_overrun=False,
    )
    assert completed["status"] == "completed"
    assert completed["terminal_reason"] == "completed"
    completion = preflight._qualification_completion_receipt(
        phase=contract.QUALIFICATION_PHASE_V313,
        mode="execution",
        result=result,
        result_literal_sha256="3" * 64,
        finalization_overrun=True,
    )
    assert result == frozen_result
    assert result["result_sha256"] == contract.canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    assert completion["status"] == "failed"
    assert completion["terminal_reason"] == "timeout"
    assert completion["deadline_overrun"] is True
    assert completion["completion_sha256"] == contract.canonical_sha256(
        {
            key: value
            for key, value in completion.items()
            if key != "completion_sha256"
        }
    )


def _dependencies(
    root: Path,
    *,
    effects: dict[str, int] | None = None,
    commitments: dict[str, object] | None = None,
    tokens: tuple[bytes | str, ...] | None = None,
    calls: dict[str, int] | None = None,
) -> preflight.PreflightDependencies:
    counter = calls if calls is not None else {}
    projection = SimpleNamespace(manifest=_bridge_manifest())

    def authenticate(_root: Path) -> object:
        counter["authenticate"] = counter.get("authenticate", 0) + 1
        return object()

    def build_projection(_authority: object) -> object:
        counter["projection"] = counter.get("projection", 0) + 1
        return projection

    zero = (
        effects
        if effects is not None
        else contract.build_effect_budgets()["zero_effect_preflight"]
    )
    return preflight.PreflightDependencies(
        inspect_repository=lambda _root: _repository_snapshot(),
        create_runtime_authority=lambda run_root, _repository: (
            _write_runtime_authority_bundle(run_root)
        ),
        authenticate_source=authenticate,
        build_projection=build_projection,
        build_request_commitments=lambda _projection: (
            commitments if commitments is not None else _commitments()
        ),
        effect_snapshot=lambda: copy.deepcopy(zero),
        run_qualification=lambda run_root: _write_qualification_private_state(
            run_root, _repository_snapshot()
        ),
        privacy_tokens=lambda _root: (
            tokens
            if tokens is not None
            else (
                b"SYNTHETIC-PRIVATE-CONTACT",
                str(root),
                str(root / preflight.V38_PRIVATE_ROOT),
                str(root / contract.PRIVATE_NAMESPACE),
            )
        ),
    )


def _make_repo(root: Path) -> None:
    root.mkdir()
    (root / ".git").mkdir()


def test_private_qualification_replay_accepts_production_shaped_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    preflight._reserve_once(root.resolve())
    repository = _repository_snapshot()
    report = _write_qualification_private_state(root, repository)
    aggregate = preflight._validate_qualification_private_state(
        root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "qualification",
        report,
        repository=repository,
    )
    assert aggregate["private_aggregate_sha256"] == report[
        "private_aggregate_sha256"
    ]
    assert all(
        _is_hash(phase["collection_completion_sha256"])
        and _is_hash(phase["execution_completion_sha256"])
        for phase in aggregate["phases"]
    )


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@pytest.mark.parametrize(
    "mutation", ("tamper", "stdout_tamper", "missing", "extra")
)
def test_private_qualification_replay_rejects_tree_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / mutation
    _make_repo(root)
    preflight._reserve_once(root.resolve())
    repository = _repository_snapshot()
    report = _write_qualification_private_state(root, repository)
    qualification_root = (
        root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "qualification"
    )
    mode_root = (
        qualification_root
        / contract.QUALIFICATION_PHASE_V313
        / "execution"
    )
    if mutation == "tamper":
        (mode_root / "output.log").write_bytes(b"tampered\n")
    elif mutation == "stdout_tamper":
        (mode_root / "stdout.bin").write_bytes(b"tampered\n")
    elif mutation == "missing":
        (mode_root / "complete.json").unlink()
    else:
        (mode_root / "deadline-failure.json").write_bytes(b"forbidden")
    _assert_code(
        "execution_preflight_qualification_invalid",
        preflight._validate_qualification_private_state,
        qualification_root,
        report,
        repository=repository,
    )


def test_private_qualification_replay_rejects_internally_consistent_wrong_selector(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wrong-selector"
    _make_repo(root)
    preflight._reserve_once(root.resolve())
    repository = _repository_snapshot()
    report = _write_qualification_private_state(
        root,
        repository,
        v313_node_path="tests/test_unrelated.py",
    )
    _assert_code(
        "execution_preflight_qualification_invalid",
        preflight._validate_qualification_private_state,
        root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "qualification",
        report,
        repository=repository,
    )


def test_private_qualification_replay_rejects_impossible_total_duration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "impossible-duration"
    _make_repo(root)
    preflight._reserve_once(root.resolve())
    repository = _repository_snapshot()
    report = _write_qualification_private_state(
        root,
        repository,
        duration_ns_override=250_000_000_000,
    )
    _assert_code(
        "execution_preflight_qualification_invalid",
        preflight._validate_qualification_private_state,
        root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "qualification",
        report,
        repository=repository,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree seal")
def test_timeout_termination_kills_spawned_child_tree(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(child_pid_path)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    child_pid: int | None = None
    try:
        for _ in range(100):
            if child_pid_path.exists():
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                break
            time.sleep(0.02)
        assert child_pid is not None
        assert preflight._qualification_terminate_process_tree(process) is True
        listing = subprocess.run(
            [
                "tasklist.exe",
                "/FI",
                f"PID eq {child_pid}",
                "/FO",
                "CSV",
                "/NH",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert f'"{child_pid}"' not in listing.stdout
    finally:
        if process.poll() is None:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        if child_pid is not None:
            subprocess.run(
                ["taskkill.exe", "/PID", str(child_pid), "/T", "/F"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object seal")
def test_job_object_close_kills_child_after_parent_exits(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "orphan.pid"
    release_path = tmp_path / "release"
    script = (
        "import pathlib,subprocess,sys,time;"
        "release=pathlib.Path(sys.argv[1]);"
        "pid_path=pathlib.Path(sys.argv[2]);"
        "\nwhile not release.exists():time.sleep(0.01)\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(60)']);"
        "pid_path.write_text(str(child.pid))"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(release_path),
            str(child_pid_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004
        ),
    )
    job = preflight._qualification_create_windows_job(process)
    child_pid: int | None = None
    try:
        assert job is not None
        assert preflight._qualification_resume_windows_process(process) is True
        release_path.write_bytes(b"go")
        assert process.wait(timeout=5.0) == 0
        for _ in range(100):
            if child_pid_path.exists():
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                break
            time.sleep(0.02)
        assert child_pid is not None
        proven, exit_code = preflight._qualification_terminate_job_and_reap(
            job, process
        )
        assert proven is True
        assert exit_code == 0
        stdout, stderr = process.communicate(timeout=5.0)
        assert stdout == b""
        assert stderr == b""
        listing = subprocess.run(
            [
                "tasklist.exe",
                "/FI",
                f"PID eq {child_pid}",
                "/FO",
                "CSV",
                "/NH",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert f'"{child_pid}"' not in listing.stdout
    finally:
        if process.poll() is None:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        if child_pid is not None:
            subprocess.run(
                ["taskkill.exe", "/PID", str(child_pid), "/T", "/F"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )


@pytest.mark.parametrize("valid_xml", (True, False))
def test_abnormal_exit_authenticates_junit_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_xml: bool,
) -> None:
    root = tmp_path / ("valid" if valid_xml else "malformed")
    _make_repo(root)
    _freeze_qualification_environment_for_process_double(monkeypatch)
    private_root = root / contract.PRIVATE_PREFLIGHT_NAMESPACE
    mode_root = (
        private_root
        / "qualification"
        / contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY
        / "execution"
    )
    mode_root.mkdir(parents=True)
    node_id = contract.QUALIFICATION_SENTINEL_NODE_ID
    node_hash = contract.QUALIFICATION_SENTINEL_NODE_LIST_SHA256
    if valid_xml:
        xml_root = ET.fromstring(_qualification_xml([node_id]))
        testcase = next(xml_root.iter("testcase"))
        ET.SubElement(testcase, "failure", {"message": "synthetic"})
        junit_payload = ET.tostring(
            xml_root, encoding="utf-8", xml_declaration=True
        )
    else:
        junit_payload = b"<testsuites><broken>"

    class FakeProcess:
        pid = 12345

        def __init__(self, argv, **kwargs) -> None:
            assert kwargs["creationflags"] & 0x00000004
            junit_argument = next(
                item for item in argv if item.startswith("--junitxml=")
            )
            Path(junit_argument.split("=", 1)[1]).write_bytes(junit_payload)
            self.stdout = io.BytesIO(b"1 failed in 0.01s\n")
            self.stderr = io.BytesIO(b"")

        def poll(self) -> int:
            return 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 1

    monkeypatch.setattr(preflight.subprocess, "Popen", FakeProcess)
    fake_job = SimpleNamespace(
        terminate=lambda: True,
        active_process_count=lambda: 0,
        close=lambda: True,
    )
    monkeypatch.setattr(
        preflight,
        "_qualification_create_windows_job",
        lambda _process: fake_job,
    )
    monkeypatch.setattr(
        preflight,
        "_qualification_resume_windows_process",
        lambda _process: True,
    )
    sealed = {
        "ordered_node_ids": [node_id],
        "node_count": 1,
        "node_list_sha256": node_hash,
        "collection_manifest_sha256": "a" * 64,
    }
    _assert_code(
        "preflight_qualification_failed",
        preflight._qualification_run_process,
        root.resolve(),
        private_root=private_root.resolve(),
        phase=contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY,
        mode="execution",
        deadline_monotonic=time.monotonic() + 30.0,
        runtime=_qualification_runtime_fixture(),
        repository_commit=SHA1_A,
        repository_tree=SHA1_B,
        clean_state_sha256=hashlib.sha256(b"").hexdigest(),
        sealed_collection=sealed,
    )
    result = json.loads((mode_root / "result.json").read_bytes())
    if valid_xml:
        assert result["xml_present"] is True
        assert result["xml_sha256"] == hashlib.sha256(junit_payload).hexdigest()
        assert result["node_count"] == 1
        assert result["failed_count"] == 1
    else:
        assert result["xml_present"] is False
        assert result["xml_bytes"] is None
        assert result["xml_sha256"] is None
        assert result["node_count"] is None
        assert all(
            result[field] is None
            for field in preflight._QUALIFICATION_COUNT_FIELDS
        )


def test_wait_failure_terminates_and_closes_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "wait-failure"
    _make_repo(root)
    _freeze_qualification_environment_for_process_double(monkeypatch)
    private_root = root / contract.PRIVATE_PREFLIGHT_NAMESPACE
    mode_root = (
        private_root
        / "qualification"
        / contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY
        / "collection"
    )
    mode_root.mkdir(parents=True)
    state = {"terminated": False, "closed": False, "waits": 0}

    class FakeProcess:
        pid = 54321

        def __init__(self, _argv, **_kwargs) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")

        def poll(self) -> int | None:
            return 1 if state["terminated"] else None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            state["waits"] += 1
            if not state["terminated"]:
                raise OSError("synthetic wait failure")
            return 1

    class FakeJob:
        def terminate(self) -> bool:
            state["terminated"] = True
            return True

        def close(self) -> bool:
            state["closed"] = True
            return True

        def active_process_count(self) -> int:
            return 0

    monkeypatch.setattr(preflight.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        preflight,
        "_qualification_create_windows_job",
        lambda _process: FakeJob(),
    )
    monkeypatch.setattr(
        preflight,
        "_qualification_resume_windows_process",
        lambda _process: True,
    )
    _assert_code(
        "preflight_qualification_failed",
        preflight._qualification_run_process,
        root.resolve(),
        private_root=private_root.resolve(),
        phase=contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY,
        mode="collection",
        deadline_monotonic=time.monotonic() + 30.0,
        runtime=_qualification_runtime_fixture(),
        repository_commit=SHA1_A,
        repository_tree=SHA1_B,
        clean_state_sha256=hashlib.sha256(b"").hexdigest(),
        sealed_collection=None,
    )
    assert state == {"terminated": True, "closed": True, "waits": 2}
    result = json.loads((mode_root / "result.json").read_bytes())
    assert result["exception_code"] == "wait_failed"
    assert result["status"] == "failed"


def test_containment_setup_failure_kills_suspended_child_before_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "containment-setup"
    _make_repo(root)
    _freeze_qualification_environment_for_process_double(monkeypatch)
    private_root = root / contract.PRIVATE_PREFLIGHT_NAMESPACE
    mode_root = (
        private_root
        / "qualification"
        / contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY
        / "collection"
    )
    mode_root.mkdir(parents=True)
    state = {"killed": False, "resumed": False}

    class SuspendedProcess:
        pid = 65432

        def __init__(self, _argv, **kwargs) -> None:
            assert kwargs["creationflags"] & 0x00000004
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")

        def kill(self) -> None:
            state["killed"] = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 1

        def poll(self) -> int | None:
            return 1 if state["killed"] else None

    monkeypatch.setattr(preflight.subprocess, "Popen", SuspendedProcess)
    monkeypatch.setattr(
        preflight, "_qualification_create_windows_job", lambda _process: None
    )
    monkeypatch.setattr(
        preflight,
        "_qualification_resume_windows_process",
        lambda _process: state.__setitem__("resumed", True) or True,
    )
    _assert_code(
        "preflight_qualification_failed",
        preflight._qualification_run_process,
        root.resolve(),
        private_root=private_root.resolve(),
        phase=contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY,
        mode="collection",
        deadline_monotonic=time.monotonic() + 30.0,
        runtime=_qualification_runtime_fixture(),
        repository_commit=SHA1_A,
        repository_tree=SHA1_B,
        clean_state_sha256=hashlib.sha256(b"").hexdigest(),
        sealed_collection=None,
    )
    assert state == {"killed": True, "resumed": False}
    result = json.loads((mode_root / "result.json").read_bytes())
    assert result["exception_code"] == "containment_failed"


def test_abnormal_collection_preserves_parsed_junit_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "collection-exit-one"
    _make_repo(root)
    _freeze_qualification_environment_for_process_double(monkeypatch)
    private_root = root / contract.PRIVATE_PREFLIGHT_NAMESPACE
    mode_root = (
        private_root
        / "qualification"
        / contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY
        / "collection"
    )
    mode_root.mkdir(parents=True)
    node_id = contract.QUALIFICATION_SENTINEL_NODE_ID
    xml_root = ET.fromstring(_qualification_xml([node_id]))
    ET.SubElement(
        next(xml_root.iter("testcase")),
        "error",
        {"message": "synthetic collection error"},
    )
    junit_payload = ET.tostring(
        xml_root, encoding="utf-8", xml_declaration=True
    )

    class CollectionProcess:
        pid = 76543

        def __init__(self, argv, **_kwargs) -> None:
            junit_argument = next(
                item for item in argv if item.startswith("--junitxml=")
            )
            Path(junit_argument.split("=", 1)[1]).write_bytes(junit_payload)
            self.stdout = io.BytesIO((node_id + "\n").encode("utf-8"))
            self.stderr = io.BytesIO(b"")

        def poll(self) -> int:
            return 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 1

    fake_job = SimpleNamespace(
        terminate=lambda: True,
        active_process_count=lambda: 0,
        close=lambda: True,
    )
    monkeypatch.setattr(preflight.subprocess, "Popen", CollectionProcess)
    monkeypatch.setattr(
        preflight,
        "_qualification_create_windows_job",
        lambda _process: fake_job,
    )
    monkeypatch.setattr(
        preflight,
        "_qualification_resume_windows_process",
        lambda _process: True,
    )
    _assert_code(
        "preflight_qualification_failed",
        preflight._qualification_run_process,
        root.resolve(),
        private_root=private_root.resolve(),
        phase=contract.QUALIFICATION_PHASE_REQUESTS_IDENTITY,
        mode="collection",
        deadline_monotonic=time.monotonic() + 30.0,
        runtime=_qualification_runtime_fixture(),
        repository_commit=SHA1_A,
        repository_tree=SHA1_B,
        clean_state_sha256=hashlib.sha256(b"").hexdigest(),
        sealed_collection=None,
    )
    result = json.loads((mode_root / "result.json").read_bytes())
    assert result["xml_present"] is True
    assert result["error_count"] == 1
    assert result["passed_count"] == 0


def test_v313_local_import_upper_bound_covers_static_and_fake_runtime_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    package = root / "agent_benchmark"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for index, relative in enumerate(contract.IMPLEMENTATION_PRODUCTION_PATHS):
        path = root / relative
        body = "from __future__ import annotations\n"
        if index == 0:
            body += (
                "import importlib\n"
                "from . import sec_gemma_lean_science_v313_bridge\n"
                "importlib.import_module('pytest')\n"
            )
        path.write_text(body, encoding="utf-8")
    _git(root, "add", "--", "agent_benchmark")
    _git(root, "commit", "-m", "synthetic local imports")
    head = _git(root, "rev-parse", "HEAD")
    runtime_paths = [
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
        *contract.IMPLEMENTATION_PRODUCTION_PATHS,
    ]
    observed = preflight._audit_v313_local_import_upper_bound(
        root,
        head_commit=head,
        runtime_observed_paths=runtime_paths,
    )
    assert set(runtime_paths).issubset(observed)

    _assert_code(
        "preflight_local_import_upper_bound_invalid",
        preflight._audit_v313_local_import_upper_bound,
        root,
        head_commit=head,
        runtime_observed_paths=[
            *runtime_paths,
            "agent_benchmark/not_preregistered.py",
        ],
    )
    dynamic_source = root / contract.IMPLEMENTATION_PRODUCTION_PATHS[0]
    dynamic_source.write_text(
        "import importlib\n"
        "module_name = 'agent_benchmark.sec_gemma_lean_science_v313_bridge'\n"
        "importlib.import_module(module_name)\n",
        encoding="utf-8",
    )
    _git(root, "add", "--", contract.IMPLEMENTATION_PRODUCTION_PATHS[0])
    _git(root, "commit", "-m", "unresolved dynamic import")
    dynamic_head = _git(root, "rev-parse", "HEAD")
    _assert_code(
        "preflight_local_import_dynamic_unresolved",
        preflight._audit_v313_local_import_upper_bound,
        root,
        head_commit=dynamic_head,
        runtime_observed_paths=runtime_paths,
    )


def test_validators_reject_scope_count_parity_and_effect_changes() -> None:
    assert preflight.validate_repository_snapshot(_repository_snapshot()) == (
        _repository_snapshot()
    )
    assert preflight.validate_bridge_manifest(_bridge_manifest()) == _bridge_manifest()
    assert preflight.validate_request_commitments(_commitments()) == _commitments()
    assert preflight.validate_qualification_report(_qualification()) == _qualification()

    changed = _bridge_manifest()
    changed["confirmation_or_final_opened"] = True
    body = dict(changed)
    body.pop("bridge_sha256")
    changed["bridge_sha256"] = contract.canonical_sha256(body)
    _assert_code(
        "preflight_bridge_manifest_invalid",
        preflight.validate_bridge_manifest,
        changed,
    )
    changed_commitments = _commitments()
    changed_commitments["request_count"] = 74
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.validate_request_commitments,
        changed_commitments,
    )
    changed_repo = _repository_snapshot()
    changed_repo["source_inventory"][0]["literal_sha256"] = "f" * 64
    _assert_code(
        "preflight_repository_identity_invalid",
        preflight.validate_repository_snapshot,
        changed_repo,
    )


def test_bridge_validator_accepts_real_production_projection_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = bridge_fixtures._make_fixture()
    projection = bridge.build_streaming_science_projection(fixture.authority)
    manifest = dict(projection.manifest)
    assert len(manifest) == 41
    assert set(manifest) == preflight._BRIDGE_MANIFEST_FIELDS
    frozen_pins = contract.build_source_authority_pins()
    monkeypatch.setattr(contract, "build_source_authority_pins", lambda: frozen_pins)
    for field, name in (
        ("stage_source_seal_sha256", "V38_STAGE_SOURCE_SEAL_SHA256"),
        ("checkpoint_sha256", "V38_LOGICAL_CHECKPOINT_SHA256"),
        ("compact_replay_sha256", "V38_COMPACT_REPLAY_SHA256"),
        ("role_manifests_sha256", "V38_ROLE_MANIFEST_INVENTORY_SHA256"),
        ("role_plan_sha256", "V38_ROLE_PLAN_SHA256"),
    ):
        monkeypatch.setattr(contract, name, manifest[field])
    assert preflight.validate_bridge_manifest(manifest) == manifest


def test_commitment_validator_accepts_real_runner_projection_output() -> None:
    fixture = bridge_fixtures._make_fixture()
    projection = bridge.build_streaming_science_projection(fixture.authority)
    commitments = runner.build_preflight_request_commitments(projection)
    assert len(commitments) == 34
    assert set(commitments) == preflight._REQUEST_COMMITMENT_FIELDS
    assert preflight.validate_request_commitments(commitments) == commitments
    preflight._validate_bridge_commitment_parity(
        dict(projection.manifest), commitments
    )
    assert commitments["model_plan_sha256"] == runner.build_model_plan(
        projection
    ).manifest["model_plan_sha256"]


def test_real_bridge_runner_and_preflight_authority_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = bridge_fixtures._make_fixture()
    projection = bridge.build_streaming_science_projection(fixture.authority)
    bridge_manifest = dict(projection.manifest)
    commitments = runner.build_preflight_request_commitments(projection)
    frozen_pins = contract.build_source_authority_pins()
    monkeypatch.setattr(contract, "build_source_authority_pins", lambda: frozen_pins)
    for field, name in (
        ("stage_source_seal_sha256", "V38_STAGE_SOURCE_SEAL_SHA256"),
        ("checkpoint_sha256", "V38_LOGICAL_CHECKPOINT_SHA256"),
        ("compact_replay_sha256", "V38_COMPACT_REPLAY_SHA256"),
        ("role_manifests_sha256", "V38_ROLE_MANIFEST_INVENTORY_SHA256"),
        ("role_plan_sha256", "V38_ROLE_PLAN_SHA256"),
    ):
        monkeypatch.setattr(contract, name, bridge_manifest[field])
    repository = _repository_snapshot()
    effects = contract.build_effect_budgets()["zero_effect_preflight"]
    runtime_authority = {"runtime_binding_authority_sha256": "d" * 64}
    private_manifest = preflight._build_private_manifest(
        repository=repository,
        bridge=bridge_manifest,
        commitments=commitments,
        effect_before=effects,
        effect_after=effects,
        tests=_qualification(),
        runtime_authority=runtime_authority,
    )
    private_literal = hashlib.sha256(
        preflight._artifact_bytes(private_manifest)
    ).hexdigest()
    public_artifact = preflight._build_public_artifact(
        repository=repository,
        bridge=bridge_manifest,
        commitments=commitments,
        effects=effects,
        tests=_qualification(),
        runtime_authority=runtime_authority,
        private_manifest_sha256=private_manifest["private_manifest_sha256"],
        private_manifest_literal_sha256=private_literal,
    )
    public_literal = hashlib.sha256(
        preflight._artifact_bytes(public_artifact)
    ).hexdigest()
    receipt = preflight._build_execution_authority_receipt(
        public_artifact=public_artifact,
        private_manifest=private_manifest,
        preflight_commit="3" * 40,
        preflight_tree="4" * 40,
        implementation_commit=SHA1_A,
        implementation_tree=SHA1_B,
        public_artifact_git_blob_sha1="5" * 40,
        public_artifact_literal_sha256=public_literal,
        direct_head=True,
    )
    authority = preflight._store_execution_authority(receipt)
    plan = runner.build_model_plan(projection)
    assert runner.build_attempt_authority(
        execution_authority=authority,
        projection=projection,
        commitments=commitments,
        plan=plan,
    ) == authority


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        (
            "schema_version",
            "aapl-sec-gemma-lean-science-v3-11-streaming-bridge-v1",
            "preflight_bridge_manifest_invalid",
        ),
        (
            "calendar_sessions_sha256",
            "f" * 64,
            "preflight_bridge_manifest_invalid",
        ),
        (
            "nullable_filename_parity",
            False,
            "preflight_bridge_manifest_invalid",
        ),
        (
            "no_fabricated_primary_url",
            False,
            "preflight_bridge_manifest_invalid",
        ),
    ],
)
def test_bridge_validator_rejects_rehashed_semantic_mutations(
    field: str, replacement: object, code: str
) -> None:
    changed = _bridge_manifest()
    changed[field] = replacement
    body = dict(changed)
    body.pop("bridge_sha256")
    changed["bridge_sha256"] = contract.canonical_sha256(body)
    _assert_code(code, preflight.validate_bridge_manifest, changed)


@pytest.mark.parametrize(
    "field",
    [
        "document_count",
        "filename_present_count",
        "filename_missing_count",
        "documents_sha256",
        "records_sha256",
        "events_sha256",
        "compatibility_manifest_sha256",
        "universe_sha256",
        "content_manifest_sha256",
        "calendar_sessions_sha256",
        "universe_event_proofs_sha256",
        "source_order_sha256",
        "prior_links_sha256",
    ],
)
def test_bridge_commitment_parity_rejects_every_cross_object_mutation(
    field: str,
) -> None:
    changed = _commitments()
    changed[field] = (
        changed[field] + 1
        if type(changed[field]) is int
        else "f" * 64
    )
    _assert_code(
        "preflight_bridge_request_parity_invalid",
        preflight._validate_bridge_commitment_parity,
        _bridge_manifest(),
        changed,
    )


def test_commitment_validator_rejects_shape_calendar_and_byte_bounds() -> None:
    missing = _commitments()
    missing.pop("model_plan_sha256")
    _assert_code(
        "preflight_request_commitments_shape_invalid",
        preflight.validate_request_commitments,
        missing,
    )
    extra = _commitments()
    extra["unregistered"] = True
    _assert_code(
        "preflight_request_commitments_shape_invalid",
        preflight.validate_request_commitments,
        extra,
    )
    calendar = _commitments()
    calendar["calendar_sessions_sha256"] = "f" * 64
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.validate_request_commitments,
        calendar,
    )
    bounds = _commitments()
    bounds["minimum_request_byte_count"] = 2_001
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.validate_request_commitments,
        bounds,
    )


def test_run_preflight_seals_private_content_addressed_and_redacted_public(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    calls: dict[str, int] = {}
    _install_creation_pretrust(root)
    artifact = preflight.run_preflight(
        root,
        dependencies=_dependencies(root, calls=calls),
    )
    assert calls == {"authenticate": 1, "projection": 1}
    assert artifact["status"] == "passed"
    assert artifact["development_authorized"] is False
    assert artifact["eligible_for_public_seal"] is True
    assert artifact["counts"]["canonical_requests"] == 75
    assert artifact["counts"]["pilots"] == 5
    assert artifact["counts"]["remaining"] == 70
    assert artifact["effect_counts"] == contract.build_effect_budgets()[
        "zero_effect_preflight"
    ]
    assert preflight.validate_public_preflight_artifact(artifact) == artifact

    public_path = root / contract.PREFLIGHT_ARTIFACT_PATH
    public_payload = public_path.read_bytes()
    assert public_payload == contract.canonical_json_bytes(artifact) + b"\n"
    assert b"SYNTHETIC-PRIVATE-CONTACT" not in public_payload
    assert str(root).encode() not in public_payload
    assert b"0000320193-24-000001" not in public_payload
    assert b"https://" not in public_payload
    assert b"canonical_request_body" not in public_payload
    assert not any("path" in key for key in artifact["aggregates"])

    private_root = root / contract.PRIVATE_PREFLIGHT_NAMESPACE
    manifests = list((private_root / "manifests").iterdir())
    assert len(manifests) == 1
    private_payload = manifests[0].read_bytes()
    assert manifests[0].name == f"{artifact['private_manifest_literal_sha256']}.json"
    private_value = json.loads(private_payload)
    assert preflight.validate_private_preflight_manifest(private_value) == private_value
    assert private_value["private_manifest_sha256"] == artifact[
        "private_manifest_sha256"
    ]
    assert b"SYNTHETIC-PRIVATE-CONTACT" not in private_payload
    assert str(root).encode() not in private_payload
    assert private_value["privacy"] == {
        "aggregate_only_bridge_serialized": True,
        "aggregate_only_request_commitments_serialized": True,
        "readable_contact_stored": False,
        "compact_manifest_absolute_private_path_stored": False,
        "qualification_intent_absolute_paths_stored": True,
    }
    qualification_intent = (
        private_root
        / "qualification"
        / contract.QUALIFICATION_PHASE_V313
        / "collection"
        / "intent.json"
    ).read_bytes()
    qualification_intent_value = json.loads(qualification_intent)
    assert any(
        str(root) in argument
        for argument in qualification_intent_value["argv"]
    )
    assert set(item.name for item in private_root.iterdir()) == {
        "preflight-intent.json",
        "manifests",
        "preflight-complete.json",
        "qualification",
        "runtime",
    }


@pytest.mark.parametrize("secret_kind", ("contact", "v38_path"))
def test_preflight_rejects_secret_in_authenticated_qualification_tree(
    tmp_path: Path,
    secret_kind: str,
) -> None:
    root = tmp_path / secret_kind
    _make_repo(root)
    contact = b"SYNTHETIC-PRIVATE-CONTACT"
    v38_path = str(root / preflight.V38_PRIVATE_ROOT).encode("ascii")
    injected = contact if secret_kind == "contact" else v38_path
    base = _dependencies(root)
    dependencies = preflight.PreflightDependencies(
        **{
            **base.__dict__,
            "run_qualification": lambda run_root: (
                _write_qualification_private_state(
                    run_root,
                    _repository_snapshot(),
                    log_injection=injected,
                )
            ),
        }
    )
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_private_privacy_scan_failed",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    _assert_failed_preflight_artifact(
        root, "preflight_private_privacy_scan_failed"
    )


def test_preflight_is_once_only_and_rerun_does_not_invoke_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    calls: dict[str, int] = {}
    dependencies = _dependencies(root, calls=calls)
    _install_creation_pretrust(root)
    preflight.run_preflight(root, dependencies=dependencies)
    assert calls == {"authenticate": 1, "projection": 1}
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_already_consumed",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    assert calls == {"authenticate": 1, "projection": 1}


def test_preboundary_nonzero_is_retryable_and_later_stage_rejection_consumes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nonzero"
    _make_repo(root)
    nonzero = contract.build_effect_budgets()["zero_effect_preflight"]
    nonzero["sec_requests"] = 1
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_nonzero_effect",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, effects=nonzero),
    )
    assert not (root / contract.PRIVATE_PREFLIGHT_NAMESPACE).exists()
    assert not (root / contract.PREFLIGHT_ARTIFACT_PATH).exists()

    root = tmp_path / "later-stage"
    _make_repo(root)
    changed = _commitments()
    changed["confirmation_or_final_opened"] = True
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, commitments=changed),
    )
    _assert_failed_preflight_artifact(
        root, "preflight_request_commitments_invalid"
    )


def test_privacy_uncertainty_fails_before_publication(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    # The implementation commit is deliberately declared forbidden; it would
    # occur in both manifests, so the scan must fail closed.
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_private_privacy_scan_failed",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, tokens=(SHA1_A, str(root))),
    )
    failed = _assert_failed_preflight_artifact(
        root, "preflight_private_privacy_scan_failed"
    )
    payload = contract.canonical_json_bytes(failed)
    assert SHA1_A.encode("utf-8") not in payload
    assert str(root).encode("utf-8") not in payload


def test_dependency_failure_is_preserved_redacted_and_never_rerunnable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    base = _dependencies(root)
    calls = {"authenticate": 0}

    class UnsafeDependencyError(RuntimeError):
        code = "UNSAFE/Synthetic Private Code"

    def fail_with_private_detail(_root: Path) -> object:
        calls["authenticate"] += 1
        raise UnsafeDependencyError(
            "Synthetic Private Contact must never be published"
        )

    dependencies = preflight.PreflightDependencies(
        **{**base.__dict__, "authenticate_source": fail_with_private_detail}
    )
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_dependency_failed",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    failed = _assert_failed_preflight_artifact(
        root, "preflight_dependency_failed"
    )
    assert "Synthetic Private Contact" not in json.dumps(failed)
    failed_payload = (root / contract.PREFLIGHT_ARTIFACT_PATH).read_bytes()
    assert b"UNSAFE/Synthetic Private Code" not in failed_payload
    assert b"Synthetic Private Contact" not in failed_payload
    private_intent = (
        root
        / contract.PRIVATE_PREFLIGHT_NAMESPACE
        / preflight.PRIVATE_INTENT_FILENAME
    ).read_bytes()
    assert private_intent == preflight._artifact_bytes(preflight._preflight_intent())
    assert b"UNSAFE/Synthetic Private Code" not in private_intent
    assert b"Synthetic Private Contact" not in private_intent
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_already_consumed",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    assert calls == {"authenticate": 1}
    assert list((root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "manifests").iterdir()) == []


def test_safe_bridge_failure_code_is_preserved_in_redacted_terminal_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "safe-bridge-code"
    _make_repo(root)
    base = _dependencies(root)

    def fail_with_bridge_code(_authority: object) -> object:
        raise bridge.BridgeViolation("legacy_record_invalid")

    dependencies = preflight.PreflightDependencies(
        **{**base.__dict__, "build_projection": fail_with_bridge_code}
    )
    _install_creation_pretrust(root)
    _assert_code(
        "legacy_record_invalid",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    failed = _assert_failed_preflight_artifact(root, "legacy_record_invalid")
    assert set(failed) == preflight._PUBLIC_FAILED_PREFLIGHT_FIELDS

    private_intent = (
        root
        / contract.PRIVATE_PREFLIGHT_NAMESPACE
        / preflight.PRIVATE_INTENT_FILENAME
    ).read_bytes()
    assert private_intent == preflight._artifact_bytes(preflight._preflight_intent())
    assert b"legacy_record_invalid" not in private_intent


def test_interrupted_failed_preflight_pending_is_promoted_before_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    target = root / contract.PREFLIGHT_ARTIFACT_PATH
    pending = target.with_name(f"{target.name}.v313-pending")
    real_replace = preflight.os.replace

    def interrupt_promotion(source: object, destination: object) -> None:
        if Path(source) == pending and Path(destination) == target:
            raise OSError("synthetic interruption after durable pending write")
        real_replace(source, destination)

    monkeypatch.setattr(preflight.os, "replace", interrupt_promotion)
    changed = _commitments()
    changed["confirmation_or_final_opened"] = True
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_failure_preservation_failed",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, commitments=changed),
    )
    assert pending.is_file()
    assert not target.exists()

    monkeypatch.setattr(preflight.os, "replace", real_replace)
    calls: dict[str, int] = {}
    _install_creation_pretrust(root)
    _assert_code(
        "preflight_already_consumed",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, calls=calls),
    )
    assert calls == {}
    assert not pending.exists()
    _assert_failed_preflight_artifact(
        root, "preflight_request_commitments_invalid"
    )


def test_public_and_private_self_hashes_reject_rehashed_semantic_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    _install_creation_pretrust(root)
    artifact = preflight.run_preflight(root, dependencies=_dependencies(root))
    changed = copy.deepcopy(artifact)
    changed["counts"]["canonical_requests"] = 74
    unsigned = dict(changed)
    unsigned.pop("public_artifact_sha256")
    changed["public_artifact_sha256"] = contract.canonical_sha256(unsigned)
    _assert_code(
        "preflight_public_artifact_invalid",
        preflight.validate_public_preflight_artifact,
        changed,
    )

    private_path = next(
        (root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "manifests").iterdir()
    )
    private_value = json.loads(private_path.read_bytes())
    private_value["request_commitments"]["request_count"] = 74
    unsigned = dict(private_value)
    unsigned.pop("private_manifest_sha256")
    private_value["private_manifest_sha256"] = contract.canonical_sha256(unsigned)
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.validate_private_preflight_manifest,
        private_value,
    )

    public_binding = {"qualification": _qualification()}
    private_binding = {"qualification": copy.deepcopy(_qualification())}
    private_binding["qualification"]["phases"][0][
        "collection_log_sha256"
    ] = "f" * 64
    _assert_code(
        "execution_preflight_public_private_mismatch",
        preflight._validate_public_private_qualification_binding,
        public_binding,
        private_binding,
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def test_inspect_pushed_implementation_rejects_merge_with_prereg_first_parent(
    tmp_path: Path,
) -> None:
    source_repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "implementation-merge"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "fetch", str(source_repo), contract.PREREGISTRATION_COMMIT)
    _git(
        root,
        "checkout",
        "-b",
        contract.BRANCH_NAME,
        contract.PREREGISTRATION_COMMIT,
    )
    for index, relative in enumerate(contract.IMPLEMENTATION_ALLOWED_PATHS):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"# synthetic implementation {index}\n".encode("ascii"))
    _git(root, "add", "--", *contract.IMPLEMENTATION_ALLOWED_PATHS)
    merge_tree = _git(root, "write-tree")
    merge_commit = _git(
        root,
        "commit-tree",
        merge_tree,
        "-p",
        contract.PREREGISTRATION_COMMIT,
        "-p",
        contract.BASE_COMMIT,
        "-m",
        "synthetic implementation merge",
    )
    _git(
        root,
        "update-ref",
        f"refs/heads/{contract.BRANCH_NAME}",
        merge_commit,
        contract.PREREGISTRATION_COMMIT,
    )

    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    assert parents == [
        merge_commit,
        contract.PREREGISTRATION_COMMIT,
        contract.BASE_COMMIT,
    ]
    changed = _git(
        root,
        "diff",
        "--name-status",
        contract.PREREGISTRATION_COMMIT,
        merge_commit,
    ).splitlines()
    assert changed == sorted(
        (
            f"A\t{relative}"
            for relative in contract.IMPLEMENTATION_ALLOWED_PATHS
        ),
        key=lambda value: value.encode("utf-8"),
    )
    _assert_code(
        "preflight_implementation_ancestry_invalid",
        preflight.inspect_pushed_implementation,
        root,
    )


@pytest.mark.parametrize("descendant", ("pause", "continuation"))
def test_reconstruct_frozen_authority_rejects_merge_descendants(
    tmp_path: Path, descendant: str
) -> None:
    remote = tmp_path / f"{descendant}-remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    root = tmp_path / f"{descendant}-work"
    subprocess.run(
        ["git", "clone", str(remote), str(root)], check=True, capture_output=True
    )
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "checkout", "-b", contract.BRANCH_NAME)
    preflight_path = root / contract.PREFLIGHT_ARTIFACT_PATH
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_bytes(b'{"synthetic_preflight":true}')
    _git(root, "add", "--", contract.PREFLIGHT_ARTIFACT_PATH)
    _git(root, "commit", "-m", "synthetic preflight")
    preflight_commit = _git(root, "rev-parse", "HEAD")

    first_parent = preflight_commit
    if descendant == "continuation":
        pause_path = root / contract.PAUSE_ARTIFACT_PATH
        pause_path.write_bytes(b'{"synthetic_pause":true}')
        _git(root, "add", "--", contract.PAUSE_ARTIFACT_PATH)
        _git(root, "commit", "-m", "synthetic single-parent pause")
        first_parent = _git(root, "rev-parse", "HEAD")
        target_path = root / contract.CONTINUATION_PREREGISTRATION_PATH
        target_relative = contract.CONTINUATION_PREREGISTRATION_PATH
        target_payload = b"# Synthetic merge continuation\n"
    else:
        target_path = root / contract.PAUSE_ARTIFACT_PATH
        target_relative = contract.PAUSE_ARTIFACT_PATH
        target_payload = b'{"synthetic_merge_pause":true}'

    first_parent_tree = _git(root, "rev-parse", f"{first_parent}^{{tree}}")
    side_parent = _git(
        root,
        "commit-tree",
        first_parent_tree,
        "-p",
        first_parent,
        "-m",
        "synthetic empty side parent",
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(target_payload)
    _git(root, "add", "--", target_relative)
    merge_tree = _git(root, "write-tree")
    merge_commit = _git(
        root,
        "commit-tree",
        merge_tree,
        "-p",
        first_parent,
        "-p",
        side_parent,
        "-m",
        f"synthetic merge {descendant}",
    )
    _git(
        root,
        "update-ref",
        f"refs/heads/{contract.BRANCH_NAME}",
        merge_commit,
        first_parent,
    )
    _git(root, "push", "-u", "origin", contract.BRANCH_NAME)
    assert len(_git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3
    assert _git(root, "diff", "--name-status", first_parent, merge_commit) == (
        f"A\t{target_relative}"
    )
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""
    _assert_code(
        "execution_preflight_descendant_invalid",
        preflight.reconstruct_frozen_execution_authority,
        root,
    )


def test_pushed_execution_authority_replays_public_private_and_source_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_repo = Path(__file__).resolve().parents[1]
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", str(remote), str(work)], check=True, capture_output=True)
    _git(work, "config", "core.longpaths", "true")
    _git(work, "config", "user.name", "Synthetic Test")
    _git(work, "config", "user.email", "synthetic@example.invalid")
    _git(work, "fetch", str(source_repo), contract.PREREGISTRATION_COMMIT)
    _git(work, "checkout", "-b", contract.BRANCH_NAME, contract.PREREGISTRATION_COMMIT)
    for index, relative in enumerate(contract.IMPLEMENTATION_ALLOWED_PATHS):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# synthetic implementation {index}\n".encode("utf-8"))
    _git(work, "add", "--", *contract.IMPLEMENTATION_ALLOWED_PATHS)
    _git(work, "commit", "-m", "synthetic v3.13 implementation")
    _git(work, "push", "-u", "origin", contract.BRANCH_NAME)

    config = work / "data" / "local_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    fake_contact = "Synthetic Contact synthetic@example.invalid"
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": fake_contact}}),
        encoding="utf-8",
    )
    dependencies = _dependencies(work)
    repository_holder: dict[str, dict[str, object]] = {}

    def inspect_repository(root: Path) -> dict[str, object]:
        snapshot = preflight.inspect_pushed_implementation(root)
        repository_holder["snapshot"] = snapshot
        return snapshot

    dependencies = preflight.PreflightDependencies(
        inspect_repository=inspect_repository,
        create_runtime_authority=lambda root, _repository: (
            _write_runtime_authority_bundle(root)
        ),
        authenticate_source=dependencies.authenticate_source,
        build_projection=dependencies.build_projection,
        build_request_commitments=dependencies.build_request_commitments,
        effect_snapshot=dependencies.effect_snapshot,
        run_qualification=lambda root: _write_qualification_private_state(
            root, repository_holder["snapshot"]
        ),
        privacy_tokens=lambda root: (
            fake_contact,
            str(root),
            str(root / preflight.V38_PRIVATE_ROOT),
            str(root / contract.PRIVATE_NAMESPACE),
        ),
    )
    _install_creation_pretrust(work, inspect_repository(work))
    artifact = preflight.run_preflight(work, dependencies=dependencies)
    _git(work, "add", "--", contract.PREFLIGHT_ARTIFACT_PATH)
    _git(work, "commit", "-m", "synthetic v3.13 preflight evidence")
    _git(work, "push", "origin", contract.BRANCH_NAME)

    qualification_scan_calls: list[str] = []
    real_qualification_scan = preflight._scan_qualification_private_tree

    def scan_qualification(*args, **kwargs):
        qualification_scan_calls.append(kwargs["code"])
        return real_qualification_scan(*args, **kwargs)

    monkeypatch.setattr(
        preflight, "_scan_qualification_private_tree", scan_qualification
    )

    # This test isolates public/private/source revision replay.  Its tiny
    # runtime fixture is intentionally not a production runtime authority;
    # bypass only the runtime loader/session seam here so production keeps the
    # strict schemas and fresh-bootstrap requirement.
    def load_synthetic_runtime_bundle(root: Path) -> dict[str, object]:
        runtime_root = (
            root
            / contract.PRIVATE_PREFLIGHT_NAMESPACE
            / preflight.PRIVATE_RUNTIME_DIRECTORY
        )
        loaded_root = (
            runtime_root / preflight.PRIVATE_RUNTIME_LOADED_CODE_DIRECTORY
        )

        def read(path: Path) -> dict[str, object]:
            return json.loads(path.read_text(encoding="utf-8"))

        return {
            "repository_runtime_manifest": read(
                runtime_root / preflight.PRIVATE_RUNTIME_REPOSITORY_FILENAME
            ),
            "execution_dependency_manifest": read(
                runtime_root / preflight.PRIVATE_RUNTIME_DEPENDENCY_FILENAME
            ),
            "loaded_code_manifests": {
                name: read(loaded_root / f"{name}.json")
                for name in preflight.RUNTIME_CHECKPOINT_NAMES
            },
            "runtime_binding_authority": read(
                runtime_root / preflight.PRIVATE_RUNTIME_AUTHORITY_FILENAME
            ),
        }

    monkeypatch.setattr(
        preflight, "_load_private_runtime_bundle", load_synthetic_runtime_bundle
    )
    monkeypatch.setattr(
        preflight,
        "_initialize_runtime_consumption_session",
        lambda *_args, **_kwargs: {},
    )
    receipt = preflight.authenticate_execution_preflight(work)
    assert qualification_scan_calls == [
        "execution_preflight_private_privacy_failed"
    ]
    authority = preflight.load_execution_authority(work)
    assert receipt["development_authorized"] is True
    assert receipt["authority_scope"] == "direct_pushed_preflight"
    assert receipt["pushed_preflight_gate_passed"] is True
    assert receipt["implementation_commit"] == artifact["implementation_commit"]
    assert receipt["public_artifact_sha256"] == artifact["public_artifact_sha256"]
    assert receipt["canonical_requests_sha256"] == SHA64["requests"]
    assert set(authority) == {
        "plan",
        "attempt",
        "implementation",
        "preflight",
        "source",
        "science",
        "effect_budget",
        "request_order",
        "pilot_order",
    }
    assert authority["attempt"] == contract.DEVELOPMENT_ATTEMPT_ID
    assert authority["request_order"]["canonical_requests_sha256"] == SHA64["requests"]
    assert fake_contact not in json.dumps(authority, sort_keys=True)
    assert str(work) not in json.dumps(authority, sort_keys=True)

    pause_path = work / contract.PAUSE_ARTIFACT_PATH
    pause_path.parent.mkdir(parents=True, exist_ok=True)
    pause_path.write_bytes(b'{"synthetic_pause":true}\n')
    _git(work, "add", "--", contract.PAUSE_ARTIFACT_PATH)
    _git(work, "commit", "-m", "synthetic clean pause")
    _git(work, "push", "origin", contract.BRANCH_NAME)
    assert preflight.reconstruct_frozen_execution_authority(work) == authority
    assert preflight.load_execution_authority(work) == authority
    _assert_code(
        "execution_preflight_ancestry_invalid",
        preflight.authenticate_execution_preflight,
        work,
    )

    continuation_path = work / contract.CONTINUATION_PREREGISTRATION_PATH
    continuation_path.parent.mkdir(parents=True, exist_ok=True)
    continuation_path.write_bytes(b"# Synthetic continuation\n")
    _git(work, "add", "--", contract.CONTINUATION_PREREGISTRATION_PATH)
    _git(work, "commit", "-m", "synthetic continuation preregistration")
    _git(work, "push", "origin", contract.BRANCH_NAME)
    assert preflight.reconstruct_frozen_execution_authority(work) == authority
    assert preflight.load_execution_authority(work) == authority


def test_load_private_contact_uses_only_synthetic_ignored_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    config = root / "data" / "local_config.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": "Synthetic Contact"}}),
        encoding="utf-8",
    )
    assert preflight.load_private_contact(root) == "Synthetic Contact"


def _changed_stat(details: os.stat_result) -> SimpleNamespace:
    names = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_file_attributes",
    )
    values = {
        name: getattr(details, name, 0)
        for name in names
    }
    values["st_mtime_ns"] += 1
    return SimpleNamespace(**values)


def test_descriptor_reader_rejects_identity_change_after_complete_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "authority.json"
    path.write_bytes(b'{"safe":true}\n')
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        details = real_fstat(descriptor)
        return _changed_stat(details) if calls == 2 else details

    monkeypatch.setattr(preflight.os, "fstat", changing_fstat)
    _assert_code(
        "synthetic_descriptor_swap",
        preflight._read_regular_file,
        path,
        maximum=1024,
        code="synthetic_descriptor_swap",
    )
    assert calls == 2


def test_streaming_privacy_reader_rejects_identity_change_after_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private.json"
    path.write_bytes(b'{"safe":"value"}\n')
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        details = real_fstat(descriptor)
        return _changed_stat(details) if calls == 2 else details

    monkeypatch.setattr(preflight.os, "fstat", changing_fstat)
    _assert_code(
        "synthetic_stream_swap",
        preflight._stream_file_sha256_and_privacy,
        path,
        byte_forms=frozenset({b"forbidden"}),
        code="synthetic_stream_swap",
    )
    assert calls == 2


def test_private_contact_rejects_descriptor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    config = root / "data" / "local_config.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": "Synthetic Contact"}}),
        encoding="utf-8",
    )
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        details = real_fstat(descriptor)
        return _changed_stat(details) if calls == 2 else details

    monkeypatch.setattr(preflight.os, "fstat", changing_fstat)
    _assert_code(
        "preflight_private_contact_invalid",
        preflight.load_private_contact,
        root,
    )


def test_production_authority_reads_do_not_use_path_read_bytes() -> None:
    source = Path(preflight.__file__).read_text(encoding="utf-8")
    assert ".read_bytes(" not in source


def test_privacy_scanners_reject_json_escaped_windows_paths_and_parsed_values() -> None:
    windows_path = r"C:\Users\Synthetic\private evidence"
    escaped = json.dumps(
        {"nested": {"value": windows_path}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert windows_path.encode("utf-8") not in escaped
    _assert_code(
        "preflight_private_privacy_scan_failed",
        preflight._scan_private_bytes,
        escaped,
        (windows_path.encode("utf-8"),),
    )

    contact = "SynthetÃ­c Contact"
    unicode_escaped = json.dumps(
        {"contact": contact}, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    assert contact.encode("utf-8") not in unicode_escaped
    _assert_code(
        "preflight_public_privacy_scan_failed",
        preflight._scan_public_bytes,
        unicode_escaped,
        (contact.encode("utf-8"),),
    )


def test_private_tree_token_split_allows_bound_paths_only(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    contact = b"Synthetic Private Contact"
    v38 = str(root / preflight.V38_PRIVATE_ROOT).encode("ascii")
    v313 = str(root / contract.PRIVATE_NAMESPACE).encode("ascii")
    split = preflight._private_tree_forbidden_tokens(
        root,
        (contact, str(root).encode("ascii"), v38, v313),
    )
    assert split == (contact, v38)


@pytest.mark.parametrize(
    "private_token",
    ("SynthetÃ­c Private Contact", "SynthetÃ­c Private Contact".encode("utf-8")),
)
def test_privacy_token_collection_rejects_non_ascii_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    private_token: str | bytes,
) -> None:
    with pytest.raises(preflight.V313PreflightError) as captured:
        preflight._privacy_token_bytes(
            tmp_path,
            lambda _root: ("safe-ascii-token", private_token),
        )
    assert captured.value.code == "preflight_privacy_tokens_invalid"
    assert "SynthetÃ­c Private Contact" not in str(captured.value)
    assert "SynthetÃ­c Private Contact" not in repr(captured.value)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_result_effect_validation_recursively_thaws_store_mapping_proxies() -> None:
    snapshot = SimpleNamespace(
        request_intent_count=0,
        response_count=0,
        checkpoint_count=0,
        yahoo_body_bytes=0,
        yahoo_intent_count=0,
        identity_intent_count=0,
        gemma_intent_count=0,
        yahoo_response_count=0,
        identity_response_count=0,
        gemma_response_count=0,
    )
    ordinary = runner.build_effect_report(snapshot)
    frozen = MappingProxyType(
        {
            key: (
                MappingProxyType(dict(value))
                if isinstance(value, dict)
                else value
            )
            for key, value in ordinary.items()
        }
    )
    assert preflight._validate_effect_report(
        frozen,
        snapshot=snapshot,
        completed_terminal=False,
        parent_kind="preflight",
    ) == ordinary


def _preflight_receipt_for_result(
    *, implementation: str, implementation_tree: str, preflight_commit: str, preflight_tree: str
) -> dict[str, object]:
    bridge = _bridge_manifest()
    commitments = _commitments()
    return {
        "schema_version": preflight.EXECUTION_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": "descendant_reconstruction",
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "preflight_commit": preflight_commit,
        "preflight_tree": preflight_tree,
        "implementation_commit": implementation,
        "implementation_tree": implementation_tree,
        "public_artifact_git_blob_sha1": "a" * 40,
        "public_artifact_sha256": "a" * 64,
        "public_artifact_literal_sha256": "b" * 64,
        "private_manifest_sha256": "c" * 64,
        "private_manifest_literal_sha256": "d" * 64,
        "bridge_sha256": bridge["bridge_sha256"],
        "compatibility_manifest_sha256": commitments[
            "compatibility_manifest_sha256"
        ],
        "universe_sha256": commitments["universe_sha256"],
        "content_manifest_sha256": commitments["content_manifest_sha256"],
        "calendar_sessions_sha256": commitments["calendar_sessions_sha256"],
        "universe_event_proofs_sha256": commitments[
            "universe_event_proofs_sha256"
        ],
        "source_order_sha256": commitments["source_order_sha256"],
        "prior_links_sha256": commitments["prior_links_sha256"],
        "request_count": commitments["request_count"],
        "remaining_count": commitments["remaining_count"],
        "pilot_count": commitments["pilot_count"],
        "canonical_requests_sha256": commitments["canonical_requests_sha256"],
        "model_slice_sha256": commitments["model_slice_sha256"],
        "canonical_request_index_sha256": commitments[
            "canonical_request_index_sha256"
        ],
        "model_plan_sha256": commitments["model_plan_sha256"],
        "pilot_order_sha256": commitments["pilot_order_sha256"],
        "remaining_order_sha256": commitments["remaining_order_sha256"],
        "minimum_request_byte_count": commitments[
            "minimum_request_byte_count"
        ],
        "maximum_request_byte_count": commitments[
            "maximum_request_byte_count"
        ],
        "production_source_inventory_sha256": "e" * 64,
        "test_source_inventory_sha256": "f" * 64,
        "runtime_binding_authority_sha256": "9" * 64,
        "effect_counts": contract.build_effect_budgets()["zero_effect_preflight"],
        "pushed_preflight_gate_passed": True,
        "development_authorized": False,
        "execution_authority_sha256": "0" * 64,
    }


class _FakeReadOnlyStore:
    def __init__(
        self,
        snapshot: object,
        receipt: object | None,
        candidate: object | None = None,
    ) -> None:
        self.snapshot = snapshot
        self._receipt = receipt
        self.terminal_candidate = candidate
        self.closed = False

    def committed_terminal_evidence(self) -> object:
        if self._receipt is None:
            raise RuntimeError("synthetic_terminal_not_committed")
        return self._receipt

    def close(self) -> None:
        self.closed = True


def _synthetic_continuation_prerequisites(
    authority: dict[str, object],
) -> tuple[object, dict[str, object], dict[str, object], dict[str, object]]:
    execution_calls = tuple(
        SimpleNamespace(
            execution_ordinal=index,
            request={
                "request_sha256": contract.sha256_bytes(
                    f"synthetic-request-{index:03d}".encode("ascii")
                )
            },
            request_byte_count=1_000 + index,
        )
        for index in range(1, contract.DEVELOPMENT_DOCUMENT_COUNT + 1)
    )
    plan_body = {
        "remaining_order_sha256": authority["request_order"][
            "remaining_order_sha256"
        ]
    }
    plan = SimpleNamespace(
        execution_calls=execution_calls,
        manifest={
            **plan_body,
            "model_plan_sha256": contract.canonical_sha256(plan_body),
        },
    )
    generation_events = [
        contract.sha256_bytes(f"pilot-event-{index}".encode("ascii"))
        for index in range(1, contract.DEVELOPMENT_PILOT_COUNT + 1)
    ]
    guard_body = {
        "schema_version": runner.RUNTIME_GUARD_SCHEMA_VERSION,
        "segment_id": "pilot",
        "store_segment_id": "initial",
        "generation_count": contract.DEVELOPMENT_PILOT_COUNT,
        "pre_runtime_receipt_sha256": "a" * 64,
        "post_runtime_receipt_sha256": "b" * 64,
        "stable_runtime_identity_sha256": "c" * 64,
        "ordered_generation_response_event_sha256s": generation_events,
        "ordered_generation_response_events_sha256": contract.canonical_sha256(
            generation_events
        ),
        "raw_show_hash_is_diagnostic_only": True,
        "modified_at_is_excluded_only": True,
        "identity_http_request_count": 4,
        "retry_count": 0,
    }
    pilot_guard = {
        **guard_body,
        "segment_guard_sha256": contract.canonical_sha256(guard_body),
    }
    durations = [600_000_000_000 + index for index in range(5)]
    pilot_rows = [
        {
            "execution_ordinal": call.execution_ordinal,
            "request_sha256": call.request["request_sha256"],
            "request_byte_count": call.request_byte_count,
            "duration_ns": duration,
        }
        for call, duration in zip(
            execution_calls[: contract.DEVELOPMENT_PILOT_COUNT],
            durations,
            strict=True,
        )
    ]
    latency_body = {
        "schema_version": runner.LATENCY_RECEIPT_SCHEMA_VERSION,
        "selection": "five_largest_request_bytes_desc_accession_asc",
        "remaining_order": "availability_acceptance_accession_ascending",
        "pilot_rows": pilot_rows,
        "pilot_order_sha256": contract.canonical_sha256(pilot_rows),
        "pilot_count": contract.DEVELOPMENT_PILOT_COUNT,
        "remaining_count": contract.DEVELOPMENT_REMAINING_COUNT,
        "formula": "sum(pilot_duration_ns)+70*max(pilot_duration_ns)",
        "projected_ns": contract.projected_pilot_ns(durations),
        "threshold_ns": contract.PILOT_PROJECTED_THRESHOLD_NS,
        "pause_required": True,
        "timing_interval": (
            "monotonic_ns_after_durable_intent_before_transport_through_"
            "bounded_body_framing_and_close_before_persistence_or_semantic_parse"
        ),
    }
    latency = {
        **latency_body,
        "latency_receipt_sha256": contract.canonical_sha256(latency_body),
    }
    pause_request_count = (
        contract.YAHOO_REQUEST_COUNT + 4 + contract.DEVELOPMENT_PILOT_COUNT
    )
    pause_snapshot = SimpleNamespace(
        yahoo_intent_count=contract.YAHOO_REQUEST_COUNT,
        identity_intent_count=4,
        gemma_intent_count=contract.DEVELOPMENT_PILOT_COUNT,
        yahoo_response_count=contract.YAHOO_REQUEST_COUNT,
        identity_response_count=4,
        gemma_response_count=contract.DEVELOPMENT_PILOT_COUNT,
        request_intent_count=pause_request_count,
        response_count=pause_request_count,
        checkpoint_count=pause_request_count + 1,
        yahoo_body_bytes=123,
    )
    pause = runner.build_public_pause_artifact(
        authority=authority,
        plan=plan,
        pilot_guard=pilot_guard,
        latency_receipt=latency,
        effect_report=runner.build_effect_report(pause_snapshot),
    )
    return plan, pilot_guard, latency, pause


def _make_result_gate_repo(
    tmp_path: Path,
    *,
    commit_result: bool = True,
    completed: bool = False,
    continued: bool = False,
    mutated_pause: bool = False,
    arbitrary_continuation: bool = False,
) -> dict[str, object]:
    assert not continued or completed
    assert not (mutated_pause or arbitrary_continuation) or continued
    remote = tmp_path / "result-remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    root = tmp_path / "result-work"
    subprocess.run(["git", "clone", str(remote), str(root)], check=True, capture_output=True)
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "checkout", "-b", contract.BRANCH_NAME)
    comparison = (
        "| Approach | Design | Development | Real money | Decision |\n"
        "|---|---|---|---|---|\n"
        "| SEC/Gemma lean evidence v3.8 | Frozen | Passed | No | Frozen |\n"
        "| SEC/Gemma lean science v3.12 | Frozen | Rejected | No | Preserved |\n"
    ).encode("utf-8")
    comparison_path = root / contract.COMPARISON_PATH
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_bytes(comparison)
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    _git(root, "add", "--", ".gitignore", contract.COMPARISON_PATH)
    _git(root, "commit", "-m", "synthetic implementation parent")
    implementation = _git(root, "rev-parse", "HEAD")
    implementation_tree = _git(root, "rev-parse", "HEAD^{tree}")

    preflight_path = root / contract.PREFLIGHT_ARTIFACT_PATH
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_bytes(b'{"synthetic_preflight":true}\n')
    _git(root, "add", "--", contract.PREFLIGHT_ARTIFACT_PATH)
    _git(root, "commit", "-m", "synthetic preflight")
    preflight_commit = _git(root, "rev-parse", "HEAD")
    preflight_tree = _git(root, "rev-parse", "HEAD^{tree}")
    receipt_authority = _preflight_receipt_for_result(
        implementation=implementation,
        implementation_tree=implementation_tree,
        preflight_commit=preflight_commit,
        preflight_tree=preflight_tree,
    )
    authority = preflight._store_execution_authority(receipt_authority)
    plan, pilot_guard, pilot_latency, canonical_pause = (
        _synthetic_continuation_prerequisites(authority)
    )
    invocation_parent = preflight_commit
    invocation_parent_kind = "preflight"
    continuation_sha256: str | None = None
    committed_pause: dict[str, object] | None = None
    continuation: bytes | None = None
    if continued:
        committed_pause = copy.deepcopy(canonical_pause)
        if mutated_pause:
            committed_pause["implementation_commit"] = "f" * 40
            pause_body = dict(committed_pause)
            pause_body.pop("pause_artifact_sha256")
            committed_pause["pause_artifact_sha256"] = contract.canonical_sha256(
                pause_body
            )
        pause_path = root / contract.PAUSE_ARTIFACT_PATH
        pause_path.write_bytes(contract.canonical_json_bytes(committed_pause))
        _git(root, "add", "--", contract.PAUSE_ARTIFACT_PATH)
        _git(root, "commit", "-m", "synthetic pause")
        continuation = (
            b"# Arbitrary but historically hash-bound continuation\n"
            if arbitrary_continuation
            else runner.build_continuation_preregistration(committed_pause)
        )
        continuation_path = root / contract.CONTINUATION_PREREGISTRATION_PATH
        continuation_path.parent.mkdir(parents=True, exist_ok=True)
        continuation_path.write_bytes(continuation)
        _git(root, "add", "--", contract.CONTINUATION_PREREGISTRATION_PATH)
        _git(root, "commit", "-m", "synthetic continuation")
        invocation_parent = _git(root, "rev-parse", "HEAD")
        invocation_parent_kind = "continuation"
        continuation_sha256 = contract.sha256_bytes(continuation)

    terminal_status = "completed" if completed else "rejected"
    terminal_code = "development_pass" if completed else "synthetic_failure"
    yahoo_count = contract.YAHOO_REQUEST_COUNT if completed else 0
    identity_count = (8 if continued else 4) if completed else 0
    generation_count = contract.DEVELOPMENT_DOCUMENT_COUNT if completed else 0
    total_count = yahoo_count + identity_count + generation_count
    summaries = (
        (
            SimpleNamespace(generation_count=contract.DEVELOPMENT_PILOT_COUNT),
            SimpleNamespace(generation_count=contract.DEVELOPMENT_REMAINING_COUNT),
        )
        if continued
        else (SimpleNamespace(generation_count=contract.DEVELOPMENT_DOCUMENT_COUNT),)
        if completed
        else ()
    )
    snapshot = SimpleNamespace(
        status=terminal_status,
        terminal_code=terminal_code,
        journal_head_sha256="1" * 64,
        request_intent_count=total_count,
        yahoo_intent_count=yahoo_count,
        identity_intent_count=identity_count,
        gemma_intent_count=generation_count,
        response_count=total_count,
        checkpoint_count=total_count,
        yahoo_response_count=yahoo_count,
        yahoo_body_bytes=123 if completed else 0,
        identity_response_count=identity_count,
        gemma_response_count=generation_count,
        segment_summaries=summaries,
        paused=continued,
        continuation_authorized=continued,
        continuation_sha256=continuation_sha256,
        continuation_commit=invocation_parent if continued else None,
        continuation_permission_sha256="5" * 64 if continued else None,
        market_values_opened=completed,
        model_responses_opened=completed,
    )
    effect_report = runner.build_effect_report(snapshot)
    if completed:
        semantic: dict[str, object] = {}
        deterministic: dict[str, object] = {}
        no_leverage: dict[str, object] = {}
        science_summary = {
            "gate_report": {"passed": True, "failed_checks": []},
            "no_leverage_proofs": no_leverage,
            "no_leverage_proofs_sha256": contract.canonical_sha256(no_leverage),
        }
        material_body = {
            "schema_version": runner.PRIVATE_TERMINAL_SCHEMA_VERSION,
            "stage": contract.DEVELOPMENT_COMMAND,
            "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
            "invocation_parent": invocation_parent,
            "invocation_parent_kind": invocation_parent_kind,
            "route": "paused_resumed" if continued else "normal",
            "authority_sha256": contract.canonical_sha256(authority),
            "bridge_manifest": _bridge_manifest(),
            "model_plan_manifest": {},
            "runtime_segment_guards": [{}, {}] if continued else [{}],
            "runtime_aggregate": {},
            "latency_receipt": {},
            "stage_slice_sha256": "4" * 64,
            "semantic_payload": semantic,
            "semantic_payload_sha256": contract.canonical_sha256(semantic),
            "deterministic_payload": deterministic,
            "deterministic_payload_sha256": contract.canonical_sha256(
                deterministic
            ),
            "science_summary": science_summary,
            "science_summary_sha256": contract.canonical_sha256(science_summary),
            "effect_report": effect_report,
            "market_values_opened": True,
            "model_responses_opened": True,
            "confirmation_and_final_opened": False,
            "raw_sec_yahoo_or_gemma_response_copied": False,
        }
        material = {
            **material_body,
            "private_terminal_material_sha256": contract.canonical_sha256(
                material_body
            ),
        }
    else:
        material = runner.build_failure_terminal_material(
            authority=authority,
            terminal_code=terminal_code,
            terminal_status=terminal_status,
            effect_report=effect_report,
            snapshot=SimpleNamespace(
                journal_head_sha256="2" * 64, segment_summaries=()
            ),
            invocation_parent=invocation_parent,
            invocation_parent_kind=invocation_parent_kind,
            market_values_opened=False,
            model_responses_opened=False,
            source_authenticated=False,
        )
    terminal_material = {
        "schema_version": store_module.TERMINAL_CANDIDATE_SCHEMA_VERSION,
        "authority_sha256": contract.canonical_sha256(authority),
        "stage": contract.DEVELOPMENT_COMMAND,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "candidate_sequence": 1,
        "journal_head_before_candidate_sha256": "2" * 64,
        "proposed_status": terminal_status,
        "proposed_terminal_code": terminal_code,
        "evidence": material,
    }
    terminal_material_sha256 = contract.canonical_sha256(terminal_material)
    projection_receipt = SimpleNamespace(
        status=terminal_status,
        terminal_code=terminal_code,
        terminal_material_sha256=terminal_material_sha256,
        evidence=material,
    )
    public = runner.build_public_terminal_artifact(
        authority=authority, terminal_receipt=projection_receipt
    )
    public_bytes = contract.canonical_json_bytes(public)
    comparison_after = runner.build_comparison_update(comparison, public)
    candidate_body = {
        **terminal_material,
        "terminal_material_sha256": terminal_material_sha256,
        "public_result_sha256": contract.sha256_bytes(public_bytes),
        "public_result_bytes": len(public_bytes),
        "comparison_before_sha256": contract.sha256_bytes(comparison),
        "comparison_after_sha256": contract.sha256_bytes(comparison_after),
        "comparison_after_bytes": len(comparison_after),
    }
    candidate_mapping = {
        **candidate_body,
        "candidate_sha256": contract.canonical_sha256(candidate_body),
    }
    candidate_bytes = contract.canonical_json_bytes(candidate_mapping)
    terminal_candidate = SimpleNamespace(
        candidate_sha256=candidate_mapping["candidate_sha256"],
        candidate_bytes=len(candidate_bytes),
        journal_head_before_candidate_sha256="2" * 64,
        proposed_status=terminal_status,
        proposed_terminal_code=terminal_code,
        terminal_material_sha256=terminal_material_sha256,
        public_result_sha256=contract.sha256_bytes(public_bytes),
        public_result_bytes=len(public_bytes),
        comparison_before_sha256=contract.sha256_bytes(comparison),
        comparison_after_sha256=contract.sha256_bytes(comparison_after),
        comparison_after_bytes=len(comparison_after),
        evidence=material,
        candidate=candidate_mapping,
        public_result=public_bytes,
        comparison_before=comparison,
        comparison_after=comparison_after,
    )
    terminal_receipt = SimpleNamespace(
        **terminal_candidate.__dict__,
        event_sha256="1" * 64,
        journal_head_before_terminal_sha256="2" * 64,
        status=terminal_status,
        terminal_code=terminal_code,
    )
    if commit_result:
        result_path = root / contract.RESULT_ARTIFACT_PATH
        result_path.write_bytes(contract.canonical_json_bytes(public))
        comparison_path.write_bytes(comparison_after)
        _git(root, "add", "--", contract.RESULT_ARTIFACT_PATH, contract.COMPARISON_PATH)
        _git(root, "commit", "-m", "synthetic terminal result")
    _git(root, "push", "-u", "origin", contract.BRANCH_NAME)

    v313 = root / contract.PRIVATE_NAMESPACE
    v38 = root / preflight.V38_PRIVATE_ROOT
    v313.mkdir(parents=True)
    v38.mkdir(parents=True)
    (v313 / "synthetic.json").write_bytes(
        contract.canonical_json_bytes(
            {"sealed": True, "bound_private_path": str(v313)}
        )
    )
    (v38 / "synthetic.json").write_bytes(b'{"frozen":true}')
    projection = SimpleNamespace(manifest=_bridge_manifest())
    store = _FakeReadOnlyStore(snapshot, terminal_receipt, terminal_candidate)
    calls: list[str] = []

    def replay_failure(**_kwargs: object) -> dict[str, object]:
        calls.append("failure_replay")
        return copy.deepcopy(material)

    def replay_completed(**_kwargs: object) -> dict[str, object]:
        calls.append("completed_replay")
        return copy.deepcopy(material)

    def rebuild_pilot(**_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        calls.append("pilot_replay")
        return copy.deepcopy(pilot_guard), copy.deepcopy(pilot_latency)

    dependencies = preflight.PushedResultDependencies(
        authenticate_preflight_revision=lambda _root, _commit: receipt_authority,
        load_private_contact=lambda _root: "Synthetic Private Contact",
        authenticate_source=lambda _root, _contact: calls.append("source") or object(),
        build_projection=lambda _source: projection,
        rebuild_attempt_context=lambda _execution, _projection: {
            "authority": authority,
            "plan": plan,
        },
        open_store=lambda _path, _authority: store,
        build_effect_report=runner.build_effect_report,
        build_public_effect_report=runner.build_public_effect_report,
        rebuild_pilot_evidence=rebuild_pilot,
        build_public_pause_artifact=runner.build_public_pause_artifact,
        build_continuation_preregistration=runner.build_continuation_preregistration,
        replay_completed_terminal=replay_completed,
        replay_failure_terminal=replay_failure,
        build_public_terminal_artifact=runner.build_public_terminal_artifact,
        build_comparison_update=runner.build_comparison_update,
        privacy_tokens=lambda path, contact: (
            contact,
            str(path),
            str(path / preflight.V38_PRIVATE_ROOT),
            str(path / contract.PRIVATE_NAMESPACE),
        ),
    )
    return {
        "root": root,
        "dependencies": dependencies,
        "authority": authority,
        "receipt_authority": receipt_authority,
        "snapshot": snapshot,
        "terminal_receipt": terminal_receipt,
        "public": public,
        "material": material,
        "store": store,
        "calls": calls,
        "preflight_commit": preflight_commit,
        "invocation_parent": invocation_parent,
        "comparison_before": comparison,
        "canonical_pause": canonical_pause,
        "committed_pause": committed_pause,
        "continuation": continuation,
    }


def test_pushed_result_gate_replays_failure_and_returns_only_redacted_receipt(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path)
    root = fixture["root"]
    before_head = _git(root, "rev-parse", "HEAD")
    receipt = preflight.authenticate_pushed_result(
        root, dependencies=fixture["dependencies"]
    )
    assert receipt["status"] == "passed"
    assert receipt["development_outcome"] == "rejected"
    assert receipt["route"] == "normal"
    assert receipt["confirmation_preregistration_eligible"] is False
    assert receipt["confirmation_execution_authorized"] is False
    assert receipt["final_execution_authorized"] is False
    assert receipt["real_money_authorized"] is False
    assert receipt["gates"]["failure_material_replayed"] is True
    assert receipt["gates"]["deterministic_science_replayed"] is False
    assert receipt["pushed_result_gate_sha256"] == contract.canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "pushed_result_gate_sha256"
        }
    )
    serialized = contract.canonical_json_bytes(receipt)
    assert b"Synthetic Private Contact" not in serialized
    assert str(root).encode() not in serialized
    assert fixture["calls"] == ["source", "failure_replay"]
    assert fixture["store"].closed is True
    assert _git(root, "rev-parse", "HEAD") == before_head
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""


@pytest.mark.parametrize("secret_kind", ("contact", "v38_path"))
def test_pushed_result_private_tree_allows_v313_path_but_rejects_secrets(
    tmp_path: Path,
    secret_kind: str,
) -> None:
    fixture = _make_result_gate_repo(tmp_path)
    root = fixture["root"]
    injected = (
        "Synthetic Private Contact"
        if secret_kind == "contact"
        else str(root / preflight.V38_PRIVATE_ROOT)
    )
    private_file = root / contract.PRIVATE_NAMESPACE / "synthetic.json"
    private_file.write_bytes(
        contract.canonical_json_bytes(
            {
                "bound_private_path": str(root / contract.PRIVATE_NAMESPACE),
                "forbidden": injected,
            }
        )
    )
    _assert_code(
        "pushed_result_private_privacy_failed",
        preflight.authenticate_pushed_result,
        root,
        dependencies=fixture["dependencies"],
    )


def test_load_execution_authority_reconstructs_clean_pushed_result_without_gate_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_result_gate_repo(tmp_path)
    calls: list[tuple[str, bool]] = []

    def authenticate(
        _root: Path,
        *,
        preflight_commit: str,
        direct_head: bool,
        allow_publication_dirty: bool = False,
    ) -> dict[str, object]:
        calls.append((preflight_commit, direct_head))
        assert allow_publication_dirty is False
        return fixture["receipt_authority"]

    monkeypatch.setattr(preflight, "_authenticate_preflight_revision", authenticate)
    authority = preflight.load_execution_authority(fixture["root"])
    assert authority == fixture["authority"]
    assert calls == [(fixture["preflight_commit"], False)]
    assert "pushed_result_gate_passed" not in authority


def test_pushed_result_cli_is_separate_and_never_invokes_one_shot_preflight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        preflight,
        "run_preflight",
        lambda *_args, **_kwargs: pytest.fail("one-shot preflight was invoked"),
    )
    monkeypatch.setattr(
        preflight,
        "authenticate_pushed_result",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "pushed_result_gate_sha256": "a" * 64,
        },
    )
    assert preflight.main(["--pushed-result", "."]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "passed",
        "pushed_result_gate_sha256": "a" * 64,
    }


def test_pushed_result_gate_rejects_independent_replay_mismatch(tmp_path: Path) -> None:
    fixture = _make_result_gate_repo(tmp_path)
    dependencies = fixture["dependencies"]
    mismatched = preflight.PushedResultDependencies(
        **{
            **dependencies.__dict__,
            "replay_failure_terminal": lambda **_kwargs: {
                "private_terminal_material_sha256": "0" * 64
            },
        }
    )
    _assert_code(
        "pushed_result_independent_replay_failed",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=mismatched,
    )


def test_pushed_result_gate_requires_completed_science_and_exact_effect_replay(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, completed=True)
    receipt = preflight.authenticate_pushed_result(
        fixture["root"], dependencies=fixture["dependencies"]
    )
    assert receipt["development_outcome"] == "passed"
    assert receipt["confirmation_preregistration_eligible"] is True
    assert receipt["gates"]["deterministic_science_replayed"] is True
    assert receipt["gates"]["failure_material_replayed"] is False
    assert receipt["science_summary_sha256"] == contract.canonical_sha256(
        fixture["material"]["science_summary"]
    )
    assert fixture["calls"] == ["source", "completed_replay"]


def test_pushed_result_gate_reconstructs_f_s_c_and_binds_exact_continuation_commit(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, completed=True, continued=True)
    receipt = preflight.authenticate_pushed_result(
        fixture["root"], dependencies=fixture["dependencies"]
    )
    assert receipt["route"] == "paused_resumed"
    assert receipt["authorized_parent_kind"] == "continuation"
    assert receipt["authorized_parent"] == fixture["invocation_parent"]
    assert receipt["development_outcome"] == "passed"
    assert fixture["committed_pause"] == fixture["canonical_pause"]
    assert fixture["continuation"] == runner.build_continuation_preregistration(
        fixture["canonical_pause"]
    )
    assert fixture["calls"] == ["source", "pilot_replay", "completed_replay"]


def test_pushed_result_gate_rejects_same_continuation_bytes_under_other_commit(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, completed=True, continued=True)
    assert fixture["snapshot"].continuation_sha256 == contract.sha256_bytes(
        fixture["continuation"]
    )
    assert fixture["snapshot"].continuation_commit == fixture["invocation_parent"]
    fixture["snapshot"].continuation_commit = "7" * 40
    _assert_code(
        "pushed_result_continuation_binding_invalid",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=fixture["dependencies"],
    )


def test_pushed_result_gate_rejects_self_hashed_pause_mutation(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(
        tmp_path, completed=True, continued=True, mutated_pause=True
    )
    committed_pause = fixture["committed_pause"]
    assert contract.validate_self_sha256(
        committed_pause, field="pause_artifact_sha256"
    ) == committed_pause
    assert committed_pause["implementation_commit"] == "f" * 40
    assert committed_pause != fixture["canonical_pause"]
    _assert_code(
        "pushed_result_pause_artifact_invalid",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=fixture["dependencies"],
    )


def test_pushed_result_gate_rejects_arbitrary_hash_bound_continuation_document(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(
        tmp_path, completed=True, continued=True, arbitrary_continuation=True
    )
    arbitrary = fixture["continuation"]
    assert fixture["snapshot"].continuation_sha256 == contract.sha256_bytes(
        arbitrary
    )
    assert arbitrary != runner.build_continuation_preregistration(
        fixture["canonical_pause"]
    )
    _assert_code(
        "pushed_result_continuation_invalid",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=fixture["dependencies"],
    )


def test_publication_recovery_accepts_only_exact_idempotent_partial_bytes(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, commit_result=False)
    root = fixture["root"]
    # Reproduce the exact two public files as an interrupted, uncommitted
    # publication while both local and remote authority remain exactly F.
    public = fixture["public"]
    result_path = root / contract.RESULT_ARTIFACT_PATH
    result_bytes = contract.canonical_json_bytes(public)
    result_pending = root / f"{contract.RESULT_ARTIFACT_PATH}.v313-pending"
    result_pending.write_bytes(result_bytes)
    comparison_path = root / contract.COMPARISON_PATH
    derived = runner.build_comparison_update(fixture["comparison_before"], public)
    pending = root / f"{contract.COMPARISON_PATH}.v313-pending"
    pending.write_bytes(derived)

    recovery = preflight.PublicationRecoveryDependencies(
        authenticate_preflight_revision=lambda _root, _commit: fixture[
            "receipt_authority"
        ],
        open_store=lambda _path, _authority: _FakeReadOnlyStore(
            fixture["snapshot"], fixture["terminal_receipt"]
        ),
        build_public_terminal_artifact=runner.build_public_terminal_artifact,
        build_comparison_update=runner.build_comparison_update,
    )
    assert preflight.load_publication_recovery_authority(
        root, dependencies=recovery
    ) == fixture["authority"]

    pending.write_bytes(b"tampered")
    _assert_code(
        "publication_recovery_pending_invalid",
        preflight.load_publication_recovery_authority,
        root,
        dependencies=recovery,
    )


def test_publication_recovery_accepts_only_exact_pause_pending_bytes(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, commit_result=False)
    root = fixture["root"]
    pause_body = {
        "status": "paused_for_justification",
        "stage": contract.DEVELOPMENT_COMMAND,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "branch": contract.BRANCH_NAME,
        "preflight_commit": fixture["preflight_commit"],
        "privacy_passed": True,
    }
    pause = {
        **pause_body,
        "pause_artifact_sha256": contract.canonical_sha256(pause_body),
    }
    pause_bytes = contract.canonical_json_bytes(pause)
    pause_path = root / contract.PAUSE_ARTIFACT_PATH
    pause_pending = root / f"{contract.PAUSE_ARTIFACT_PATH}.v313-pending"
    pause_pending.write_bytes(pause_bytes)
    pause_snapshot = SimpleNamespace(
        status="paused", paused=True, continuation_authorized=False
    )
    recovery = preflight.PublicationRecoveryDependencies(
        authenticate_preflight_revision=lambda _root, _commit: fixture[
            "receipt_authority"
        ],
        open_store=lambda _path, _authority: _FakeReadOnlyStore(
            pause_snapshot, None
        ),
        build_public_terminal_artifact=runner.build_public_terminal_artifact,
        build_comparison_update=runner.build_comparison_update,
        rebuild_public_pause_artifact=lambda **_kwargs: pause,
    )
    assert preflight.load_publication_recovery_authority(
        root, dependencies=recovery
    ) == fixture["authority"]

    pause_pending.write_bytes(b"truncated")
    _assert_code(
        "publication_recovery_pause_invalid",
        preflight.load_publication_recovery_authority,
        root,
        dependencies=recovery,
    )
