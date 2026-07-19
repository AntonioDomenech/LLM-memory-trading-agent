from __future__ import annotations

import copy
import builtins
import hashlib
import inspect
import json
import socket
import subprocess
import time
import urllib.request

import pytest

from agent_benchmark import sec_gemma_lean_science_v319_contract as contract
from agent_benchmark.sec_filing_gemma_contract import (
    CANONICAL_IDENTITY_LEXICON,
    build_extractor_model_payload,
)
from agent_benchmark.sec_filing_gemma_preprocessor import preprocess_filing_event


SHA1_A = "1" * 40
SHA1_B = "2" * 40
SHA1_C = "3" * 40
SHA1_D = "4" * 40
SHA1_E = "5" * 40


def _assert_code(code: str, function, *args, **kwargs) -> None:
    with pytest.raises(contract.ContractViolation) as captured:
        function(*args, **kwargs)
    assert captured.value.code == code
    assert str(captured.value) == code


def test_canonical_json_and_hash_helpers_are_exact_and_redacted() -> None:
    value = {"z": [1, True, None], "a": "café"}
    expected = b'{"a":"caf\xc3\xa9","z":[1,true,null]}'
    assert contract.canonical_json_bytes(value) == expected
    assert contract.canonical_sha256(value) == contract.sha256_bytes(expected)
    assert contract.sha256_bytes(memoryview(expected)) == contract.sha256_bytes(
        expected
    )
    _assert_code(
        "v319_contract_json_nonfinite", contract.canonical_json_bytes, {"x": float("nan")}
    )
    _assert_code(
        "v319_contract_json_key", contract.canonical_json_bytes, {1: "not allowed"}
    )
    _assert_code(
        "v319_contract_json_type", contract.canonical_json_bytes, {"x": object()}
    )
    _assert_code("v319_contract_hash_input", contract.sha256_bytes, "bytes required")


def test_generic_self_hash_is_strict_and_detached() -> None:
    source = {"schema_version": "test-v1", "items": [1, 2]}
    sealed = contract.add_self_sha256(source)
    assert source == {"schema_version": "test-v1", "items": [1, 2]}
    assert sealed["manifest_sha256"] == contract.canonical_sha256(source)
    assert contract.validate_self_sha256(sealed) == sealed

    detached = contract.validate_self_sha256(sealed)
    detached["items"].append(3)
    assert contract.validate_self_sha256(sealed) == sealed

    changed = copy.deepcopy(sealed)
    changed["items"].append(3)
    _assert_code("v319_contract_self_hash_mismatch", contract.validate_self_sha256, changed)
    _assert_code("v319_contract_self_hash_present", contract.add_self_sha256, sealed)


def test_tagged_sha256_transform_accepts_only_one_exact_lowercase_prefix() -> None:
    digest = "a" * 64
    assert contract.strip_exact_sha256_tag(f"sha256:{digest}") == digest
    for invalid in (
        digest,
        f"SHA256:{digest}",
        f"sha256:SHA256:{digest}",
        f"sha256:{digest.upper()}",
        f"sha256:{digest}0",
        None,
    ):
        _assert_code(
            "v319_contract_tagged_sha256",
            contract.strip_exact_sha256_tag,
            invalid,
        )


def test_frozen_science_projection_is_exact_38320_bytes() -> None:
    projection = contract.build_frozen_science_projection()
    payload = contract.canonical_json_bytes(projection)
    assert tuple(projection) == contract.SCIENCE_PROJECTION_KEYS
    assert len(payload) == contract.SCIENCE_PROJECTION_BYTE_COUNT == 38_320
    assert contract.sha256_bytes(payload) == contract.SCIENCE_PROJECTION_SHA256
    assert (
        contract.FROZEN_SCIENCE_PROJECTION_SHA256
        == "1ee05d2916752752bbef3710d70c7dab664fb82b9ac22829dac8897401058609"
    )
    assert len(projection["gates"]["development"]) == 22
    assert projection["gates"]["development"] == {
        "annual_win_rate_10bps_at_least": 0.55,
        "combined_edge_without_best_block_10bps_at_least": 0.005,
        "combined_total_active_log_edge_10bps_at_least": 0.02,
        "complete_sec_overlay_episodes_at_least": 12,
        "incremental_vs_baseline_10bps_at_least": 0.005,
        "incremental_vs_baseline_without_best_block_10bps_strictly_positive": True,
        "largest_positive_episode_share_10bps_at_most": 0.35,
        "negative_aapl_year_win_rate_10bps_at_least": 0.60,
        "nonzero_filing_meaning_rows_at_least": 24,
        "online_vs_block_frozen_10bps_edge_strictly_positive": True,
        "online_vs_block_frozen_action_differences_at_least": 5,
        "online_vs_block_frozen_difference_blocks_at_least": 3,
        "overlay_episode_win_rate_10bps_at_least": 0.55,
        "overlay_median_edge_10bps_strictly_positive": True,
        "positive_combined_blocks_10bps_at_least": 4,
        "schema_valid_extraction_rate_at_least": 0.90,
        "semantic_brier_relative_improvement_at_least": 0.01,
        "semantic_edge_without_best_xor_10bps_strictly_positive": True,
        "semantic_vs_no_filing_meaning_10bps_edge_at_least": 0.005,
        "semantic_vs_no_filing_meaning_action_differences_at_least": 5,
        "semantic_vs_no_filing_meaning_complete_xor_intervals_at_least": 4,
        "semantic_vs_no_filing_meaning_difference_blocks_at_least": 3,
    }
    assert contract.verify_frozen_science_projection(projection) == projection

    changed = copy.deepcopy(projection)
    changed["policy"]["baseline_parameters"]["contextual_percentile"] = 0.8
    _assert_code(
        "v319_contract_science_projection_hash",
        contract.verify_frozen_science_projection,
        changed,
    )
    missing = copy.deepcopy(projection)
    missing.pop("ledger")
    _assert_code(
        "v319_contract_science_projection_keys",
        contract.verify_frozen_science_projection,
        missing,
    )
    extra = {**projection, "operational_plumbing": {}}
    _assert_code(
        "v319_contract_science_projection_keys",
        contract.verify_frozen_science_projection,
        extra,
    )


def test_source_authority_contains_every_preregistered_pin() -> None:
    pins = contract.build_source_authority_pins()
    assert pins["base"] == {
        "commit": "0c4b01cf5f1ef77548658d9bfa38e76fb1b70635",
        "tree": "68a9176ed22edeb2edc0f00ae1179c8724603539",
        "parent": "a0f971184ad26630478182be97f01c46c80498e1",
    }
    assert pins["terminal"]["git_blob_sha1"] == (
        "28075db4e7cb85bfcdeb92e7b1aecce28f3a49fa"
    )
    assert pins["terminal"]["literal_sha256"] == (
        "c176c6fb5e200dd363be656562d72bc40f722912221fe7f5cd52d8e42cc9723a"
    )
    assert pins["terminal"]["internal_sha256"] == (
        "d72059f39eb4b1019ce83799682eadc7eecf3fd77310f43bee8537433a20cb20"
    )
    assert pins["source_authority"]["literal_sha256"] == (
        "d7f87ac22e58ac4b50033af95e6f768692f1e31a6890e9f50e49f20d59ff0d0a"
    )
    aggregates = pins["private_aggregates"]
    assert aggregates["role_counts"] == {"I": 97, "U": 75, "D": 75}
    assert aggregates["v38_sec_requests"] == 199
    assert aggregates["experiment_family_sec_requests"] == 964
    assert aggregates["inventory_file_count"] == 1_410
    assert aggregates["inventory_byte_count"] == 612_601_642
    assert aggregates["inventory_sha256"] == (
        "850f3a022fcaca5b56156d7843f35b05d3bc5732bf480634bd9872bfac7194f2"
    )

    pins["base"]["commit"] = "tampered"
    assert (
        contract.build_source_authority_pins()["base"]["commit"]
        == contract.V38_SOURCE_COMMIT
    )


def test_git_authority_families_preserve_exact_successor_topology() -> None:
    assert (contract.BASE_COMMIT, contract.BASE_TREE, contract.BASE_PARENT) == (
        "503826b5ff0f14b0f1c1a11432bdbd7f41712fc2",
        "ffa89ed5b2fcd41e4930fa9854ebf25b99308d6b",
        "f0080b056634866955e7ee73845d509f2bfcf0f2",
    )
    assert (
        contract.PREREGISTRATION_COMMIT,
        contract.PREREGISTRATION_TREE,
        contract.PREREGISTRATION_GIT_BLOB_SHA1,
        contract.PREREGISTRATION_LITERAL_SHA256,
        contract.PREREGISTRATION_LITERAL_BYTES,
    ) == (
        "299a1458f2021be4ea8e38dbb2048b80fac64a18",
        "163a88727f45082d69a0c7b80b134e585bb9dbce",
        "bb206dbeafe958b42e35dcfcc5798a2e4154b4f1",
        "5ff7f4070008d2dbb5dce40e48e309282803f7ef40c0b62125eebe5dd30e5fe7",
        24_009,
    )
    assert contract.PREREGISTRATION_PATH == (
        "docs/aapl_sec_gemma_lean_science_v3_19.md"
    )
    predecessor = contract.build_v316_rejection_pins()
    assert predecessor["preregistration"] == {
        "commit": "2cb4abf35b13502953d4f0b502ab6fc5c638be58",
        "tree": "bac40934de7bd299590ac18ffd6a912e125b356f",
        "path": "docs/aapl_sec_gemma_lean_science_v3_16.md",
        "git_blob_sha1": "fac61696f74b15108d43ff0518ec7d3fbb5dce4c",
        "literal_sha256": (
            "e4609a7063784323e94637008a2d034c5904f3f2acf113d629f684ac147c93ba"
        ),
        "literal_bytes": 30_557,
    }
    assert predecessor["implementation"] == {
        "commit": "14eea6f536e1fc93ea54dbe8fa6b436a3f462cf4",
        "tree": "8b0b6783aeeaa5d6ee7ec0264f79e29b2bc71ca5",
        "parent": predecessor["preregistration"]["commit"],
    }
    assert predecessor["preflight_failure"] == {
        "commit": "475084cb48ef081a3b10ac35a554d5e8ca408f7b",
        "tree": "910fa694aede2b27167af6adbbaa2e9c2522bf2c",
        "parent": predecessor["implementation"]["commit"],
        "path": (
            "e/aapl_sec_gemma_lean_science_v3_16/DEVELOPMENT_PREFLIGHT.json"
        ),
        "git_blob_sha1": "c58a04277da9b35581de08086bb0c75df927595c",
        "literal_sha256": (
            "8485edee1943b54f97ff263db0f6318ea8767b79eee12ae3e096dd5316a5d3fc"
        ),
        "literal_bytes": 622,
        "internal_sha256": (
            "e770a83bfcf4ebb2afa477e63e67812fe6d1480c08ce258bbf0b5e1c73e5e94e"
        ),
        "failure_code": "nullable_model_request_invalid",
        "preflight_consumed": True,
        "rerun_authorized": False,
        "development_authorized": False,
        "confirmation_and_final_opened": False,
        "real_money_authorized": False,
    }
    assert predecessor["rejection"]["commit"] == contract.V316_REJECTION_COMMIT
    assert predecessor["rejection"]["tree"] == contract.V316_REJECTION_TREE
    assert predecessor["rejection"]["parent"] == (
        contract.V316_PREFLIGHT_FAILURE_COMMIT
    )
    assert predecessor["rejection"]["path"] == "e/APPROACH_COMPARISON.md"
    assert predecessor["rejection"]["git_blob_sha1"] == (
        "bbe16d8fe516ed1d86dd5d2af027afd80145b595"
    )
    assert predecessor["rejection"]["literal_sha256"] == (
        "b34df02d6d9eeadaaf77133a966463cc8418266631d11e970ef04145611b9250"
    )
    assert predecessor["rejection"]["literal_bytes"] == 35_283
    assert predecessor["phase1_node_count"] == 325
    assert predecessor["phase1_case_sensitive_unique_node_count"] == 325
    assert predecessor["phase1_duplicate_node_count"] == 0
    assert predecessor["phase1_required_node_count"] == 24
    assert predecessor["official_qualification_run_count"] == 1
    assert predecessor["preflight_consumed"] is True
    assert predecessor["external_effect_count"] == 0
    assert predecessor["execution_authority"] is False

    document_only = contract.build_v317_rejection_pins()
    assert document_only["preregistration"] == {
        "commit": "6df6e07a1bfb5b8cc41b2ae5f793ff43df05599a",
        "tree": "357a9f04b527c6fa13d83faf1d6c89afaae51c66",
        "parent": contract.V316_REJECTION_COMMIT,
        "path": "docs/aapl_sec_gemma_lean_science_v3_17.md",
        "git_blob_sha1": "fc9509edcba502169312d59f31f05a038b4f2d16",
        "literal_sha256": (
            "3b7272fc1b4db83f053f534375df8ff1752fa878933dd23cd8718d8b41b26b9d"
        ),
        "literal_bytes": 21_021,
    }
    assert document_only["rejection"] == {
        "commit": "4a5bedd72c1e718b46c80b357ecc83561771e837",
        "tree": "f084829c18826a6d37e266de3847bc48a312eb1e",
        "parent": "6df6e07a1bfb5b8cc41b2ae5f793ff43df05599a",
        "changed_paths": {
            "docs/aapl_sec_gemma_lean_science_v3_17_rejection.md": "A",
            "e/APPROACH_COMPARISON.md": "M",
        },
        "document": {
            "path": "docs/aapl_sec_gemma_lean_science_v3_17_rejection.md",
            "git_blob_sha1": "41f8eacfbd857cf297155c79f34db475168afcf0",
            "literal_sha256": (
                "f8b07cb6dced73b6c4a30388264798503ccb1f058644133a6b1b3aa4bcfde42b"
            ),
            "literal_bytes": 2_601,
        },
        "comparison": {
            "path": "e/APPROACH_COMPARISON.md",
            "git_blob_sha1": "58aac6a63b9e33d777d94f7832afbb109c20f969",
            "literal_sha256": (
                "594dbaa7b6f4f76bcb1fea96033276fb21e5e17580265b6dd4ea4ecd5c18e03f"
            ),
            "literal_bytes": 36_142,
        },
    }
    assert document_only["implementation_exists"] is False
    assert document_only["preflight_artifact_exists"] is False
    assert document_only["private_authority_exists"] is False
    assert document_only["official_preflight_run_count"] == 0
    assert document_only["external_effect_count"] == 0
    assert document_only["execution_authority"] is False

    v318 = contract.build_v318_rejection_pins()
    assert v318["preregistration"] == {
        "commit": "f0080b056634866955e7ee73845d509f2bfcf0f2",
        "tree": "999181ff11f60596dbeeb17466e9c660472252b9",
        "parent": "4a5bedd72c1e718b46c80b357ecc83561771e837",
        "path": "docs/aapl_sec_gemma_lean_science_v3_18.md",
        "git_blob_sha1": "507746c8a9a6c07093fe0aee4a19ab1a5e0e74f5",
        "literal_sha256": (
            "1873ab7a0b42c0b347344d4c7c58d252f5f29ab4570018b6cfa138274fe03de9"
        ),
        "literal_bytes": 24_821,
    }
    assert v318["rejection"]["commit"] == contract.BASE_COMMIT
    assert v318["rejection"]["tree"] == contract.BASE_TREE
    assert v318["rejection"]["parent"] == contract.BASE_PARENT
    assert v318["rejection"]["document"] == {
        "path": "docs/aapl_sec_gemma_lean_science_v3_18_rejection.md",
        "git_blob_sha1": "ed59c67a9cdcab165cc1440716dd15fbd43e617b",
        "literal_sha256": (
            "21bff88a7da5faa0e9907c2908884a46dd28998db77365e1013270ae8a1e6e95"
        ),
        "literal_bytes": 3_740,
    }
    assert v318["rejection"]["comparison"] == {
        "path": "e/APPROACH_COMPARISON.md",
        "git_blob_sha1": "6aac9a11428447266a8d64c4c68ca6b626cd3725",
        "literal_sha256": (
            "769dec11a0c12f1a454cd0a99a71c63f3addc12e49b222b6201e5a4eb90c21cb"
        ),
        "literal_bytes": 37_132,
    }
    assert v318["implementation_exists"] is False
    assert v318["preflight_artifact_exists"] is False
    assert v318["private_authority_exists"] is False
    assert v318["official_preflight_run_count"] == 0
    assert v318["external_effect_count"] == 0
    assert v318["execution_authority"] is False
    assert (
        contract.V38_SOURCE_COMMIT,
        contract.V38_SOURCE_TREE,
        contract.V38_SOURCE_PARENT,
    ) == (
        "0c4b01cf5f1ef77548658d9bfa38e76fb1b70635",
        "68a9176ed22edeb2edc0f00ae1179c8724603539",
        "a0f971184ad26630478182be97f01c46c80498e1",
    )
    inherited = contract.build_v39_inherited_preregistration_pins()
    assert inherited == {
        "commit": "d50c33515ed9597b2fc07bb40c922a3f3166fbdd",
        "tree": "946a009fb9e479778ea1881bae302d802578248b",
        "parent": "0c4b01cf5f1ef77548658d9bfa38e76fb1b70635",
        "path": "docs/aapl_sec_gemma_lean_science_v3_9.md",
        "git_blob_sha1": "2b7042fd948cad91925c9fe9e01784ea0ff739c8",
        "literal_sha256": (
            "feb9bd040e0ed65aed125e52e79e10bc8f87cd7e96776dc67353502fc0983446"
        ),
        "literal_bytes": 43_098,
        "execution_authority": False,
    }
    assert len(
        {
            contract.BASE_COMMIT,
            contract.PREREGISTRATION_COMMIT,
            contract.V318_PREREGISTRATION_COMMIT,
            contract.V317_REJECTION_COMMIT,
            contract.V317_PREREGISTRATION_COMMIT,
            contract.V316_PREREGISTRATION_COMMIT,
            contract.V316_IMPLEMENTATION_COMMIT,
            contract.V316_PREFLIGHT_FAILURE_COMMIT,
            contract.V316_REJECTION_COMMIT,
            contract.V38_SOURCE_COMMIT,
            contract.V39_INHERITED_PREREG_COMMIT,
        }
    ) == 11


def test_model_yahoo_paths_and_orders_are_frozen() -> None:
    assert contract.MODEL_NAME == "gemma4:12b"
    assert contract.MODEL_MANIFEST_SHA256 == (
        "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
    )
    assert contract.RUNTIME_FINGERPRINT_SHA256 == (
        "816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77"
    )
    assert contract.PROMPT_SHA256 == (
        "9ed8496ed101c138cdbee162bdf6dfd53434f6f1e0c64fc93d844405d6eae9f7"
    )
    assert contract.SCHEMA_SHA256 == (
        "1707ae581abb1a256dfb1ee8f51efd9dd67df5d9b3e8f7c22ee2d92b67b6f82b"
    )
    assert len(contract.MODEL_LAYER_DIGESTS) == 4
    assert contract.MODEL_ACTIVE_FROM_BLOB_COUNT == 2
    assert contract.RUNTIME_PROBE_ORDER == (
        ("GET", "http://127.0.0.1:11434/api/version"),
        ("POST", "http://127.0.0.1:11434/api/show"),
    )
    assert contract.YAHOO_SYMBOL_ORDER == ("AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX")
    assert contract.YAHOO_QUERY_ITEMS == (
        ("period1", "883612800"),
        ("period2", "1546300800"),
        ("interval", "1d"),
        ("includePrePost", "false"),
        ("includeAdjustedClose", "true"),
        ("events", "div,splits"),
    )
    assert len(contract.YAHOO_URLS) == 6
    assert contract.YAHOO_URLS[-1].startswith(
        "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?period1=883612800"
    )
    assert all(url.endswith("events=div%2Csplits") for url in contract.YAHOO_URLS)
    assert contract.SOURCE_PROJECTION_ORDER == ("availability_session", "accession")
    assert contract.SCIENCE_EVENT_ORDER == (
        "availability_session",
        "acceptance_datetime",
        "accession",
    )


def test_effect_budgets_cover_normal_pause_resume_and_preflight() -> None:
    budgets = contract.build_effect_budgets()
    assert set(budgets) == {
        "zero_effect_preflight",
        "normal_complete",
        "pilot_pause",
        "paused_resumed_complete",
    }
    for budget in budgets.values():
        assert tuple(budget) == contract.EFFECT_COUNT_KEYS
        assert budget["sec_requests"] == 0
        assert budget["experiment_family_sec_requests"] == 964
        assert budget["retries"] == budget["paid_calls"] == 0
        assert budget["confirmation_final_data_opens"] == 0
        assert budget["broker_effects"] == budget["real_money_effects"] == 0
    assert budgets["zero_effect_preflight"]["yahoo_requests"] == 0
    assert budgets["normal_complete"]["yahoo_requests"] == 6
    assert budgets["normal_complete"]["ollama_identity_http_requests"] == 4
    assert budgets["normal_complete"]["ollama_chat_generations"] == 75
    assert budgets["pilot_pause"]["ollama_identity_http_requests"] == 4
    assert budgets["pilot_pause"]["ollama_chat_generations"] == 5
    assert budgets["paused_resumed_complete"]["ollama_identity_http_requests"] == 8
    assert budgets["paused_resumed_complete"]["ollama_chat_generations"] == 75

    for route, budget in budgets.items():
        assert contract.validate_effect_counts(budget, route=route) == budget
    changed = copy.deepcopy(budgets["normal_complete"])
    changed["yahoo_requests"] = 5
    _assert_code(
        "v319_contract_effect_counts",
        contract.validate_effect_counts,
        changed,
        route="normal_complete",
    )
    _assert_code(
        "v319_contract_effect_route",
        contract.validate_effect_counts,
        changed,
        route="best_result",
    )


def test_pilot_projection_has_strict_greater_than_boundary() -> None:
    exact_duration = 576_000_000_000
    exact = [exact_duration] * 5
    assert contract.projected_pilot_ns(exact) == 43_200_000_000_000
    assert contract.pilot_pause_required(exact) is False
    above = [exact_duration + 1, *([exact_duration] * 4)]
    assert contract.projected_pilot_ns(above) == 43_200_000_000_071
    assert contract.pilot_pause_required(above) is True
    _assert_code("v319_contract_pilot_durations", contract.projected_pilot_ns, exact[:4])
    _assert_code(
        "v319_contract_pilot_durations",
        contract.projected_pilot_ns,
        [exact_duration, exact_duration, exact_duration, exact_duration, 0],
    )


def test_only_literal_development_command_is_accepted() -> None:
    assert contract.validate_command("development") == "development"
    for rejected in ("confirmation", "final", "development ", None, b"development"):
        _assert_code("v319_contract_command", contract.validate_command, rejected)


def test_local_production_closure_and_import_upper_bound_are_exact() -> None:
    assert len(contract.LOCAL_PRODUCTION_CLOSURE_PATHS) == 37
    assert contract.LOCAL_PRODUCTION_CLOSURE_PATHS == tuple(
        sorted(contract.LOCAL_PRODUCTION_CLOSURE_PATHS, key=lambda path: path.encode())
    )
    assert len(contract.LOCAL_PRODUCTION_CLOSURE_ORDERED_ROOTS) == 13
    assert contract.LOCAL_PRODUCTION_CLOSURE_EXPLICIT_DYNAMIC_PATHS == (
        "agent_benchmark/sec_filing_gemma_learner.py",
        "agent_benchmark/sec_gemma_online_risk_overlay_production.py",
    )
    assert contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_BYTES == 8_756
    assert contract.LOCAL_PRODUCTION_CLOSURE_MANIFEST_SHA256 == (
        "863a293ee604e855d480230068f9e53b609aed36e1af2d2d316b8b87bdeedf2a"
    )
    assert contract.LOCAL_PRODUCTION_CLOSURE_PATHS_BYTES == 2_023
    assert contract.LOCAL_PRODUCTION_CLOSURE_PATHS_SHA256 == (
        "0a1e7374b35677596418cfa82575bd6614df631eba34f1978ce9c9017a74aae9"
    )
    assert contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH == (
        "agent_benchmark/__init__.py"
    )
    assert len(contract.LOCAL_IMPORT_ALLOWED_PATHS) == 44
    assert not any(path.startswith("tests/") for path in contract.LOCAL_IMPORT_ALLOWED_PATHS)
    observed = (
        contract.IMPLEMENTATION_PRODUCTION_PATHS[0],
        contract.LOCAL_PRODUCTION_CLOSURE_PATHS[0],
        contract.LOCAL_IMPORT_PACKAGE_BOOTSTRAP_PATH,
    )
    assert contract.validate_local_import_paths(observed) == observed
    _assert_code(
        "v319_contract_local_import_duplicate",
        contract.validate_local_import_paths,
        (observed[0], observed[0]),
    )
    _assert_code(
        "v319_contract_local_import_outside_closure",
        contract.validate_local_import_paths,
        ("agent_benchmark/unpreregistered_strategy.py",),
    )
    _assert_code(
        "v319_contract_local_import_shape",
        contract.validate_local_import_paths,
        "agent_benchmark/sec_filing_content.py",
    )


def test_qualification_scope_schemas_runtime_and_counts_are_frozen() -> None:
    qualification = contract.build_qualification_contract()
    assert qualification["suite_id"] == (
        "latest_approach_actual_dependencies_pinned_parent_authority"
    )
    assert qualification["phases"] == ["v319", "dependencies", "requests_identity"]
    assert qualification["modes"] == ["collection", "execution"]
    assert qualification["scientific_environment"] == {
        "names": [
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "PATH",
            "PYTHONNOUSERSITE",
            "PYTHONHASHSEED",
            "PYTHONPATH",
            "PYTHONUTF8",
            "PYTHONIOENCODING",
            "TZ",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ],
        "fixed_values": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "TZ": "America/New_York",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }
    assert qualification["qualification_environment"]["names"] == [
        *qualification["scientific_environment"]["names"],
        "PYTHONDONTWRITEBYTECODE",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    ]
    assert len(qualification["scientific_environment"]["names"]) == 14
    assert len(qualification["qualification_environment"]["names"]) == 16
    assert qualification["qualification_environment"]["launch_names"] == (
        qualification["qualification_environment"]["names"]
    )
    assert qualification["qualification_environment"][
        "controlled_pytest_additions"
    ] == [
        {"name": "USERPROFILE", "authority": "canonical_TEMP"},
        {"name": "LOCALAPPDATA", "authority": "canonical_TEMP"},
        {"name": "APPDATA", "authority": "canonical_TEMP"},
        {"name": "HOME", "authority": "canonical_TEMP"},
        {
            "name": "COMSPEC",
            "authority": "canonical_SYSTEMROOT/System32/cmd.exe",
        },
        {"name": "PATHEXT", "authority": ".COM;.EXE;.BAT;.CMD"},
    ]
    assert qualification["qualification_environment"]["pytest_names"] == list(
        contract.QUALIFICATION_PYTEST_ENVIRONMENT_NAMES
    )
    assert len(qualification["qualification_environment"]["pytest_names"]) == 22
    assert qualification["qualification_environment"]["fixed_values"] == {
        **qualification["scientific_environment"]["fixed_values"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    assert qualification["launcher"] == {
        "python_flags": ["-s", "-S", "-B"],
        "bootstrap_bytes": 5_482,
        "bootstrap_sha256": (
            "59aa7f29b7200e5a915b15a840da050e7803c825bf40ad42720dc0a4092e4407"
        ),
        "dispatch_prefix": [
            "pytest",
            "--repo-root",
            "{canonical_absolute_root}",
            "--",
        ],
    }
    bootstrap_bytes = contract.QUALIFICATION_BOOTSTRAP_LITERAL.encode("utf-8")
    assert len(bootstrap_bytes) == contract.QUALIFICATION_BOOTSTRAP_BYTES == 5_482
    assert hashlib.sha256(bootstrap_bytes).hexdigest() == (
        contract.QUALIFICATION_BOOTSTRAP_SHA256
    )
    assert not contract.QUALIFICATION_BOOTSTRAP_LITERAL.startswith("\n")
    assert not contract.QUALIFICATION_BOOTSTRAP_LITERAL.endswith("\n")
    assert qualification["pytest_args"] == ["-q", "-p", "no:cacheprovider"]
    assert qualification["latest"] == {
        "phase": "v319",
        "test_paths": list(contract.IMPLEMENTATION_TEST_PATHS),
        "minimum_node_count": 333,
    }
    dependencies = qualification["dependencies"]
    assert len(dependencies["test_paths"]) == 14
    assert dependencies["node_count"] == 655
    assert dependencies["case_sensitive_unique_node_count"] == 655
    assert dependencies["duplicate_node_count"] == 0
    assert dependencies["multiplicity_policy"] == (
        "python_unicode_case_sensitive_collections_counter_v1"
    )
    assert dependencies["node_list_sha256"] == (
        "1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c"
    )
    sentinel = qualification["requests_identity"]
    assert sentinel["node_count"] == 1
    assert sentinel["node_id"].endswith(
        "::test_runtime_modules_share_exact_verified_requests_identity"
    )
    assert sentinel["node_list_sha256"] == (
        "465bbb7fb1bd0633502006db2b84f6adab6ca7542d8c4e3d75b13d6cb7e73229"
    )
    assert qualification["runtime"] == {
        "python_version": (
            "3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) "
            "[MSC v.1937 64 bit (AMD64)]"
        ),
        "python_cache_tag": "cpython-312",
        "os_name": "nt",
        "sys_platform": "win32",
        "executable_basename": "python.exe",
        "executable_bytes": 103_192,
        "executable_sha256": (
            "624bbc0586d8855633b875e911883bbef8a0e8b8711e11126df480dd86f54181"
        ),
        "pytest_version": "9.0.3",
        "pytest_init_bytes": 5_582,
        "pytest_init_sha256": (
            "7be7a1e2218dc59a19d1ad131e4abe21172a295087efc72898938248782e8766"
        ),
    }
    assert set(qualification["schemas"]) == {
        "collection",
        "intent",
        "result",
        "private_aggregate",
        "public_receipt",
    }
    assert qualification["timeout_seconds"] == 1_500
    assert qualification["durability_mode"] == (
        "windows_file_fsync_rename_noreplace_marker_last_v1"
    )

    qualification["dependencies"]["test_paths"].append("tests/forbidden.py")
    assert len(contract.build_qualification_contract()["dependencies"]["test_paths"]) == 14


def test_contract_manifest_is_strict_self_hashed_and_detached() -> None:
    manifest = contract.build_contract_manifest()
    assert manifest["contract_manifest_sha256"] == (
        "fd77730b9483ccdff930a8b398aa71a4a3fabc77fee9a26dcb6c8cb50b9331d7"
    )
    assert manifest["science_authority"]["projection_bytes"] == 38_320
    assert manifest["science_authority"]["development_gate_count"] == 22
    assert manifest["implementation"]["all_paths"] == list(
        contract.IMPLEMENTATION_ALLOWED_PATHS
    )
    assert len(manifest["implementation"]["all_paths"]) == 12
    assert manifest["git_authority"]["base_commit"] == contract.BASE_COMMIT
    assert manifest["source_authority"]["base"]["commit"] == (
        contract.V38_SOURCE_COMMIT
    )
    assert manifest["inherited_v39_preregistration_authority"][
        "execution_authority"
    ] is False
    assert manifest["v316_rejection_authority"] == (
        contract.build_v316_rejection_pins()
    )
    assert manifest["v317_rejection_authority"] == (
        contract.build_v317_rejection_pins()
    )
    assert manifest["v318_rejection_authority"] == (
        contract.build_v318_rejection_pins()
    )
    assert manifest["model"]["input_caps"] == {
        "bytes": 20_000,
        "canonical_request_bytes": 131_072,
        "sentences": 72,
        "characters_per_sentence": 220,
    }
    assert manifest["qualification"] == contract.build_qualification_contract()
    assert contract.validate_contract_manifest(manifest) == manifest
    assert json.loads(contract.canonical_json_bytes(manifest)) == manifest

    validated = contract.validate_contract_manifest(manifest)
    validated["market"]["request_count"] = 7
    assert contract.build_contract_manifest()["market"]["request_count"] == 6

    changed = copy.deepcopy(manifest)
    changed["market"]["request_count"] = 7
    unsigned = dict(changed)
    unsigned.pop("contract_manifest_sha256")
    changed["contract_manifest_sha256"] = contract.canonical_sha256(unsigned)
    _assert_code(
        "v319_contract_manifest_mismatch", contract.validate_contract_manifest, changed
    )


def _prereg_evidence() -> dict[str, object]:
    return {
        "branch": contract.BRANCH_NAME,
        "commit": contract.PREREGISTRATION_COMMIT,
        "tree": contract.PREREGISTRATION_TREE,
        "parent": contract.BASE_COMMIT,
        "local_head": contract.PREREGISTRATION_COMMIT,
        "remote_head": contract.PREREGISTRATION_COMMIT,
        "changed_paths": {contract.PREREGISTRATION_PATH: "A"},
        "clean_worktree": True,
        "git_blob_sha1": contract.PREREGISTRATION_GIT_BLOB_SHA1,
        "literal_sha256": contract.PREREGISTRATION_LITERAL_SHA256,
        "literal_bytes": contract.PREREGISTRATION_LITERAL_BYTES,
    }


def _implementation_evidence() -> dict[str, object]:
    return {
        "branch": contract.BRANCH_NAME,
        "commit": SHA1_A,
        "tree": SHA1_B,
        "parent": contract.PREREGISTRATION_COMMIT,
        "local_head": SHA1_A,
        "remote_head": SHA1_A,
        "changed_paths": {path: "A" for path in contract.IMPLEMENTATION_ALLOWED_PATHS},
        "clean_worktree": True,
        "preregistration_authenticated": True,
        "predecessor_blobs_unchanged": True,
    }


def _preflight_evidence() -> dict[str, object]:
    return {
        "branch": contract.BRANCH_NAME,
        "commit": SHA1_C,
        "tree": SHA1_D,
        "parent": SHA1_A,
        "implementation_commit": SHA1_A,
        "implementation_tree": SHA1_B,
        "local_head": SHA1_C,
        "remote_head": SHA1_C,
        "changed_paths": {contract.PREFLIGHT_ARTIFACT_PATH: "A"},
        "clean_worktree": True,
        "implementation_authenticated": True,
        "private_replay_passed": True,
        "zero_effects_verified": True,
        "preflight_run_count": 1,
    }


def test_preregistration_implementation_and_preflight_ancestry() -> None:
    prereg = _prereg_evidence()
    assert contract.validate_preregistration_ancestry(prereg) == prereg
    changed_prereg = copy.deepcopy(prereg)
    changed_prereg["remote_head"] = SHA1_A
    _assert_code(
        "v319_contract_prereg_ancestry",
        contract.validate_preregistration_ancestry,
        changed_prereg,
    )

    implementation = _implementation_evidence()
    assert contract.validate_implementation_ancestry(implementation) == implementation
    changed_implementation = copy.deepcopy(implementation)
    changed_implementation["changed_paths"].pop(contract.IMPLEMENTATION_TEST_PATHS[-1])
    _assert_code(
        "v319_contract_impl_delta",
        contract.validate_implementation_ancestry,
        changed_implementation,
    )
    changed_implementation = copy.deepcopy(implementation)
    changed_implementation["predecessor_blobs_unchanged"] = False
    _assert_code(
        "v319_contract_impl_ancestry",
        contract.validate_implementation_ancestry,
        changed_implementation,
    )

    preflight = _preflight_evidence()
    assert contract.validate_preflight_ancestry(preflight) == preflight
    changed_preflight = copy.deepcopy(preflight)
    changed_preflight["preflight_run_count"] = 2
    _assert_code(
        "v319_contract_preflight_ancestry",
        contract.validate_preflight_ancestry,
        changed_preflight,
    )
    changed_preflight = copy.deepcopy(preflight)
    changed_preflight["preflight_run_count"] = True
    _assert_code(
        "v319_contract_preflight_ancestry",
        contract.validate_preflight_ancestry,
        changed_preflight,
    )


def test_pause_continuation_and_result_ancestry() -> None:
    pause = {
        "branch": contract.BRANCH_NAME,
        "commit": SHA1_D,
        "tree": SHA1_E,
        "parent": SHA1_C,
        "preflight_commit": SHA1_C,
        "local_head": SHA1_D,
        "remote_head": SHA1_D,
        "changed_paths": {contract.PAUSE_ARTIFACT_PATH: "A"},
        "clean_worktree": True,
        "pilot_guard_authenticated": True,
    }
    assert contract.validate_pause_ancestry(pause) == pause

    continuation = {
        "branch": contract.BRANCH_NAME,
        "commit": SHA1_E,
        "tree": SHA1_A,
        "parent": SHA1_D,
        "pause_commit": SHA1_D,
        "preflight_commit": SHA1_C,
        "local_head": SHA1_E,
        "remote_head": SHA1_E,
        "changed_paths_from_pause": {
            contract.CONTINUATION_PREREGISTRATION_PATH: "A"
        },
        "changed_paths_from_preflight": {
            contract.PAUSE_ARTIFACT_PATH: "A",
            contract.CONTINUATION_PREREGISTRATION_PATH: "A",
        },
        "clean_worktree": True,
        "earlier_blobs_equal_preflight": True,
    }
    assert contract.validate_continuation_ancestry(continuation) == continuation

    normal_result = {
        "route": "normal",
        "branch": contract.BRANCH_NAME,
        "commit": SHA1_D,
        "tree": SHA1_E,
        "parent": SHA1_C,
        "authorized_parent": SHA1_C,
        "authorized_parent_kind": "preflight",
        "local_head": SHA1_D,
        "remote_head": SHA1_D,
        "changed_paths": {
            contract.RESULT_ARTIFACT_PATH: "A",
            contract.COMPARISON_PATH: "M",
        },
        "clean_worktree": True,
        "terminal_sealed": True,
        "independent_replay_passed": True,
        "privacy_passed": True,
        "v38_unchanged": True,
    }
    assert contract.validate_result_ancestry(normal_result) == normal_result

    paused_result = copy.deepcopy(normal_result)
    paused_result.update(
        {
            "route": "paused_resumed",
            "parent": SHA1_E,
            "authorized_parent": SHA1_E,
            "authorized_parent_kind": "continuation",
        }
    )
    assert contract.validate_result_ancestry(paused_result) == paused_result
    wrong_parent = copy.deepcopy(paused_result)
    wrong_parent["authorized_parent_kind"] = "preflight"
    _assert_code(
        "v319_contract_result_ancestry",
        contract.validate_result_ancestry,
        wrong_parent,
    )


def test_implementation_allowlist_and_public_paths_are_exact() -> None:
    assert len(contract.IMPLEMENTATION_PRODUCTION_PATHS) == 6
    assert len(contract.IMPLEMENTATION_TEST_PATHS) == 6
    assert len(contract.IMPLEMENTATION_ALLOWED_PATHS) == 12
    assert len(set(contract.IMPLEMENTATION_ALLOWED_PATHS)) == 12
    assert contract.IMPLEMENTATION_PRODUCTION_PATHS[0].endswith("v319_contract.py")
    assert contract.IMPLEMENTATION_TEST_PATHS[0].endswith("v319_contract.py")
    assert contract.PREFLIGHT_ARTIFACT_PATH == (
        "e/aapl_sec_gemma_lean_science_v3_19/DEVELOPMENT_PREFLIGHT.json"
    )
    assert contract.PAUSE_ARTIFACT_PATH.endswith("/DEVELOPMENT_PAUSE.json")
    assert contract.RESULT_ARTIFACT_PATH.endswith("/DEVELOPMENT_RESULT.json")
    assert contract.COMPARISON_PATH == "e/APPROACH_COMPARISON.md"
    assert contract.PRIVATE_PREFLIGHT_NAMESPACE != contract.PRIVATE_DEVELOPMENT_NAMESPACE


def _direction_source_event(
    current_texts: tuple[str, ...],
    prior_texts: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Build a real inherited source event without claiming raw provenance here."""

    current = ". ".join(current_texts) + "."
    prior = None if prior_texts is None else ". ".join(prior_texts) + "."
    return preprocess_filing_event(
        current_normalized_text=current,
        prior_same_form_normalized_text=prior,
        identity_lexicon=CANONICAL_IDENTITY_LEXICON,
    )


def _reseal_direction_candidate(value: dict[str, object]) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop("preprocessed_event_sha256", None)
    return {
        **body,
        "preprocessed_event_sha256": contract.canonical_sha256(body),
    }


def test_direction_sanitizer_filters_complete_security_direction_grammar_and_preserves_exclusions() -> None:
    linker_cases = {
        "are": "Shares are lower.",
        "closed": "Stock closed lower.",
        "had": "Market had very sharply underperformed.",
        "has": "Stock has gained.",
        "have": "Shares have materially dropped.",
        "is": "Stock is lower.",
        "moved": "Shares moved sharply lower.",
        "moving": "Stocks moving moderately down.",
        "traded": "Securities traded very sharply lower.",
        "was": "Security was up.",
        "were": "Securities were down.",
    }
    assert set(linker_cases) == {
        "are",
        "closed",
        "had",
        "has",
        "have",
        "is",
        "moved",
        "moving",
        "traded",
        "was",
        "were",
    }
    forbidden = tuple(
        dict.fromkeys(
            (
                *linker_cases.values(),
                "Stock rose.",
                "Stocks moving moderately down.",
                "Security gained.",
                "Securities were down.",
                "Share value was slightly higher.",
                "Shares fell.",
                "Market rose.",
                "Stock is lower.",
                "Stock is sharply lower.",
                "Security was very sharply lower.",
                "Securities are quite significantly down.",
                "Stock price is lower.",
                "Stock prices are materially lower.",
                "Share value was slightly higher.",
                "Security values were very sharply lower.",
                "Higher share price.",
                "Lower market values.",
                "Stock-price was lower.",
                "Shares-are-very-sharply-lower.",
            )
        )
    )
    assert len(forbidden) == 26

    allowed = (
        "Stock compensation was lower.",
        "Shares issued under the plan were lower.",
        "Lower operating costs affected stock compensation.",
        "Stock; lower business costs.",
        "Shares, sharply lower input costs.",
        "Market: very sharply lower expenses.",
        "Security (lower legal costs).",
        "Shareholder value rose.",
        "Aftermarket rose.",
        "Stock unexpectedly rose.",
        "Stock very sharply significantly lower.",
        "Market share rose.",
        "Market shares rose.",
        "Market-share rose.",
        "Market-shares rose.",
    )
    dangerous_span_cooccurrences = (
        "Market share improved while shares rose.",
        "Shares rose while market share improved.",
        "Market shares rose while security fell.",
        "Higher shares followed market share gains.",
        "Market; shares rose.",
        "Market / shares rose.",
    )

    for text in forbidden:
        assert contract._BLINDED_TEXT_SECURITY_DIRECTION_RE.search(
            text.casefold()
        ) is not None, text
        assert contract._blinded_text_has_security_direction({"text": text}) is True
    for text in allowed:
        assert contract._blinded_text_has_security_direction({"text": text}) is False
    for text in dangerous_span_cooccurrences:
        assert contract._blinded_text_has_security_direction({"text": text}) is True

    assert contract._BLINDED_TEXT_MARKET_SHARE_SPAN_RE.pattern == (
        r"(?<![a-z0-9])market[ -]+shares?(?![a-z0-9])"
    )

    source = _direction_source_event(
        tuple(text.removesuffix(".") for text in (*forbidden, "Business demand stayed stable.")),
        tuple(text.removesuffix(".") for text in allowed),
    )
    source_sentences = list(source["sentences"])
    expected_retained_texts = [
        sentence["text"]
        for sentence in source_sentences
        if not contract._blinded_text_has_security_direction(sentence)
    ]
    expected_removed = sum(
        contract._blinded_text_has_security_direction(sentence)
        for sentence in source_sentences
    )
    sanitized = contract.build_direction_sanitized_preprocessed_event(source)

    assert [sentence["text"] for sentence in sanitized["sentences"]] == (
        expected_retained_texts
    )
    assert (
        sanitized["removed_current_sentence_count"]
        + sanitized["removed_prior_sentence_count"]
        == expected_removed
    )
    assert all(
        not contract._blinded_text_has_security_direction(sentence)
        for sentence in sanitized["sentences"]
    )
    # The span is used only as a temporary predicate barrier. It never edits
    # a retained sentence or inserts the internal semicolon mask.
    for preserved in ("Market share rose", "Market shares rose", "Market-share rose"):
        assert preserved in [sentence["text"] for sentence in sanitized["sentences"]]


def test_direction_sanitizer_renumbers_current_and_prior_deterministically() -> None:
    current = (
        "Shares rose",
        "Business demand remained stable",
        "Customer adoption remained durable",
        "Stock was up",
        "Supply relationships remained resilient",
        "Operating discipline remained consistent",
        "Security gained",
    )
    prior = (
        "Market fell",
        "Prior demand remained stable",
        "Prior customer adoption remained durable",
        "Shares moved sharply lower",
        "Prior supply relationships remained resilient",
        "Prior operating discipline remained consistent",
        "Stock closed lower",
    )
    source = _direction_source_event(current, prior)
    assert [sentence["id"] for sentence in source["sentences"]] == [
        *(f"C{index:04d}" for index in range(1, 8)),
        *(f"P{index:04d}" for index in range(1, 8)),
    ]

    first = contract.build_direction_sanitized_preprocessed_event(source)
    second = contract.build_direction_sanitized_preprocessed_event(
        copy.deepcopy(source)
    )
    assert first == second
    assert first["removed_current_sentence_count"] == 3
    assert first["removed_prior_sentence_count"] == 3
    assert first["retained_current_sentence_count"] == 4
    assert first["retained_prior_sentence_count"] == 4
    assert first["sentences"] == [
        {"id": "C0001", "text": current[1]},
        {"id": "C0002", "text": current[2]},
        {"id": "C0003", "text": current[4]},
        {"id": "C0004", "text": current[5]},
        {"id": "P0001", "text": prior[1]},
        {"id": "P0002", "text": prior[2]},
        {"id": "P0003", "text": prior[4]},
        {"id": "P0004", "text": prior[5]},
    ]
    retained_source_sentences = [
        source["sentences"][1],
        source["sentences"][2],
        source["sentences"][4],
        source["sentences"][5],
        source["sentences"][8],
        source["sentences"][9],
        source["sentences"][11],
        source["sentences"][12],
    ]
    assert [
        output["text"].encode("utf-8") for output in first["sentences"]
    ] == [
        source_sentence["text"].encode("utf-8")
        for source_sentence in retained_source_sentences
    ]


def test_direction_sanitizer_fails_closed_when_required_partition_would_be_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_current = _direction_source_event(("Shares rose",))
    empty_prior = _direction_source_event(
        ("Business demand remained stable",),
        ("Stock was up",),
    )
    request_calls: list[object] = []
    external_calls: list[str] = []

    def forbidden_request(*args: object, **kwargs: object) -> None:
        request_calls.append((args, kwargs))
        raise AssertionError("request construction or validation was reached")

    def forbidden_external(*_args: object, **_kwargs: object) -> None:
        external_calls.append("effect")
        raise AssertionError("empty-partition rejection touched an external surface")

    monkeypatch.setattr(contract, "validate_blinded_model_request", forbidden_request)
    monkeypatch.setattr(builtins, "open", forbidden_external)
    monkeypatch.setattr(socket, "create_connection", forbidden_external)
    monkeypatch.setattr(subprocess, "run", forbidden_external)
    monkeypatch.setattr(time, "time", forbidden_external)
    monkeypatch.setattr(time, "monotonic", forbidden_external)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_external)
    _assert_code(
        "v319_direction_sanitizer_empty_current",
        contract.build_direction_sanitized_preprocessed_event,
        empty_current,
    )
    _assert_code(
        "v319_direction_sanitizer_empty_prior",
        contract.build_direction_sanitized_preprocessed_event,
        empty_prior,
    )
    monkeypatch.undo()
    assert request_calls == []
    assert external_calls == []


def test_direction_sanitizer_rebuilds_payload_and_binds_source_and_output_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _direction_source_event(
        (
            "Shares rose",
            "Business demand remained stable",
            "Customer adoption remained durable",
        ),
        (
            "Stock was up",
            "Prior demand remained stable",
            "Prior customer adoption remained durable",
        ),
    )
    source_body = copy.deepcopy(source)
    source_hash = source_body.pop("preprocessed_event_sha256")
    assert source_hash == contract.canonical_sha256(source_body)
    assert source["sentences_sha256"] == contract.canonical_sha256(
        source["sentences"]
    )
    assert source["model_payload"] == build_extractor_model_payload(
        source["sentences"]
    )
    assert source["model_payload_sha256"] == contract.canonical_sha256(
        source["model_payload"]
    )

    external_calls: list[str] = []

    def forbidden_external(*_args: object, **_kwargs: object) -> None:
        external_calls.append("effect")
        raise AssertionError("pure sanitizer touched an external surface")

    monkeypatch.setattr(builtins, "open", forbidden_external)
    monkeypatch.setattr(socket, "create_connection", forbidden_external)
    monkeypatch.setattr(subprocess, "run", forbidden_external)
    monkeypatch.setattr(time, "time", forbidden_external)
    monkeypatch.setattr(time, "monotonic", forbidden_external)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_external)

    sanitized = contract.build_direction_sanitized_preprocessed_event(source)
    monkeypatch.undo()
    assert external_calls == []
    retained = [
        {"id": "C0001", "text": "Business demand remained stable"},
        {"id": "C0002", "text": "Customer adoption remained durable"},
        {"id": "P0001", "text": "Prior demand remained stable"},
        {"id": "P0002", "text": "Prior customer adoption remained durable"},
    ]
    rebuilt_payload = build_extractor_model_payload(retained)
    expected_body = {
        "schema_version": (
            "aapl-sec-gemma-lean-science-v3-19-direction-sanitized-event-v1"
        ),
        "sanitizer_schema_version": (
            "aapl-sec-gemma-lean-science-v3-19-security-direction-filter-v1"
        ),
        "source_preprocessed_event_sha256": source_hash,
        "source_sentence_count": 6,
        "removed_current_sentence_count": 1,
        "removed_prior_sentence_count": 1,
        "retained_current_sentence_count": 2,
        "retained_prior_sentence_count": 2,
        "sentences": retained,
        "sentences_sha256": contract.canonical_sha256(retained),
        "model_payload": rebuilt_payload,
        "model_payload_sha256": contract.canonical_sha256(rebuilt_payload),
    }
    assert sanitized == {
        **expected_body,
        "preprocessed_event_sha256": contract.canonical_sha256(expected_body),
    }
    assert inspect.signature(
        contract.validate_direction_sanitized_preprocessed_event
    ).parameters["expected_source_preprocessed_event_sha256"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    validated = contract.validate_direction_sanitized_preprocessed_event(
        source,
        sanitized,
        expected_source_preprocessed_event_sha256=source_hash,
    )
    assert validated == sanitized
    validated["sentences"][0]["text"] = "detached mutation"
    assert sanitized["sentences"][0]["text"] == retained[0]["text"]

    stale_outer = copy.deepcopy(sanitized)
    stale_outer["retained_current_sentence_count"] = 3
    _assert_code(
        "v319_direction_sanitized_event_invalid",
        contract.validate_direction_sanitized_preprocessed_event,
        source,
        stale_outer,
        expected_source_preprocessed_event_sha256=source_hash,
    )

    outer_rehashed_cases: list[dict[str, object]] = []
    wrong_schema = copy.deepcopy(sanitized)
    wrong_schema["schema_version"] = "wrong-v1"
    outer_rehashed_cases.append(_reseal_direction_candidate(wrong_schema))

    wrong_payload = copy.deepcopy(sanitized)
    wrong_payload["model_payload"]["stream"] = True
    wrong_payload["model_payload_sha256"] = contract.canonical_sha256(
        wrong_payload["model_payload"]
    )
    outer_rehashed_cases.append(_reseal_direction_candidate(wrong_payload))

    wrong_count = copy.deepcopy(sanitized)
    wrong_count["removed_current_sentence_count"] = 2
    outer_rehashed_cases.append(_reseal_direction_candidate(wrong_count))

    wrong_order = copy.deepcopy(sanitized)
    wrong_order["sentences"][0], wrong_order["sentences"][1] = (
        wrong_order["sentences"][1],
        wrong_order["sentences"][0],
    )
    wrong_order["sentences_sha256"] = contract.canonical_sha256(
        wrong_order["sentences"]
    )
    wrong_order["model_payload"] = build_extractor_model_payload(
        wrong_order["sentences"]
    )
    wrong_order["model_payload_sha256"] = contract.canonical_sha256(
        wrong_order["model_payload"]
    )
    outer_rehashed_cases.append(_reseal_direction_candidate(wrong_order))

    wrong_nested_hash = copy.deepcopy(sanitized)
    wrong_nested_hash["sentences_sha256"] = "f" * 64
    outer_rehashed_cases.append(_reseal_direction_candidate(wrong_nested_hash))

    for candidate in outer_rehashed_cases:
        _assert_code(
            "v319_direction_sanitized_event_invalid",
            contract.validate_direction_sanitized_preprocessed_event,
            source,
            candidate,
            expected_source_preprocessed_event_sha256=source_hash,
        )

    outer_rehashed_source = copy.deepcopy(source)
    outer_rehashed_source["sentences"][0]["text"] = "Changed source text"
    outer_rehashed_source = _reseal_direction_candidate(outer_rehashed_source)
    _assert_code(
        "v319_direction_sanitizer_source_invalid",
        contract.build_direction_sanitized_preprocessed_event,
        outer_rehashed_source,
    )
    _assert_code(
        "v319_direction_sanitized_event_invalid",
        contract.validate_direction_sanitized_preprocessed_event,
        source,
        sanitized,
        expected_source_preprocessed_event_sha256="f" * 64,
    )
