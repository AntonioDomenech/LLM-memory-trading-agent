from __future__ import annotations

import copy
import json

import pytest

from agent_benchmark import sec_gemma_lean_science_v310_contract as contract


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
        "v310_contract_json_nonfinite", contract.canonical_json_bytes, {"x": float("nan")}
    )
    _assert_code(
        "v310_contract_json_key", contract.canonical_json_bytes, {1: "not allowed"}
    )
    _assert_code(
        "v310_contract_json_type", contract.canonical_json_bytes, {"x": object()}
    )
    _assert_code("v310_contract_hash_input", contract.sha256_bytes, "bytes required")


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
    _assert_code("v310_contract_self_hash_mismatch", contract.validate_self_sha256, changed)
    _assert_code("v310_contract_self_hash_present", contract.add_self_sha256, sealed)


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
        "v310_contract_science_projection_hash",
        contract.verify_frozen_science_projection,
        changed,
    )
    missing = copy.deepcopy(projection)
    missing.pop("ledger")
    _assert_code(
        "v310_contract_science_projection_keys",
        contract.verify_frozen_science_projection,
        missing,
    )
    extra = {**projection, "operational_plumbing": {}}
    _assert_code(
        "v310_contract_science_projection_keys",
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


def test_three_git_authority_families_are_exact_and_disjoint() -> None:
    assert (contract.BASE_COMMIT, contract.BASE_TREE, contract.BASE_PARENT) == (
        "23a444d4f0b0e147823a33f48a7affcfce48d5a8",
        "f405b14858db263d7185072c04623621c75e353d",
        "29e7ecb4f3e42f3d50329e9d2337cd3eb91a49ee",
    )
    assert (
        contract.PREREGISTRATION_COMMIT,
        contract.PREREGISTRATION_TREE,
        contract.PREREGISTRATION_GIT_BLOB_SHA1,
        contract.PREREGISTRATION_LITERAL_SHA256,
        contract.PREREGISTRATION_LITERAL_BYTES,
    ) == (
        "bf66bb95f1ab44bb36b98523f6246761c23bcca3",
        "db57492353bdc90c49bd6b2a5762656e323b16a5",
        "f6b5c022a909e0a3e0d3a53fc999c34206be91b4",
        "d09ff0d302a910f3bdea37c9406342ffffc77b0686e3e28267d0e0079493e47b",
        30_746,
    )
    assert contract.PREREGISTRATION_PATH == (
        "docs/aapl_sec_gemma_lean_science_v3_10.md"
    )
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
            contract.V38_SOURCE_COMMIT,
            contract.V39_INHERITED_PREREG_COMMIT,
        }
    ) == 4


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
        "v310_contract_effect_counts",
        contract.validate_effect_counts,
        changed,
        route="normal_complete",
    )
    _assert_code(
        "v310_contract_effect_route",
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
    _assert_code("v310_contract_pilot_durations", contract.projected_pilot_ns, exact[:4])
    _assert_code(
        "v310_contract_pilot_durations",
        contract.projected_pilot_ns,
        [exact_duration, exact_duration, exact_duration, exact_duration, 0],
    )


def test_only_literal_development_command_is_accepted() -> None:
    assert contract.validate_command("development") == "development"
    for rejected in ("confirmation", "final", "development ", None, b"development"):
        _assert_code("v310_contract_command", contract.validate_command, rejected)


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
        "fdf7f282ddebd74eeb1e8a18c68a216aec2edf72dce68eda06ee83107591ee46"
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
        "v310_contract_local_import_duplicate",
        contract.validate_local_import_paths,
        (observed[0], observed[0]),
    )
    _assert_code(
        "v310_contract_local_import_outside_closure",
        contract.validate_local_import_paths,
        ("agent_benchmark/unpreregistered_strategy.py",),
    )
    _assert_code(
        "v310_contract_local_import_shape",
        contract.validate_local_import_paths,
        "agent_benchmark/sec_filing_content.py",
    )


def test_qualification_scope_schemas_runtime_and_counts_are_frozen() -> None:
    qualification = contract.build_qualification_contract()
    assert qualification["suite_id"] == (
        "latest_approach_actual_dependencies_pinned_parent_authority"
    )
    assert qualification["phases"] == ["v310", "dependencies", "requests_identity"]
    assert qualification["modes"] == ["collection", "execution"]
    assert qualification["environment"] == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    assert qualification["pytest_args"] == [
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    assert qualification["latest"] == {
        "phase": "v310",
        "test_paths": list(contract.IMPLEMENTATION_TEST_PATHS),
        "minimum_node_count": 130,
    }
    dependencies = qualification["dependencies"]
    assert len(dependencies["test_paths"]) == 14
    assert dependencies["node_count"] == 655
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
        "469e42a430cf194197a2e7dc76c6a603be5e0d01c7569cd33b53ef4f54372fc2"
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
        "v310_contract_manifest_mismatch", contract.validate_contract_manifest, changed
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
        "v310_contract_prereg_ancestry",
        contract.validate_preregistration_ancestry,
        changed_prereg,
    )

    implementation = _implementation_evidence()
    assert contract.validate_implementation_ancestry(implementation) == implementation
    changed_implementation = copy.deepcopy(implementation)
    changed_implementation["changed_paths"].pop(contract.IMPLEMENTATION_TEST_PATHS[-1])
    _assert_code(
        "v310_contract_impl_delta",
        contract.validate_implementation_ancestry,
        changed_implementation,
    )
    changed_implementation = copy.deepcopy(implementation)
    changed_implementation["predecessor_blobs_unchanged"] = False
    _assert_code(
        "v310_contract_impl_ancestry",
        contract.validate_implementation_ancestry,
        changed_implementation,
    )

    preflight = _preflight_evidence()
    assert contract.validate_preflight_ancestry(preflight) == preflight
    changed_preflight = copy.deepcopy(preflight)
    changed_preflight["preflight_run_count"] = 2
    _assert_code(
        "v310_contract_preflight_ancestry",
        contract.validate_preflight_ancestry,
        changed_preflight,
    )
    changed_preflight = copy.deepcopy(preflight)
    changed_preflight["preflight_run_count"] = True
    _assert_code(
        "v310_contract_preflight_ancestry",
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
        "v310_contract_result_ancestry",
        contract.validate_result_ancestry,
        wrong_parent,
    )


def test_implementation_allowlist_and_public_paths_are_exact() -> None:
    assert len(contract.IMPLEMENTATION_PRODUCTION_PATHS) == 6
    assert len(contract.IMPLEMENTATION_TEST_PATHS) == 6
    assert len(contract.IMPLEMENTATION_ALLOWED_PATHS) == 12
    assert len(set(contract.IMPLEMENTATION_ALLOWED_PATHS)) == 12
    assert contract.IMPLEMENTATION_PRODUCTION_PATHS[0].endswith("v310_contract.py")
    assert contract.IMPLEMENTATION_TEST_PATHS[0].endswith("v310_contract.py")
    assert contract.PREFLIGHT_ARTIFACT_PATH == (
        "e/aapl_sec_gemma_lean_science_v3_10/DEVELOPMENT_PREFLIGHT.json"
    )
    assert contract.PAUSE_ARTIFACT_PATH.endswith("/DEVELOPMENT_PAUSE.json")
    assert contract.RESULT_ARTIFACT_PATH.endswith("/DEVELOPMENT_RESULT.json")
    assert contract.COMPARISON_PATH == "e/APPROACH_COMPARISON.md"
    assert contract.PRIVATE_PREFLIGHT_NAMESPACE != contract.PRIVATE_DEVELOPMENT_NAMESPACE
