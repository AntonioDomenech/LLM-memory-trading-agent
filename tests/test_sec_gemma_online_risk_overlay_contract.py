from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from agent_benchmark.sec_session_calendar import (
    CALENDAR_ID,
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
    MARKET_HISTORY_CALENDAR_ID,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    FEATURES,
    FINAL_ATTEMPT_ID,
    HORIZON_SESSIONS,
    MAX_TOTAL_RUNTIME_SECONDS,
    MODEL_MANIFEST_SHA256,
    RUNTIME_FINGERPRINT_SHA256,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    SecGemmaOnlineRiskOverlayContractError,
    build_contract_manifest,
    build_runtime_fingerprint_material,
    canonical_sha256,
    validate_contract_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_exact_deterministic_and_detached() -> None:
    first = build_contract_manifest()
    second = build_contract_manifest()

    assert first == second
    assert first is not second
    assert canonical_sha256(first) == CONTRACT_SHA256
    assert CONTRACT_SHA256 == (
        "57b325b25ae53f650538a0265622a6c662e7fb1894a78e805705f7bae3c55d5d"
    )
    assert first["contract_version"] == CONTRACT_VERSION
    assert first["features"]["ordered_names"] == list(FEATURES)
    assert first["features"]["count"] == 12
    assert first["policy"]["overlay_horizon_sessions"] == HORIZON_SESSIONS
    assert first["runtime"]["total_seconds_strictly_below"] == 3600
    assert MAX_TOTAL_RUNTIME_SECONDS == 3600

    first["features"]["ordered_names"][0] = "mutated"
    assert build_contract_manifest() == second


def test_contract_freezes_live_causal_learning_and_controls() -> None:
    manifest = validate_contract_manifest(build_contract_manifest())
    chronology = manifest["chronology"]
    learner = manifest["learner"]

    assert learner["training_mode"] == (
        "continuous expanding causal refit before each filing"
    )
    assert "t+21" in learner["training_rows"]
    assert "incremental" in learner["label"]
    assert "fixed baseline" in learner["probability_head"]
    assert chronology["updates_inside_every_period"] == (
        "permitted only after the complete 20-session outcome matures"
    )
    assert "no annual cutoff" in chronology["every_2025_decision_state"]
    assert "through-2023" in chronology["controls"]["final"]
    assert learner["same_session_matured_label_admission"] == (
        "maturity_session <= decision_session"
    )


def test_contract_freezes_long_cash_only_and_same_ledger_costs() -> None:
    manifest = build_contract_manifest()
    objective = manifest["objective"]
    policy = manifest["policy"]

    assert objective["allowed_target_exposures"] == [0, 1]
    assert objective["maximum_target_exposure"] == 1
    assert objective["maximum_realized_exposure"] == 1
    assert objective["shorting"] is False
    assert objective["leverage"] is False
    assert objective["borrowing"] is False
    assert objective["negative_cash"] is False
    assert objective["paid_api_calls"] == 0
    assert policy["costs_bps_per_changing_leg"] == [5, 10]
    assert policy["same_action_stream_at_both_costs"] is True


def test_contract_requires_learning_and_semantics_to_change_actions() -> None:
    gates = build_contract_manifest()["gates"]

    assert gates["zero_action_difference_is_rejection"] is True
    assert (
        gates["development"][
            "online_vs_block_frozen_action_differences_at_least"
        ]
        == 5
    )
    assert (
        gates["development"][
            "semantic_vs_no_filing_meaning_action_differences_at_least"
        ]
        == 5
    )
    assert (
        gates["final"]["post_2023_online_vs_frozen_action_differences_at_least"]
        == 3
    )
    assert gates["final"]["online_vs_frozen_continuous_10bps_strictly_positive"]


def test_bound_source_files_match_literal_sha256_pins() -> None:
    assert set(SOURCE_PIN_FILES) == set(SOURCE_PINS)
    for role, relative_path in SOURCE_PIN_FILES.items():
        payload = (REPO_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == SOURCE_PINS[role]


def test_final_reveal_registry_predecessor_pin_matches_literal_bytes() -> None:
    final_access = build_contract_manifest()["stage_access"]["final"]
    relative_path = final_access["predecessor_registry_pin_file"]
    payload = (REPO_ROOT / relative_path).read_bytes()

    assert hashlib.sha256(payload).hexdigest() == final_access[
        "predecessor_registry_pin_file_sha256"
    ]
    assert final_access["historical_final_reveal_count_lower_bound"] == 10
    assert len(final_access["predecessor_registry_sha256"]) == 64
    assert len(final_access["predecessor_registry_tip_sha256"]) == 64


def test_model_runtime_and_horizon_are_literal_not_placeholders() -> None:
    manifest = build_contract_manifest()
    gemma = manifest["gemma"]
    horizon = manifest["data"]["market"]["horizon"]

    assert gemma["model_manifest_sha256"] == MODEL_MANIFEST_SHA256
    assert len(MODEL_MANIFEST_SHA256) == 64
    assert gemma["runtime_fingerprint_sha256"] == RUNTIME_FINGERPRINT_SHA256
    assert len(RUNTIME_FINGERPRINT_SHA256) == 64
    assert canonical_sha256(build_runtime_fingerprint_material()) == (
        RUNTIME_FINGERPRINT_SHA256
    )
    assert horizon == {
        "decision_close_index": "t",
        "entry_open_index": "t+1",
        "exit_open_index": "t+21",
        "held_open_to_open_intervals": 20,
        "label_maturity_session_index": "t+21",
    }


def test_calendar_ids_and_canonical_date_sequences_match_literal_pins() -> None:
    calendar = build_contract_manifest()["data"]["market"]["calendar"]

    assert calendar["calendar_id"] == CALENDAR_ID
    assert calendar["market_calendar_id"] == MARKET_HISTORY_CALENDAR_ID
    assert canonical_sha256(list(EXPECTED_SESSIONS)) == calendar[
        "calendar_dates_sha256"
    ]
    assert canonical_sha256(list(EXPECTED_MARKET_HISTORY_SESSIONS)) == calendar[
        "market_calendar_dates_sha256"
    ]


def test_baseline_is_prefix_invariant_raw_union_not_actionable_mask() -> None:
    policy = build_contract_manifest()["policy"]

    assert policy["baseline_kind"] == (
        "fixed raw expert union, not binary-regime selector"
    )
    assert policy["baseline_signal_column"] == "unfiltered_union_signal"
    assert policy["baseline_forbidden_column"] == (
        "unfiltered_union_target_exposure"
    )
    assert "may not rewrite" in policy["baseline_prefix_invariance"]
    assert "never suppresses" in policy["baseline_cooldown_independence"]


def test_stage_counts_attempts_and_metric_math_are_unambiguous() -> None:
    manifest = build_contract_manifest()

    assert manifest["data"]["sec"][
        "minimum_development_corpus_filings_2000_2018"
    ] == 72
    assert manifest["chronology"]["development_corpus"] == [
        "2000-01-01",
        "2018-12-31",
    ]
    assert manifest["stage_access"]["confirmation"]["one_shot"] is True
    assert manifest["stage_access"]["final"]["one_shot"] is True
    assert manifest["metric_definitions"]["positive_tolerance"] == 1e-12
    assert "max(ablation_brier, 1e-12)" in manifest["metric_definitions"][
        "brier_relative_improvement"
    ]


def test_all_effectful_stages_are_one_shot_and_locked_before_reveal() -> None:
    access = build_contract_manifest()["stage_access"]

    assert access["development_acquisition"]["attempt_id"] == (
        DEVELOPMENT_ACQUISITION_ID
    )
    assert access["development_acquisition"]["one_shot"] is True
    assert access["development_acquisition"][
        "consume_before_first_official_sec_or_market_network_request"
    ] is True
    assert access["development_acquisition"][
        "failed_or_indeterminate_acquisition_is_terminal"
    ] is True
    assert access["development"]["attempt_id"] == DEVELOPMENT_ATTEMPT_ID
    assert access["development"]["one_shot"] is True
    assert access["development"][
        "consume_before_first_real_gemma_call_or_first_canonical_market_value_read"
    ] is True
    assert access["development"][
        "latency_preflight_calls_occur_after_consumption"
    ] is True
    assert access["confirmation"]["attempt_id"] == CONFIRMATION_ATTEMPT_ID
    assert access["confirmation"]["consume_before_first_stage_network_request"] is True
    assert access["confirmation"][
        "consume_before_first_2019_feature_or_outcome_read"
    ] is True
    assert access["final"]["attempt_id"] == FINAL_ATTEMPT_ID
    assert access["final"]["consume_before_first_stage_network_request"] is True
    assert access["final"][
        "consume_before_first_2024_feature_or_outcome_read"
    ] is True
    assert "semantic extraction" in access["result_release"]


def test_market_provider_windows_and_prefix_continuity_are_frozen() -> None:
    market = build_contract_manifest()["data"]["market"]
    source = market["source_acquisition"]

    assert source["provider_family"] == (
        "yahoo-finance-chart-v8-public-unauthenticated"
    )
    assert source["endpoint"] == (
        "https://query1.finance.yahoo.com/v8/finance/chart"
    )
    assert source["request_order"] == [
        "AAPL",
        "SPY",
        "QQQ",
        "IWM",
        "VIX",
        "TNX",
    ]
    assert source["request_count_each_stage"] == 6
    assert source["request_windows"]["development"]["period2_utc"] == 1546300800
    assert source["request_windows"]["confirmation"]["period2_utc"] == 1704067200
    assert source["request_windows"]["final"]["period2_utc"] == 1783728000
    assert source["request_windows"]["final"][
        "last_exposed_market_value_session"
    ] == "2026-07-09"
    assert source["request_windows"]["final"][
        "last_scored_fill_origin_decision_session"
    ] == "2026-07-08"
    assert source["request_windows"]["final"][
        "last_pending_prediction_session"
    ] == "2026-07-09"
    assert market["ledger_price_fields"] == [
        "raw_open",
        "raw_close",
        "adjusted_close",
    ]
    assert "float bit pattern" in source["canonical_prefix_rule"]
    assert "terminally fails" in source["canonical_prefix_rule"]
    assert "only after their durable stage locks" in source[
        "visibility_and_lock_rule"
    ]
    assert "adjusted_open" in market["ledger_price_validation"]
    assert "terminally fails the stage" in market["ledger_price_validation"]


def test_event_unavailability_never_becomes_an_implicit_imputation() -> None:
    availability = build_contract_manifest()["event_availability"]

    assert "not_comparable" in availability["first_same_form_filing"]
    assert "quality_risk to 1" in availability[
        "authenticated_schema_invalid_output"
    ]
    assert "exclude it from learner membership" in availability[
        "missing_or_unauthenticated_model_output"
    ]
    assert "no imputation" in availability["market_unavailable"]
    assert "audit-only" in availability["unavailable_label_rule"]
    assert "no fitted prediction" in availability["learner_unready"]
    assert "effective schedule flag false" in availability["active_overlay_event"]
    assert availability["undefined_or_nonfinite_metric"] == "terminal stage failure"


def test_filing_meaning_is_isolated_from_extraction_quality() -> None:
    manifest = build_contract_manifest()
    features = manifest["features"]
    gates = manifest["gates"]

    assert features["meaning_names"] == [
        "commercial_deterioration",
        "financial_deterioration",
        "risk_outlook_deterioration",
        "adverse_flag_fraction",
    ]
    assert features["quality_name"] == "semantic_quality_risk"
    assert "set only the four filing-meaning features to zero" in features[
        "no_filing_meaning_ablation"
    ]
    assert "preserve semantic_quality_risk exactly" in features[
        "no_filing_meaning_ablation"
    ]
    assert "cannot satisfy or rescue" in features["no_gemma_channel_diagnostic"]
    assert "schema-invalid outputs remain in the denominator" in features[
        "schema_valid_extraction_rate"
    ]
    assert "earlier cumulative rows cannot satisfy" in features[
        "extraction_coverage_stage_assignment"
    ]
    assert gates["development"]["schema_valid_extraction_rate_at_least"] == 0.90
    assert gates["development"]["nonzero_filing_meaning_rows_at_least"] == 24
    assert gates["confirmation"]["schema_valid_extraction_rate_at_least"] == 0.90
    assert gates["confirmation"]["nonzero_filing_meaning_rows_at_least"] == 6
    assert gates["final"]["schema_valid_extraction_rate_at_least"] == 0.90
    assert gates["final"]["nonzero_filing_meaning_rows_at_least"] == 3
    assert gates["final"][
        "semantic_vs_no_filing_meaning_action_differences_at_least"
    ] == 2


def test_semantic_arms_and_frozen_controls_have_exact_continuity() -> None:
    chronology = build_contract_manifest()["chronology"]
    controls = chronology["controls"]

    assert "start at portfolio genesis" in controls["semantic_arm_continuity"]
    assert "without reset" in controls["semantic_arm_continuity"]
    assert "same matured counterfactual labels" in controls["semantic_arm_labels"]
    assert "preceding session" in controls["fork_boundary_order"]
    assert "before any label maturing on the boundary session" in controls[
        "fork_boundary_order"
    ]
    assert "before any 2019-session admission" in controls["confirmation"]
    assert "before any 2024-session admission" in controls["final"]
    assert "acceptance timestamp ascending" in chronology["same_session_event_order"]
    assert "scheduled overlay blocks" in chronology["same_close_pending_overlay_rule"]


def test_metric_boundaries_drawdown_and_counting_are_literal() -> None:
    manifest = build_contract_manifest()
    chronology = manifest["chronology"]
    metrics = manifest["metric_definitions"]

    assert "performance and both terminal valuations end on 2026-07-09" in chronology[
        "final_cutoff_rule"
    ]
    assert "Only decisions through 2026-07-08" in chronology["final_cutoff_rule"]
    assert "append exactly one" in metrics["terminal_adjusted_close_report_variant"]
    assert "MDD = min(wealth/running_peak - 1)" in metrics["maximum_drawdown"]
    assert "strategy_MDD" in metrics["drawdown_comparison"]
    assert "count each block once" in metrics["difference_block"]
    assert "count each year once" in metrics["semantic_difference_year"]
    assert "zero and negative contributions are non-wins" in metrics[
        "episode_win_rate"
    ]
    assert "Semantic arms never fork after genesis" in metrics[
        "comparison_fork_rule"
    ]


def test_runtime_limit_applies_to_each_attempt_not_only_the_lifecycle() -> None:
    runtime = build_contract_manifest()["runtime"]

    assert "each scored stage attempt" in runtime["scope"]
    assert "multi-stage research lifecycle" in runtime["cumulative_lifecycle_rule"]
    assert runtime["market_seconds_at_most_within_acquisition"] == 210
    assert runtime["market_requests_each_stage"] == 6
    assert runtime["sec_plus_market_combined_seconds_at_most"] == 720
    assert runtime["contingency_seconds"] == 239
    assert runtime["maximum_phase_caps_plus_contingency_seconds"] == 3599
    assert (
        runtime["sec_plus_market_combined_seconds_at_most"]
        + runtime["gemma_seconds_at_most"]
        + runtime["deterministic_seconds_at_most"]
        + runtime["contingency_seconds"]
        == 3599
    )
    assert runtime["per_attempt_phase_budgets"]["development_acquisition"][
        "gemma_seconds_at_most"
    ] == 0
    assert runtime["per_attempt_phase_budgets"]["development_scored"][
        "acquisition_seconds_at_most"
    ] == 0
    for stage_budget in runtime["per_attempt_phase_budgets"].values():
        assert stage_budget["total_seconds_strictly_below"] == 3600
        assert (
            stage_budget["acquisition_seconds_at_most"]
            + stage_budget["gemma_seconds_at_most"]
            + stage_budget["deterministic_seconds_at_most"]
            + runtime["contingency_seconds"]
            < stage_budget["total_seconds_strictly_below"]
        )
    assert runtime["model_call_caps"] == {
        "development_2000_2018": 80,
        "confirmation_2019_2023": 20,
        "final_2024_2026_ytd": 12,
    }


def test_ledger_and_final_gates_freeze_both_terminal_valuations() -> None:
    manifest = build_contract_manifest()
    ledger = manifest["ledger"]
    final = manifest["gates"]["final"]

    assert ledger["buy_cost_factor"] == "1/(1+cost_bps/10000)"
    assert ledger["sell_cost_factor"] == "1-cost_bps/10000"
    assert "shares = pre_fill_cash" in ledger["buy_fill_arithmetic"]
    assert "shares = 0" in ledger["sell_fill_arithmetic"]
    assert ledger["adjusted_open_valuation"] == "cash + shares * adjusted_open"
    assert "no synthetic liquidation" in ledger[
        "terminal_adjusted_close_valuation"
    ]
    assert final["terminal_valuation_methods_required"] == [
        "adjusted_open",
        "terminal_adjusted_close",
    ]
    assert final[
        "all_return_edge_and_drawdown_gates_pass_under_both_terminal_valuations"
    ] is True
    assert final["undefined_metric_fails"] is True


def test_every_ambiguous_gate_has_an_explicit_cost_scope() -> None:
    gates = build_contract_manifest()["gates"]
    development = gates["development"]
    confirmation = gates["confirmation"]
    final = gates["final"]

    for key in (
        "combined_edge_without_best_block_10bps_at_least",
        "positive_combined_blocks_10bps_at_least",
        "annual_win_rate_10bps_at_least",
        "negative_aapl_year_win_rate_10bps_at_least",
        "overlay_episode_win_rate_10bps_at_least",
        "overlay_median_edge_10bps_strictly_positive",
        "largest_positive_episode_share_10bps_at_most",
        "incremental_vs_baseline_without_best_block_10bps_strictly_positive",
        "semantic_edge_without_best_xor_10bps_strictly_positive",
    ):
        assert key in development
    assert confirmation[
        "combined_positive_years_at_least_3_at_both_5_and_10bps"
    ] is True
    assert "incremental_without_best_episode_10bps_strictly_positive" in confirmation
    assert "incremental_vs_baseline_positive_periods_10bps_at_least" in final
    assert "overlay_episode_win_rate_10bps_at_least" in final
    assert "largest_positive_episode_share_10bps_at_most" in final
    assert (
        final["max_drawdown_not_worse_than_aapl_by_more_than_at_5_and_10bps"]
        == 0.01
    )
    assert gates["undefined_or_nonfinite_metric_fails_every_dependent_gate"] is True


def test_document_matches_key_machine_contract_facts() -> None:
    document = (
        REPO_ROOT / "docs/aapl_sec_gemma_online_risk_overlay_v2.md"
    ).read_text(encoding="utf-8")
    normalized_document = " ".join(document.split())

    for expected in (
        "`unfiltered_union_signal`",
        "`maturity_session <= decision_session`",
        "no-filing-meaning ablation",
        "adjusted open `t+21`",
        "one fixed attempt ID",
        "1,000 requests",
        "owned HTTPS request per symbol",
        "exactly six requests",
        "transport requests through 2026-07-10",
        "terminal valuation end on 2026-07-09",
        "Only a decision through July 8",
        "Only the four actual filing-meaning values",
        "quality-preserving no-meaning",
        "`MDD = min(wealth/running_peak - 1)`",
        "239 seconds reserved as contingency",
        "total 3,599 seconds",
        "at least two of the three final periods",
    ):
        assert expected in normalized_document

    ordered_features = "\n".join(
        (
            "1. AAPL minus QQQ 20-session log return;",
            "2. AAPL 63-session drawdown;",
            "3. AAPL 20-session realized volatility;",
            "4. SPY 20-session log return;",
            "5. IWM 20-session log return;",
            "6. VIX 20-session log change;",
            "7. commercial deterioration;",
            "8. financial deterioration;",
            "9. risk/outlook deterioration;",
            "10. adverse-flag fraction;",
            "11. semantic quality risk; and",
            "12. 10-K form indicator.",
        )
    )
    assert ordered_features in document


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("objective", "leverage"), True),
        (("gemma", "temperature"), 0.1),
        (("learner", "action_gate", "probability_at_least"), 0.5),
        (("policy", "overlay_horizon_sessions"), 21),
        (("runtime", "total_seconds_strictly_below"), 3601),
        (("gates", "zero_action_difference_is_rejection"), False),
        (("chronology", "every_2025_decision_state"), "future-aware"),
    ],
)
def test_any_material_mutation_is_rejected(
    path: tuple[str, ...], replacement: object
) -> None:
    value = copy.deepcopy(build_contract_manifest())
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(
        SecGemmaOnlineRiskOverlayContractError,
        match="differs from preregistration",
    ):
        validate_contract_manifest(value)


def test_non_mapping_and_noncanonical_json_are_rejected() -> None:
    with pytest.raises(SecGemmaOnlineRiskOverlayContractError, match="mapping"):
        validate_contract_manifest([])

    value = build_contract_manifest()
    value["not_json"] = float("nan")
    with pytest.raises(
        SecGemmaOnlineRiskOverlayContractError,
        match="not canonical JSON",
    ):
        validate_contract_manifest(value)
