from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import sys

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    ACTIVE_EDGE_TOLERANCE,
    BRIER_TARGET_COST_BPS,
    CANDIDATE_IDS,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    SecFilingGemmaContractError,
    canonical_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
    INTERMEDIATE_FOLD_ID,
    MODEL_VARIANTS,
    PREDICTION_PREFIX_SCHEMA_VERSION,
    PREDICTION_ROW_SCHEMA_VERSION,
    UNAVAILABLE_PREDICTION_STATUS,
    append_prediction_row,
    build_label_release_ledger,
    build_prediction_ledger,
    build_prelabel_seal_ledger,
    prediction_prefix_sha256,
    validate_label_release_ledger,
    validate_prediction_prefix,
    validate_prelabel_seal_ledger,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tagged(value: str) -> str:
    return f"sha256:{_hash(value)}"


def _event(
    accession: str,
    decision: str,
    *,
    fold_id: str = "fold_1",
    salt: str,
) -> dict[str, str]:
    return {
        "accession_number": accession,
        "form": "10-Q",
        "stage": "development",
        "decision_session": decision,
        "extraction_identity_sha256": _hash(f"{salt}:extraction"),
        "market_prefix_chain_identity_sha256": _hash(
            f"{salt}:market-prefix-chain"
        ),
        "market_feature_row_sha256": _hash(f"{salt}:market-row"),
        "fold_id": fold_id,
    }


def _fold(cutoff: str, maximum_maturity: str, salt: str) -> dict[str, object]:
    return {
        "fold_train_cutoff_session": cutoff,
        "training_set_count": 40,
        "training_positive_count": 18,
        "training_set_membership_sha256": _hash(f"{salt}:membership"),
        "semantic_training_feature_matrix_sha256": _hash(
            f"{salt}:semantic-features"
        ),
        "ablation_training_feature_matrix_sha256": _hash(
            f"{salt}:ablation-features"
        ),
        "training_binary_target_sha256": _hash(f"{salt}:binary"),
        "training_edge_target_sha256": _hash(f"{salt}:edge"),
        "training_set_max_label_maturity_session": maximum_maturity,
        "semantic_fold_state_sha256": _hash(f"{salt}:semantic-state"),
        "ablation_fold_state_sha256": _hash(f"{salt}:ablation-state"),
    }


def _available(
    event: dict[str, str],
    *,
    semantic_probability: float,
    semantic_edge: float,
    ablation_probability: float,
    ablation_edge: float,
) -> dict[str, object]:
    return {
        **event,
        "prediction_status": AVAILABLE_PREDICTION_STATUS,
        "unavailable_reason": None,
        "semantic_cash_probability": semantic_probability,
        "semantic_expected_edge": semantic_edge,
        "ablation_cash_probability": ablation_probability,
        "ablation_expected_edge": ablation_edge,
    }


def _unavailable(event: dict[str, str], reason: str) -> dict[str, object]:
    return {
        **event,
        "prediction_status": UNAVAILABLE_PREDICTION_STATUS,
        "unavailable_reason": reason,
        "semantic_cash_probability": None,
        "semantic_expected_edge": None,
        "ablation_cash_probability": None,
        "ablation_expected_edge": None,
    }


def _next_session(value: str, offset: int = 1) -> str:
    sessions = list(EXPECTED_SESSIONS)
    return sessions[sessions.index(value) + offset]


@pytest.fixture()
def evidence() -> dict[str, object]:
    calendar_hash = session_calendar_sha256(EXPECTED_SESSIONS)
    candidate_hash = _hash("candidate")
    universe_hash = _hash("universe")
    events = [
        _event("0000320193-05-000001", "2005-01-03", salt="one"),
        _event("0000320193-05-000002", "2005-01-10", salt="two"),
        _event("0000320193-05-000003", "2005-01-18", salt="three"),
        _event("0000320193-05-000004", "2005-02-10", salt="four"),
    ]
    specs = [
        _available(
            events[0],
            semantic_probability=0.90,
            semantic_edge=0.10,
            ablation_probability=0.90,
            ablation_edge=0.10,
        ),
        _available(
            events[1],
            semantic_probability=0.95,
            semantic_edge=0.20,
            ablation_probability=0.95,
            ablation_edge=0.20,
        ),
        _unavailable(events[2], "missing_required_market_features"),
        _available(
            events[3],
            semantic_probability=0.10,
            semantic_edge=-0.10,
            ablation_probability=0.10,
            ablation_edge=-0.10,
        ),
    ]
    fold_contexts = {
        "fold_1": _fold("2004-12-31", "2004-12-30", "fold-one")
    }
    prefix = build_prediction_ledger(
        specs,
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=calendar_hash,
        candidate_sha256=candidate_hash,
        corpus_universe_sha256=universe_hash,
        expected_event_bindings=events,
        fold_contexts=fold_contexts,
    )
    validation_kwargs = {
        "session_dates": EXPECTED_SESSIONS,
        "expected_calendar_sessions_sha256": calendar_hash,
        "expected_candidate_sha256": candidate_hash,
        "expected_corpus_universe_sha256": universe_hash,
        "expected_event_bindings": events,
    }
    summary = validate_prediction_prefix(prefix, **validation_kwargs)
    checksums = {
        prefix_hash: _tagged(f"external-prefix:{index}")
        for index, prefix_hash in enumerate(summary["prefix_ancestry_sha256s"])
    }
    seal = build_prelabel_seal_ledger(
        prefix,
        external_artifact_checksums_by_prefix_sha256=checksums,
        **validation_kwargs,
    )
    edges = [ACTIVE_EDGE_TOLERANCE, 0.02, -0.01, 0.03]
    labels = {
        row["prediction_row_sha256"]: {
            "cash_active_log_edge_10bps": edge,
            "strategy_ledger_slice_sha256": _hash(f"strategy:{index}"),
            "benchmark_ledger_slice_sha256": _hash(f"benchmark:{index}"),
            "outcome_ledger_row_sha256": _hash(f"outcome:{index}"),
        }
        for index, (row, edge) in enumerate(
            zip(prefix["rows"], edges, strict=True)
        )
    }
    release_as_of = _next_session(prefix["rows"][-1]["label_maturity_session"])
    release = build_label_release_ledger(
        prefix,
        seal,
        as_of_decision_session=release_as_of,
        labels_by_prediction_sha256=labels,
        external_artifact_checksums_by_prefix_sha256=checksums,
        **validation_kwargs,
    )
    return {
        "calendar_hash": calendar_hash,
        "candidate_hash": candidate_hash,
        "universe_hash": universe_hash,
        "events": events,
        "specs": specs,
        "fold_contexts": fold_contexts,
        "prefix": prefix,
        "summary": summary,
        "checksums": checksums,
        "seal": seal,
        "labels": labels,
        "release_as_of": release_as_of,
        "release": release,
        "validation_kwargs": validation_kwargs,
    }


def _rehash_seal(seal: dict[str, object]) -> None:
    parent = seal["genesis_sha256"]
    for index, entry in enumerate(seal["entries"], start=1):
        entry["sequence_number"] = index
        entry["parent_seal_sha256"] = parent
        body = {key: value for key, value in entry.items() if key != "seal_sha256"}
        entry["seal_sha256"] = canonical_sha256(body)
        parent = entry["seal_sha256"]
    seal["tip_sha256"] = parent
    seal["sealed_prediction_count"] = len(seal["entries"])
    body = {
        key: value
        for key, value in seal.items()
        if key != "prelabel_seal_ledger_sha256"
    }
    seal["prelabel_seal_ledger_sha256"] = canonical_sha256(body)


def _refresh_prefix_container(prefix: dict[str, object]) -> None:
    prefix["row_count"] = len(prefix["rows"])
    prefix["rows_sha256"] = canonical_sha256(prefix["rows"])


def _rehash_release(release: dict[str, object]) -> None:
    parent = release["genesis_sha256"]
    for index, entry in enumerate(release["entries"], start=1):
        entry["sequence_number"] = index
        entry["parent_release_sha256"] = parent
        body = {key: value for key, value in entry.items() if key != "release_sha256"}
        entry["release_sha256"] = canonical_sha256(body)
        parent = entry["release_sha256"]
    release["tip_sha256"] = parent
    release["release_count"] = len(release["entries"])
    release["released_prediction_sequence_sha256"] = canonical_sha256(
        [entry["prediction_row_sha256"] for entry in release["entries"]]
    )
    body = {
        key: value
        for key, value in release.items()
        if key != "label_release_ledger_sha256"
    }
    release["label_release_ledger_sha256"] = canonical_sha256(body)


def test_exact_append_replay_and_external_prior_pins(evidence) -> None:
    first_event = evidence["events"][0]
    first_spec = evidence["specs"][0]
    prefix_1 = append_prediction_row(
        None,
        first_spec,
        current_event_binding=first_event,
        current_fold_context=evidence["fold_contexts"]["fold_1"],
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        candidate_sha256=evidence["candidate_hash"],
        corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=evidence["events"][:1],
        expected_prior_prefix_sha256=None,
        expected_prior_tip_sha256=None,
    )
    assert prefix_1["prediction_prefix_sha256"] == prediction_prefix_sha256(
        evidence["prefix"], through_sequence_number=1
    )
    prefix_2 = append_prediction_row(
        prefix_1,
        evidence["specs"][1],
        current_event_binding=evidence["events"][1],
        current_fold_context=evidence["fold_contexts"]["fold_1"],
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        candidate_sha256=evidence["candidate_hash"],
        corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=evidence["events"][:2],
        expected_prior_prefix_sha256=prefix_1["prediction_prefix_sha256"],
        expected_prior_tip_sha256=prefix_1["tip_sha256"],
    )
    partial_validation_kwargs = {
        **evidence["validation_kwargs"],
        "expected_event_bindings": evidence["events"][:2],
    }
    summary = validate_prediction_prefix(
        prefix_2,
        expected_prediction_prefix_sha256=prefix_2["prediction_prefix_sha256"],
        expected_tip_sha256=prefix_2["tip_sha256"],
        **partial_validation_kwargs,
    )
    assert summary["row_count"] == 2
    assert summary["prefix_ancestry_sha256s"] == [
        prefix_1["prediction_prefix_sha256"],
        prefix_2["prediction_prefix_sha256"],
    ]

    with pytest.raises(SecFilingGemmaContractError, match="externally pinned"):
        append_prediction_row(
            prefix_1,
            evidence["specs"][1],
            current_event_binding=evidence["events"][1],
            current_fold_context=evidence["fold_contexts"]["fold_1"],
            session_dates=EXPECTED_SESSIONS,
            expected_calendar_sessions_sha256=evidence["calendar_hash"],
            candidate_sha256=evidence["candidate_hash"],
            corpus_universe_sha256=evidence["universe_hash"],
            expected_event_bindings=evidence["events"][:2],
            expected_prior_prefix_sha256=_hash("wrong-prior"),
            expected_prior_tip_sha256=prefix_1["tip_sha256"],
        )


@pytest.mark.parametrize("positive_count", [0, 40, 41])
def test_fold_context_requires_both_training_target_classes(
    evidence, positive_count: int
) -> None:
    changed = copy.deepcopy(evidence["fold_contexts"]["fold_1"])
    changed["training_positive_count"] = positive_count

    with pytest.raises(
        SecFilingGemmaContractError,
        match="training_positive_count|both binary classes",
    ):
        append_prediction_row(
            None,
            evidence["specs"][0],
            current_event_binding=evidence["events"][0],
            current_fold_context=changed,
            session_dates=EXPECTED_SESSIONS,
            expected_calendar_sessions_sha256=evidence["calendar_hash"],
            candidate_sha256=evidence["candidate_hash"],
            corpus_universe_sha256=evidence["universe_hash"],
            expected_event_bindings=evidence["events"][:1],
            expected_prior_prefix_sha256=None,
            expected_prior_tip_sha256=None,
        )


def test_earlier_prefix_hash_has_no_future_event_or_fold_context(evidence) -> None:
    prefix_1_hash = prediction_prefix_sha256(
        evidence["prefix"], through_sequence_number=1
    )
    future_event = _event(
        "0000320193-19-000099",
        "2019-01-02",
        fold_id=INTERMEDIATE_FOLD_ID,
        salt="future",
    )
    future_event["stage"] = "intermediate"
    future_spec_a = _available(
        future_event,
        semantic_probability=0.2,
        semantic_edge=-0.1,
        ablation_probability=0.3,
        ablation_edge=-0.2,
    )
    future_event_b = copy.deepcopy(future_event)
    future_event_b["extraction_identity_sha256"] = _hash("future-b:extraction")
    future_event_b["market_prefix_chain_identity_sha256"] = _hash(
        "future-b:market-prefix-chain"
    )
    future_event_b["market_feature_row_sha256"] = _hash("future-b:market-row")
    future_spec_b = _available(
        future_event_b,
        semantic_probability=0.9,
        semantic_edge=0.2,
        ablation_probability=0.8,
        ablation_edge=0.1,
    )
    fold_2_a = _fold("2018-12-31", "2018-12-28", "future-a")
    fold_2_b = _fold("2018-12-31", "2018-12-28", "future-b")
    first_prefix = append_prediction_row(
        None,
        evidence["specs"][0],
        current_event_binding=evidence["events"][0],
        current_fold_context=evidence["fold_contexts"]["fold_1"],
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        candidate_sha256=evidence["candidate_hash"],
        corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=evidence["events"][:1],
        expected_prior_prefix_sha256=None,
        expected_prior_tip_sha256=None,
    )
    finals = []
    for spec, fold_context, appended_event in (
        (future_spec_a, fold_2_a, future_event),
        (future_spec_b, fold_2_b, future_event_b),
    ):
        finals.append(
            append_prediction_row(
                first_prefix,
                spec,
                current_event_binding=appended_event,
                current_fold_context=fold_context,
                session_dates=EXPECTED_SESSIONS,
                expected_calendar_sessions_sha256=evidence["calendar_hash"],
                candidate_sha256=evidence["candidate_hash"],
                corpus_universe_sha256=evidence["universe_hash"],
                expected_event_bindings=[evidence["events"][0], appended_event],
                expected_prior_prefix_sha256=first_prefix[
                    "prediction_prefix_sha256"
                ],
                expected_prior_tip_sha256=first_prefix["tip_sha256"],
            )
        )
    assert first_prefix["prediction_prefix_sha256"] == prefix_1_hash
    assert prediction_prefix_sha256(finals[0], through_sequence_number=1) == prefix_1_hash
    assert prediction_prefix_sha256(finals[1], through_sequence_number=1) == prefix_1_hash
    assert finals[0]["prediction_prefix_sha256"] != finals[1]["prediction_prefix_sha256"]


def test_v3_feature_identity_and_scheduled_state_are_explicit_and_legacy_fails(
    evidence,
) -> None:
    assert PREDICTION_ROW_SCHEMA_VERSION.endswith("-v3")
    assert PREDICTION_PREFIX_SCHEMA_VERSION.endswith("-v3")
    assert evidence["prefix"]["schema_version"] == PREDICTION_PREFIX_SCHEMA_VERSION
    assert all(
        row["schema_version"] == PREDICTION_ROW_SCHEMA_VERSION
        for row in evidence["prefix"]["rows"]
    )

    legacy_event = copy.deepcopy(evidence["events"][0])
    legacy_spec = copy.deepcopy(evidence["specs"][0])
    for value in (legacy_event, legacy_spec):
        value.pop("extraction_identity_sha256")
        value.pop("market_prefix_chain_identity_sha256")
        value.pop("market_feature_row_sha256")
        value["extraction_output_sha256"] = _hash("legacy-output-only")

    with pytest.raises(
        SecFilingGemmaContractError,
        match="extraction_identity_sha256.*market_feature_row_sha256",
    ):
        append_prediction_row(
            None,
            legacy_spec,
            current_event_binding=legacy_event,
            current_fold_context=evidence["fold_contexts"]["fold_1"],
            session_dates=EXPECTED_SESSIONS,
            expected_calendar_sessions_sha256=evidence["calendar_hash"],
            candidate_sha256=evidence["candidate_hash"],
            corpus_universe_sha256=evidence["universe_hash"],
            expected_event_bindings=[legacy_event],
            expected_prior_prefix_sha256=None,
            expected_prior_tip_sha256=None,
        )

    legacy_prefix = copy.deepcopy(evidence["prefix"])
    legacy_prefix["schema_version"] = (
        "aapl-sec-gemma-pre-label-prediction-prefix-v2"
    )
    with pytest.raises(SecFilingGemmaContractError, match="schema_version"):
        validate_prediction_prefix(
            legacy_prefix, **evidence["validation_kwargs"]
        )


@pytest.mark.parametrize(
    "identity_field",
    [
        "extraction_identity_sha256",
        "market_prefix_chain_identity_sha256",
        "market_feature_row_sha256",
    ],
)
def test_prediction_row_cannot_substitute_a_feature_identity(
    evidence, identity_field: str
) -> None:
    changed = copy.deepcopy(evidence["prefix"])
    changed["rows"][0][identity_field] = _hash(f"substituted:{identity_field}")
    _refresh_prefix_container(changed)
    with pytest.raises(
        SecFilingGemmaContractError,
        match="bound to another external event",
    ):
        validate_prediction_prefix(changed, **evidence["validation_kwargs"])


def test_cross_prefix_tampering_omission_and_reordering_fail(evidence) -> None:
    changed = copy.deepcopy(evidence["prefix"])
    changed["rows"][1]["prior_prediction_prefix_sha256"] = _hash("other-prefix")
    _refresh_prefix_container(changed)
    with pytest.raises(SecFilingGemmaContractError, match="wrong prior prefix"):
        validate_prediction_prefix(changed, **evidence["validation_kwargs"])

    omitted = copy.deepcopy(evidence["prefix"])
    omitted["rows"].pop(1)
    with pytest.raises(SecFilingGemmaContractError):
        validate_prediction_prefix(omitted, **evidence["validation_kwargs"])

    reordered = copy.deepcopy(evidence["prefix"])
    reordered["rows"][0], reordered["rows"][1] = (
        reordered["rows"][1],
        reordered["rows"][0],
    )
    with pytest.raises(SecFilingGemmaContractError):
        validate_prediction_prefix(reordered, **evidence["validation_kwargs"])


def test_coherently_rebuilt_omission_cannot_validate_or_seal_as_full_universe(
    evidence,
) -> None:
    omitted_events = [
        evidence["events"][0],
        evidence["events"][2],
        evidence["events"][3],
    ]
    omitted_specs = [
        evidence["specs"][0],
        evidence["specs"][2],
        evidence["specs"][3],
    ]
    rebuilt = build_prediction_ledger(
        omitted_specs,
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        candidate_sha256=evidence["candidate_hash"],
        corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=omitted_events,
        fold_contexts=evidence["fold_contexts"],
    )
    omitted_summary = validate_prediction_prefix(
        rebuilt,
        expected_event_bindings=omitted_events,
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        expected_candidate_sha256=evidence["candidate_hash"],
        expected_corpus_universe_sha256=evidence["universe_hash"],
    )
    assert rebuilt["corpus_universe_sha256"] == evidence["prefix"][
        "corpus_universe_sha256"
    ]
    assert rebuilt["event_sequence_sha256"] != evidence["prefix"][
        "event_sequence_sha256"
    ]
    rebuilt_checksums = {
        prefix_hash: _tagged(f"rebuilt:{index}")
        for index, prefix_hash in enumerate(
            omitted_summary["prefix_ancestry_sha256s"]
        )
    }
    with pytest.raises(SecFilingGemmaContractError, match="omits or adds"):
        build_prelabel_seal_ledger(
            rebuilt,
            external_artifact_checksums_by_prefix_sha256=rebuilt_checksums,
            session_dates=EXPECTED_SESSIONS,
            expected_calendar_sessions_sha256=evidence["calendar_hash"],
            expected_candidate_sha256=evidence["candidate_hash"],
            expected_corpus_universe_sha256=evidence["universe_hash"],
            expected_event_bindings=evidence["events"],
        )


def test_all_candidate_variant_states_and_active_episode_cannot_extend(evidence) -> None:
    rows = evidence["prefix"]["rows"]
    first, second, unavailable, after_exit = rows
    for row in rows:
        assert set(row["candidate_policy_input_states"]) == set(CANDIDATE_IDS)
        assert set(row["candidate_policy_output_states"]) == set(CANDIDATE_IDS)
        for candidate_id in CANDIDATE_IDS:
            assert set(row["candidate_policy_input_states"][candidate_id]) == set(
                MODEL_VARIANTS
            )
            assert set(row["candidate_policy_output_states"][candidate_id]) == set(
                MODEL_VARIANTS
            )
        assert "candidate_policy_state_sha256" not in row

    for candidate_id in CANDIDATE_IDS:
        for variant in MODEL_VARIANTS:
            assert first["raw_gate_signals"][candidate_id][variant] == "CASH"
            assert (
                first["effective_episode_actions"][candidate_id][variant]
                == "START_CASH_EPISODE"
            )
            assert first["candidate_policy_input_states"][candidate_id][variant] == {
                "position_at_decision_close": "LONG",
                "episode_phase": "INACTIVE",
                "episode_origin_decision_session": None,
                "episode_fill_session": None,
                "episode_exit_session": None,
            }
            assert first["candidate_policy_output_states"][candidate_id][variant][
                "position_at_decision_close"
            ] == "LONG"
            assert first["candidate_policy_output_states"][candidate_id][variant][
                "episode_phase"
            ] == "SCHEDULED"
            first_exit = first["candidate_policy_output_states"][candidate_id][
                variant
            ]["episode_exit_session"]
            assert second["candidate_policy_input_states"][candidate_id][variant][
                "position_at_decision_close"
            ] == "CASH"
            assert second["candidate_policy_input_states"][candidate_id][variant][
                "episode_phase"
            ] == "ACTIVE"
            assert (
                second["effective_episode_actions"][candidate_id][variant]
                == "HOLD_EXISTING_CASH_EPISODE"
            )
            assert (
                second["candidate_policy_output_states"][candidate_id][variant][
                    "episode_exit_session"
                ]
                == first_exit
            )
            assert (
                unavailable["candidate_policy_output_states"][candidate_id][variant][
                    "episode_exit_session"
                ]
                == first_exit
            )
            assert (
                after_exit["candidate_policy_input_states"][candidate_id][variant][
                    "position_at_decision_close"
                ]
                == "LONG"
            )

    extended = copy.deepcopy(evidence["prefix"])
    state = extended["rows"][1]["candidate_policy_output_states"][CANDIDATE_IDS[0]][
        "semantic"
    ]
    state.update(
        {
            "episode_origin_decision_session": extended["rows"][1][
                "decision_session"
            ],
            "episode_fill_session": extended["rows"][1]["fill_session"],
            "episode_exit_session": extended["rows"][1]["cash_exit_session"],
        }
    )
    extended["rows"][1]["candidate_policy_output_states_sha256"] = canonical_sha256(
        extended["rows"][1]["candidate_policy_output_states"]
    )
    _refresh_prefix_container(extended)
    with pytest.raises(SecFilingGemmaContractError, match="extended an existing episode"):
        validate_prediction_prefix(extended, **evidence["validation_kwargs"])


def test_policy_state_is_scheduled_then_active_and_long_at_exact_exit_close(
    evidence,
) -> None:
    origin = "2005-03-01"
    fill = _next_session(origin)
    exit_session = _next_session(origin, LABEL_MATURITY_OFFSET)
    events = [
        _event("0000320193-05-000061", origin, salt="timing-origin"),
        _event("0000320193-05-000062", origin, salt="timing-same-close"),
        _event("0000320193-05-000063", fill, salt="timing-fill"),
        _event("0000320193-05-000064", exit_session, salt="timing-exit"),
    ]
    specs = [
        _available(
            events[0],
            semantic_probability=0.9,
            semantic_edge=0.1,
            ablation_probability=0.9,
            ablation_edge=0.1,
        ),
        *[
            _available(
                event,
                semantic_probability=0.1,
                semantic_edge=-0.1,
                ablation_probability=0.1,
                ablation_edge=-0.1,
            )
            for event in events[1:]
        ],
    ]
    prefix = build_prediction_ledger(
        specs,
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        candidate_sha256=evidence["candidate_hash"],
        corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=events,
        fold_contexts=evidence["fold_contexts"],
    )
    validate_prediction_prefix(
        prefix,
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        expected_candidate_sha256=evidence["candidate_hash"],
        expected_corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=events,
    )

    scheduled, still_scheduled, active, exact_exit = prefix["rows"]
    for candidate_id in CANDIDATE_IDS:
        for variant in MODEL_VARIANTS:
            assert scheduled["candidate_policy_output_states"][candidate_id][variant][
                "position_at_decision_close"
            ] == "LONG"
            assert scheduled["candidate_policy_output_states"][candidate_id][variant][
                "episode_phase"
            ] == "SCHEDULED"
            assert still_scheduled["candidate_policy_input_states"][candidate_id][
                variant
            ]["position_at_decision_close"] == "LONG"
            assert still_scheduled["candidate_policy_input_states"][candidate_id][
                variant
            ]["episode_phase"] == "SCHEDULED"
            assert (
                still_scheduled["effective_episode_actions"][candidate_id][variant]
                == "KEEP_SCHEDULED_CASH_EPISODE"
            )
            assert active["candidate_policy_input_states"][candidate_id][variant][
                "position_at_decision_close"
            ] == "CASH"
            assert active["candidate_policy_input_states"][candidate_id][variant][
                "episode_phase"
            ] == "ACTIVE"
            assert (
                active["effective_episode_actions"][candidate_id][variant]
                == "HOLD_EXISTING_CASH_EPISODE"
            )
            assert exact_exit["candidate_policy_input_states"][candidate_id][variant] == {
                "position_at_decision_close": "LONG",
                "episode_phase": "INACTIVE",
                "episode_origin_decision_session": None,
                "episode_fill_session": None,
                "episode_exit_session": None,
            }
            assert (
                exact_exit["effective_episode_actions"][candidate_id][variant]
                == "STAY_LONG"
            )


def test_unavailable_row_has_no_probability_or_gate_claims_but_is_sealed_and_labeled(
    evidence,
) -> None:
    row = evidence["prefix"]["rows"][2]
    assert row["prediction_status"] == UNAVAILABLE_PREDICTION_STATUS
    assert row["unavailable_reason"] == "missing_required_market_features"
    assert row["unavailable_fail_safe_action_semantics"] == "NO_NEW_EPISODE"
    for identity_field in (
        "extraction_identity_sha256",
        "market_prefix_chain_identity_sha256",
        "market_feature_row_sha256",
    ):
        assert row[identity_field] == evidence["events"][2][identity_field]
    for field in (
        "semantic_cash_probability_hex",
        "semantic_expected_edge_hex",
        "ablation_cash_probability_hex",
        "ablation_expected_edge_hex",
        "raw_gate_signals",
        "raw_gate_signals_sha256",
    ):
        assert row[field] is None
    assert row["effective_episode_actions_sha256"] == canonical_sha256(
        row["effective_episode_actions"]
    )
    for candidate_id in CANDIDATE_IDS:
        for variant in MODEL_VARIANTS:
            assert (
                row["effective_episode_actions"][candidate_id][variant]
                == "HOLD_EXISTING_CASH_EPISODE"
            )
    seal_entry = evidence["seal"]["entries"][2]
    release_entry = evidence["release"]["entries"][2]
    assert seal_entry["prediction_row_sha256"] == row["prediction_row_sha256"]
    assert release_entry["prediction_row_sha256"] == row["prediction_row_sha256"]
    assert release_entry["prediction_status"] == UNAVAILABLE_PREDICTION_STATUS
    assert release_entry["unavailable_reason"] == row["unavailable_reason"]

    invalid = copy.deepcopy(evidence["specs"][2])
    invalid["semantic_cash_probability"] = 0.5
    with pytest.raises(SecFilingGemmaContractError, match="cannot claim"):
        append_prediction_row(
            None,
            invalid,
            current_event_binding=evidence["events"][2],
            current_fold_context=evidence["fold_contexts"]["fold_1"],
            session_dates=EXPECTED_SESSIONS,
            expected_calendar_sessions_sha256=evidence["calendar_hash"],
            candidate_sha256=evidence["candidate_hash"],
            corpus_universe_sha256=evidence["universe_hash"],
            expected_event_bindings=[evidence["events"][2]],
            expected_prior_prefix_sha256=None,
            expected_prior_tip_sha256=None,
        )


def test_unavailable_event_has_exact_mixed_per_candidate_effective_actions(
    evidence,
) -> None:
    first_event = _event(
        "0000320193-05-000050", "2005-03-01", salt="mixed-first"
    )
    unavailable_event = _event(
        "0000320193-05-000051", "2005-03-08", salt="mixed-unavailable"
    )
    mixed_signal = _available(
        first_event,
        semantic_probability=0.53,
        semantic_edge=0.001,
        ablation_probability=0.56,
        ablation_edge=0.003,
    )
    prefix_1 = append_prediction_row(
        None,
        mixed_signal,
        current_event_binding=first_event,
        current_fold_context=evidence["fold_contexts"]["fold_1"],
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        candidate_sha256=evidence["candidate_hash"],
        corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=[first_event],
        expected_prior_prefix_sha256=None,
        expected_prior_tip_sha256=None,
    )
    prefix_2 = append_prediction_row(
        prefix_1,
        _unavailable(
            unavailable_event, "missing_required_extraction_features"
        ),
        current_event_binding=unavailable_event,
        current_fold_context=evidence["fold_contexts"]["fold_1"],
        session_dates=EXPECTED_SESSIONS,
        expected_calendar_sessions_sha256=evidence["calendar_hash"],
        candidate_sha256=evidence["candidate_hash"],
        corpus_universe_sha256=evidence["universe_hash"],
        expected_event_bindings=[first_event, unavailable_event],
        expected_prior_prefix_sha256=prefix_1["prediction_prefix_sha256"],
        expected_prior_tip_sha256=prefix_1["tip_sha256"],
    )
    row = prefix_2["rows"][1]
    positions: set[str] = set()
    for candidate_id in CANDIDATE_IDS:
        for variant in MODEL_VARIANTS:
            input_state = row["candidate_policy_input_states"][candidate_id][variant]
            output_state = row["candidate_policy_output_states"][candidate_id][variant]
            positions.add(input_state["position_at_decision_close"])
            expected_action = {
                "INACTIVE": "STAY_LONG",
                "SCHEDULED": "KEEP_SCHEDULED_CASH_EPISODE",
                "ACTIVE": "HOLD_EXISTING_CASH_EPISODE",
            }[input_state["episode_phase"]]
            assert row["effective_episode_actions"][candidate_id][variant] == expected_action
            assert output_state == input_state
    assert positions == {"LONG", "CASH"}
    assert row["unavailable_fail_safe_action_semantics"] == "NO_NEW_EPISODE"
    assert row["raw_gate_signals"] is None

    forged = copy.deepcopy(prefix_2)
    target_candidate = next(
        candidate_id
        for candidate_id in CANDIDATE_IDS
        if forged["rows"][1]["candidate_policy_input_states"][candidate_id][
            "semantic"
        ]["position_at_decision_close"]
        == "LONG"
    )
    forged["rows"][1]["effective_episode_actions"][target_candidate][
        "semantic"
    ] = "HOLD_EXISTING_CASH_EPISODE"
    forged["rows"][1]["effective_episode_actions_sha256"] = canonical_sha256(
        forged["rows"][1]["effective_episode_actions"]
    )
    _refresh_prefix_container(forged)
    mixed_validation_kwargs = {
        **evidence["validation_kwargs"],
        "expected_event_bindings": [first_event, unavailable_event],
    }
    with pytest.raises(SecFilingGemmaContractError, match="current policy states"):
        validate_prediction_prefix(forged, **mixed_validation_kwargs)


def test_prediction_rows_bind_exact_timeline_and_contain_no_outcomes(evidence) -> None:
    first = evidence["prefix"]["rows"][0]
    assert first["market_feature_cutoff_session"] == first["decision_session"]
    assert first["fill_session"] == _next_session(first["decision_session"])
    assert first["cash_exit_session"] == _next_session(
        first["decision_session"], LABEL_MATURITY_OFFSET
    )
    assert first["label_maturity_session"] == first["cash_exit_session"]
    assert first["horizon_sessions"] == HORIZON_SESSIONS
    assert first["semantic_cash_probability_hex"] == float(0.9).hex()
    assert "stage_context" not in first
    assert "stage_context_sha256" not in first
    assert first["fold_context_sha256"] == canonical_sha256(first["fold_context"])
    assert evidence["prefix"]["initial_event_sequence_sha256"] == canonical_sha256(
        evidence["events"][:1]
    )
    assert evidence["prefix"]["event_sequence_sha256"] == canonical_sha256(
        evidence["events"]
    )
    for index, row in enumerate(evidence["prefix"]["rows"], start=1):
        assert row["event_sequence_sha256"] == canonical_sha256(
            evidence["events"][:index]
        )
    assert {
        "outcome",
        "return",
        "score",
        "gate_result",
        "cash_beats_long_10bps",
    }.isdisjoint(first)

    wrong = copy.deepcopy(evidence["prefix"])
    wrong["rows"][0]["market_feature_cutoff_session"] = wrong["rows"][0][
        "fill_session"
    ]
    _refresh_prefix_container(wrong)
    with pytest.raises(SecFilingGemmaContractError, match="completed decision-session"):
        validate_prediction_prefix(wrong, **evidence["validation_kwargs"])


def test_external_prelabel_seals_are_exact_and_make_no_historic_time_claim(evidence) -> None:
    seal = evidence["seal"]
    summary = validate_prelabel_seal_ledger(
        seal,
        prediction_prefix=evidence["prefix"],
        external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
        expected_seal_ledger_sha256=seal["prelabel_seal_ledger_sha256"],
        expected_tip_sha256=seal["tip_sha256"],
        **evidence["validation_kwargs"],
    )
    assert summary["sealed_prediction_count"] == 4
    assert seal["event_sequence_sha256"] == evidence["prefix"][
        "event_sequence_sha256"
    ]
    for entry in seal["entries"]:
        assert entry["seal_protocol_phase"] == "pre_label_before_outcome_access"
        assert entry["prediction_artifact_checksum_sha256"].startswith("sha256:")
        assert "sealed_session" not in entry
        assert "sealed_at" not in entry

    wrong_checksum = copy.deepcopy(seal)
    wrong_checksum["entries"][1]["prediction_artifact_checksum_sha256"] = _tagged(
        "wrong"
    )
    _rehash_seal(wrong_checksum)
    with pytest.raises(SecFilingGemmaContractError, match="checksum-mismatched"):
        validate_prelabel_seal_ledger(
            wrong_checksum,
            prediction_prefix=evidence["prefix"],
            external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
            **evidence["validation_kwargs"],
        )

    cross_prefix = copy.deepcopy(seal)
    cross_prefix["entries"][0]["prediction_prefix_sha256"] = cross_prefix[
        "entries"
    ][1]["prediction_prefix_sha256"]
    _rehash_seal(cross_prefix)
    with pytest.raises(SecFilingGemmaContractError, match="cross-prefix"):
        validate_prelabel_seal_ledger(
            cross_prefix,
            prediction_prefix=evidence["prefix"],
            external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
            **evidence["validation_kwargs"],
        )


def test_label_maturity_must_strictly_precede_as_of_decision(evidence) -> None:
    first = evidence["prefix"]["rows"][0]
    same_session = build_label_release_ledger(
        evidence["prefix"],
        evidence["seal"],
        as_of_decision_session=first["label_maturity_session"],
        labels_by_prediction_sha256={},
        external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
        **evidence["validation_kwargs"],
    )
    assert same_session["release_count"] == 0
    validate_label_release_ledger(
        same_session,
        prediction_prefix=evidence["prefix"],
        prelabel_seal_ledger=evidence["seal"],
        labels_by_prediction_sha256={},
        external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
        **evidence["validation_kwargs"],
    )

    first_label = {
        first["prediction_row_sha256"]: evidence["labels"][
            first["prediction_row_sha256"]
        ]
    }
    with pytest.raises(SecFilingGemmaContractError, match="premature"):
        build_label_release_ledger(
            evidence["prefix"],
            evidence["seal"],
            as_of_decision_session=first["label_maturity_session"],
            labels_by_prediction_sha256=first_label,
            external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
            **evidence["validation_kwargs"],
        )

    next_session = _next_session(first["label_maturity_session"])
    eligible = build_label_release_ledger(
        evidence["prefix"],
        evidence["seal"],
        as_of_decision_session=next_session,
        labels_by_prediction_sha256=first_label,
        external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
        **evidence["validation_kwargs"],
    )
    assert eligible["release_count"] == 1
    entry = eligible["entries"][0]
    assert entry["release_session"] == next_session
    assert entry["first_eligible_prediction_session"] == next_session
    assert entry["label_maturity_session"] < entry["as_of_decision_session"]


def test_label_release_exact_binding_strict_target_and_adversarial_failures(evidence) -> None:
    release = evidence["release"]
    summary = validate_label_release_ledger(
        release,
        prediction_prefix=evidence["prefix"],
        prelabel_seal_ledger=evidence["seal"],
        labels_by_prediction_sha256=evidence["labels"],
        external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
        expected_label_release_ledger_sha256=release[
            "label_release_ledger_sha256"
        ],
        expected_tip_sha256=release["tip_sha256"],
        **evidence["validation_kwargs"],
    )
    assert summary["release_count"] == 4
    assert release["event_sequence_sha256"] == evidence["prefix"][
        "event_sequence_sha256"
    ]
    assert release["entries"][0]["cost_bps"] == BRIER_TARGET_COST_BPS
    assert release["entries"][0]["cash_beats_long_10bps"] is False
    assert release["entries"][1]["cash_beats_long_10bps"] is True

    wrong_checksum = copy.deepcopy(release)
    wrong_checksum["entries"][0]["prediction_artifact_checksum_sha256"] = _tagged(
        "wrong-label-checksum"
    )
    _rehash_release(wrong_checksum)
    with pytest.raises(SecFilingGemmaContractError, match="checksum-mismatched"):
        validate_label_release_ledger(
            wrong_checksum,
            prediction_prefix=evidence["prefix"],
            prelabel_seal_ledger=evidence["seal"],
            labels_by_prediction_sha256=evidence["labels"],
            external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
            **evidence["validation_kwargs"],
        )

    duplicate = copy.deepcopy(release)
    duplicate["entries"][1] = copy.deepcopy(duplicate["entries"][0])
    _rehash_release(duplicate)
    with pytest.raises(SecFilingGemmaContractError, match="sequence changed"):
        validate_label_release_ledger(
            duplicate,
            prediction_prefix=evidence["prefix"],
            prelabel_seal_ledger=evidence["seal"],
            labels_by_prediction_sha256=evidence["labels"],
            external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
            **evidence["validation_kwargs"],
        )

    wrong_row = copy.deepcopy(release)
    wrong_row["entries"][0]["outcome_ledger_row_sha256"] = wrong_row["entries"][1][
        "outcome_ledger_row_sha256"
    ]
    _rehash_release(wrong_row)
    with pytest.raises(SecFilingGemmaContractError, match="cross-row"):
        validate_label_release_ledger(
            wrong_row,
            prediction_prefix=evidence["prefix"],
            prelabel_seal_ledger=evidence["seal"],
            labels_by_prediction_sha256=evidence["labels"],
            external_artifact_checksums_by_prefix_sha256=evidence["checksums"],
            **evidence["validation_kwargs"],
        )


def test_import_is_effect_free_and_has_no_io_clients() -> None:
    forbidden = {
        "os",
        "pathlib",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "subprocess",
        "pandas",
        "numpy",
    }
    sys.modules.pop("agent_benchmark.sec_filing_gemma_prediction_evidence", None)
    module = importlib.import_module(
        "agent_benchmark.sec_filing_gemma_prediction_evidence"
    )
    assert forbidden.isdisjoint(module.__dict__)
    for function_name in (
        "append_prediction_row",
        "validate_prediction_prefix",
        "build_prelabel_seal_ledger",
        "validate_prelabel_seal_ledger",
        "build_label_release_ledger",
        "validate_label_release_ledger",
    ):
        parameter = inspect.signature(getattr(module, function_name)).parameters[
            "expected_event_bindings"
        ]
        assert parameter.default is inspect.Parameter.empty
