from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.chronological_exhaustion_experiment as experiment
from agent_benchmark.chronological_exhaustion_expert import (
    CAUSAL_ONLINE_MODE,
    build_chronological_exhaustion_forecast,
)
from agent_benchmark.deterministic_aapl import EvaluationPeriod


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.linspace(100.0, 130.0, len(index))
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": values * 2.0,
            "qqq_adj_close": values * 1.5,
        },
        index=index,
    )


def _coverage(index: pd.DatetimeIndex) -> dict[str, object]:
    payload = "".join(f"{value.date().isoformat()}\n" for value in index).encode(
        "ascii"
    )
    return {
        "first_session": index.min().date().isoformat(),
        "last_session": index.max().date().isoformat(),
        "observations": len(index),
        "date_sequence_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _duckdb_bounded_hash(path) -> str:
    import duckdb

    connection = duckdb.connect(":memory:")
    try:
        raw = connection.execute(experiment.PRICE_QUERY, [str(path)]).fetchdf()
    finally:
        connection.close()
    canonical = experiment.canonical_context_frame(raw)
    return experiment._sha256(
        experiment._frame_csv_bytes(canonical.reset_index(names="date"))
    )


def test_episode_edge_uses_exact_two_sided_log_friction():
    frame = _frame(pd.bdate_range("2018-01-02", periods=6))
    frame.iloc[2, frame.columns.get_loc("aapl_open")] = 110.0
    frame.iloc[3, frame.columns.get_loc("aapl_open")] = 100.0
    target = pd.Series(1.0, index=frame.index)
    target.iloc[1] = 0.0

    rows = experiment._episode_rows(
        frame,
        target,
        start=frame.index[0],
        end=frame.index[-1],
        cost_bps=10.0,
    )

    assert len(rows) == 1
    expected = math.log(110.0 / 100.0) + math.log(0.999 / 1.001)
    assert rows[0]["net_active_log_edge"] == pytest.approx(expected)
    assert rows[0]["win"] is True


def test_always_long_is_exactly_same_ledger_buy_and_hold():
    frame = _frame(pd.bdate_range("2017-01-02", periods=260))
    target = pd.Series(1.0, index=frame.index)
    result, strategy, benchmark, episodes = experiment._evaluate_policy(
        frame,
        target,
        periods=(EvaluationPeriod("2017", "2017-01-03", "2017-12-29"),),
        cost_bps=10.0,
    )

    assert result["total_active_log_edge"] == pytest.approx(0.0, abs=1e-15)
    assert result["comparison"]["relative_wealth_vs_aapl_buy_hold"] == pytest.approx(
        0.0, abs=1e-15
    )
    pd.testing.assert_series_equal(strategy["equity"], benchmark["equity"])
    assert episodes.empty
    assert result["no_leverage_proof"]["passed"] is True


def test_cross_year_episode_is_wholly_attributed_to_entry_open_year():
    index = pd.bdate_range("2022-12-28", "2023-01-06")
    frame = _frame(index)
    frame.loc[pd.Timestamp("2022-12-30"), "aapl_open"] = 110.0
    frame.loc[pd.Timestamp("2023-01-02"), "aapl_open"] = 90.0
    target = pd.Series(1.0, index=index)
    target.loc[pd.Timestamp("2022-12-29")] = 0.0

    result, _, _, episodes = experiment._evaluate_policy(
        frame,
        target,
        periods=(
            EvaluationPeriod("2022", "2022-12-29", "2022-12-31"),
            EvaluationPeriod("2023", "2023-01-01", "2023-01-06"),
        ),
        cost_bps=10.0,
    )

    assert len(episodes) == 1
    edge = float(episodes.iloc[0]["net_active_log_edge"])
    assert result["periods"]["2022"]["active_log_edge"] == pytest.approx(edge)
    assert result["periods"]["2023"]["active_log_edge"] == pytest.approx(0.0)
    assert result["total_active_log_edge"] == pytest.approx(edge)
    assert result["episode_ledger_identity_error"] == pytest.approx(0.0, abs=1e-12)


def test_administrative_stage_start_cannot_retroactively_trade_prior_signal(
    monkeypatch,
):
    frame = _frame(pd.bdate_range("2018-01-02", periods=180))
    forecast = build_chronological_exhaustion_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    stage_start = frame.index[150]
    prior = frame.index[149]
    forecast["combined_trusted_candidate_signal"] = False
    forecast.loc[[prior, stage_start], "combined_trusted_candidate_signal"] = True
    forecast["target_exposure"] = 1.0
    forecast.loc[prior, "target_exposure"] = 0.0

    synthetic_signals = pd.DataFrame(
        False,
        index=frame.index,
        columns=[
            "contextual_virtual_signal",
            "weak_trend_virtual_signal",
            "unfiltered_union_candidate_signal",
            "stage_outcome_available",
        ],
    )
    synthetic_signals["stage_outcome_available"] = True
    synthetic_signals.loc[
        [prior, stage_start], "unfiltered_union_candidate_signal"
    ] = True
    monkeypatch.setattr(
        experiment,
        "build_fixed_expert_signals",
        lambda _frame: synthetic_signals,
    )
    seen_targets: list[pd.Series] = []

    def fake_evaluate(frame, target, *, periods, cost_bps):
        seen_targets.append(target.copy())
        ledger = pd.DataFrame({"equity": [1000.0]})
        return ({}, ledger, ledger.copy(), pd.DataFrame())

    monkeypatch.setattr(experiment, "_evaluate_policy", fake_evaluate)
    experiment._evaluate_policy_set(
        frame,
        forecast,
        periods=(
            EvaluationPeriod(
                "stage", stage_start.date().isoformat(), frame.index[-1].date().isoformat()
            ),
        ),
        administrative_start=stage_start,
    )

    learner_target = seen_targets[0]
    assert (learner_target.loc[learner_target.index < stage_start] == 1.0).all()
    assert learner_target.loc[stage_start] == 0.0
    union_target = seen_targets[3]
    assert union_target.loc[prior] == 1.0
    assert union_target.loc[stage_start] == 0.0


def test_validation_rejects_bad_manifest_before_later_price_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(
        experiment,
        "_clean_git_identity",
        lambda root: {"branch": "test", "commit": "0" * 40},
    )

    def reject(**kwargs):
        raise experiment.ChronologicalExhaustionExperimentError("bad authority")

    loader_called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("later loader must not run")

    monkeypatch.setattr(experiment, "_validated_prior_manifest", reject)
    monkeypatch.setattr(experiment, "load_bounded_prices", forbidden_loader)

    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError, match="bad authority"
    ):
        experiment.run_validation(
            repo_root=tmp_path,
            price_artifact=tmp_path / "prices.csv",
            development_manifest=tmp_path / "manifest.json",
            output_dir=tmp_path / "out",
        )
    assert loader_called is False


def test_development_runner_resets_account_cooldown_at_2005(monkeypatch, tmp_path):
    frame = _frame(pd.bdate_range("2004-01-02", periods=400))
    forecast = build_chronological_exhaustion_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )

    class ReplayReached(Exception):
        pass

    def capture(*args, **kwargs):
        assert kwargs["administrative_start"] == experiment.DEVELOPMENT_START
        raise ReplayReached

    monkeypatch.setattr(experiment, "_clean_git_identity", lambda root: {})
    monkeypatch.setattr(experiment, "_tracked_input_identity", lambda *args: {})
    monkeypatch.setattr(
        experiment,
        "load_bounded_prices",
        lambda *args, **kwargs: (frame, {}),
    )
    monkeypatch.setattr(
        experiment,
        "build_chronological_exhaustion_forecast",
        lambda *args, **kwargs: forecast,
    )
    monkeypatch.setattr(experiment, "_evaluate_policy_set", capture)

    with pytest.raises(ReplayReached):
        experiment.run_development(
            repo_root=tmp_path,
            price_artifact=tmp_path / "prices.csv",
            output_dir=tmp_path / "out",
        )


def test_final_runner_resets_lifetime_account_cooldown_at_2005(
    monkeypatch, tmp_path
):
    frame = _frame(pd.bdate_range("2004-01-02", periods=400))
    forecast = build_chronological_exhaustion_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    starts: list[pd.Timestamp | None] = []

    def capture(*args, **kwargs):
        starts.append(kwargs.get("administrative_start"))
        return {}, {}, {}

    monkeypatch.setattr(experiment, "_clean_git_identity", lambda root: {})
    monkeypatch.setattr(
        experiment,
        "_validated_prior_manifest",
        lambda **kwargs: {"manifest_sha256": "sha256:parent"},
    )
    monkeypatch.setattr(experiment, "_require_dependency_continuity", lambda *a: None)
    monkeypatch.setattr(experiment, "_tracked_input_identity", lambda *args: {})
    monkeypatch.setattr(
        experiment,
        "load_bounded_prices",
        lambda *args, **kwargs: (frame, {}),
    )
    monkeypatch.setattr(experiment, "_require_source_continuity", lambda *a, **k: None)
    monkeypatch.setattr(
        experiment,
        "build_chronological_exhaustion_forecast",
        lambda *args, **kwargs: forecast,
    )
    monkeypatch.setattr(
        experiment, "_require_checkpoint_continuity", lambda *a, **k: None
    )
    monkeypatch.setattr(experiment, "_evaluate_policy_set", capture)
    monkeypatch.setattr(
        experiment,
        "apply_final_gates",
        lambda *args: {"strict_pass": False},
    )
    monkeypatch.setattr(experiment, "_checkpoint_from_forecast", lambda *a, **k: {})
    monkeypatch.setattr(experiment, "_stage_bundle", lambda **kwargs: {})

    experiment.run_final(
        repo_root=tmp_path,
        price_artifact=tmp_path / "prices.csv",
        validation_manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
    )

    assert starts == [
        experiment.FINAL_START,
        experiment.FINAL_START,
        experiment.DEVELOPMENT_START,
    ]


def test_bounded_loader_requires_a_physically_truncated_snapshot(
    monkeypatch, tmp_path
):
    index = pd.bdate_range("1999-03-10", "2018-12-31")
    monkeypatch.setitem(
        experiment.STAGE_SESSION_COVERAGE, "2018-12-31", _coverage(index)
    )
    frame = _frame(index).reset_index(names="date")
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setitem(
        experiment.STAGE_BOUNDED_RESULT_SHA256,
        "2018-12-31",
        _duckdb_bounded_hash(path),
    )

    loaded, provenance = experiment.load_bounded_prices(
        path,
        end=pd.Timestamp("2018-12-31"),
        required_last_session=pd.Timestamp("2018-12-31"),
    )

    assert loaded.index.max() == pd.Timestamp("2018-12-31")
    assert not (loaded.index > pd.Timestamp("2018-12-31")).any()
    assert provenance["bounded_last_date"] == "2018-12-31"
    assert provenance["physical_snapshot_has_later_rows"] is False
    assert provenance["rows_after_bound_returned"] is False
    assert "source_file_sha256" not in provenance


def test_future_filled_snapshot_is_rejected_instead_of_silently_filtered(tmp_path):
    index = pd.bdate_range("1999-03-10", "2019-01-04")
    path = tmp_path / "future_filled.csv"
    _frame(index).reset_index(names="date").to_csv(path, index=False)

    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="Physical stage snapshot does not end",
    ):
        experiment.load_bounded_prices(
            path,
            end=pd.Timestamp("2018-12-31"),
            required_last_session=pd.Timestamp("2018-12-31"),
        )


def test_physically_bounded_snapshot_rejects_revised_prices(monkeypatch, tmp_path):
    index = pd.bdate_range("1999-03-10", "2018-12-31")
    original = _frame(index).reset_index(names="date")
    path = tmp_path / "prices.csv"
    original.to_csv(path, index=False)
    approved_hash = _duckdb_bounded_hash(path)

    revised = original.copy()
    revised.loc[10, "aapl_close"] += 1.0
    revised.to_csv(path, index=False)
    monkeypatch.setitem(
        experiment.STAGE_SESSION_COVERAGE, "2018-12-31", _coverage(index)
    )
    monkeypatch.setitem(
        experiment.STAGE_BOUNDED_RESULT_SHA256,
        "2018-12-31",
        approved_hash,
    )

    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="prices do not match",
    ):
        experiment.load_bounded_prices(
            path,
            end=pd.Timestamp("2018-12-31"),
            required_last_session=pd.Timestamp("2018-12-31"),
        )


def test_source_continuity_uses_only_authorized_prefix_and_allows_future_append():
    index = pd.bdate_range("1999-03-10", "2019-01-04")
    frame = _frame(index)
    parent_prefix = frame.loc[:"2018-12-31"]
    parent = {
        "source_path": "C:/through_2018.csv",
        "source_provenance": {
            "source_type": "physically_bounded_local_csv_duckdb_query"
        },
        "bounded_result_sha256": experiment._prefix_sha256(
            parent_prefix, end=pd.Timestamp("2018-12-31")
        ),
    }
    provenance = {
        "source_path": "C:/through_2023.csv",
        "source_type": "physically_bounded_local_csv_duckdb_query",
    }

    # A later physically bounded snapshot may extend the committed parent
    # without changing its already-authorized historical prefix.
    experiment._require_source_continuity(
        frame,
        provenance,
        parent,
        parent_end=pd.Timestamp("2018-12-31"),
    )

    changed = frame.copy()
    changed.loc[pd.Timestamp("2018-12-28"), "aapl_close"] += 1.0
    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="historical price prefix changed",
    ):
        experiment._require_source_continuity(
            changed,
            provenance,
            parent,
            parent_end=pd.Timestamp("2018-12-31"),
        )


def test_later_stage_requires_exact_parent_dependency_hashes():
    current = {
        "tracked_dependency_sha256": {"runner.py": "sha256:abc"},
        "runtime_versions": {"python": "3.12.0"},
    }
    parent = {
        "git_identity": {
            "tracked_dependency_sha256": {"runner.py": "sha256:abc"},
            "runtime_versions": {"python": "3.12.0"},
        }
    }
    experiment._require_dependency_continuity(current, parent)

    parent["git_identity"]["tracked_dependency_sha256"]["runner.py"] = "sha256:def"
    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="dependencies changed",
    ):
        experiment._require_dependency_continuity(current, parent)

    parent["git_identity"]["tracked_dependency_sha256"] = {
        "runner.py": "sha256:abc"
    }
    parent["git_identity"]["runtime_versions"]["python"] = "3.13.0"
    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="dependencies changed",
    ):
        experiment._require_dependency_continuity(current, parent)


def test_parent_manifest_cannot_omit_its_payload_evidence():
    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="incomplete or unsafe",
    ):
        experiment._validated_payload_inventory({}, expected_stage="development")

    complete = {
        name: "sha256:" + "0" * 64
        for name in experiment.REQUIRED_PARENT_PAYLOADS["development"]
    }
    assert (
        experiment._validated_payload_inventory(
            complete, expected_stage="development"
        )
        == complete
    )

    complete["../escape"] = "sha256:" + "0" * 64
    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="incomplete or unsafe",
    ):
        experiment._validated_payload_inventory(
            complete, expected_stage="development"
        )


def test_manifest_flag_cannot_launder_a_failed_gate_report():
    manifest = {"stage_pass": True, "run_id": "run-1"}
    gate = {"passed": False, "gates": {"edge": False}}
    report = {
        "contract_version": experiment.CONTRACT_VERSION,
        "stage": "development",
        "run_id": "run-1",
        "gate_report": gate,
    }

    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="do not agree on a pass",
    ):
        experiment._require_pass_evidence_consistency(
            manifest, gate, report, expected_stage="development"
        )

    gate["passed"] = True
    experiment._require_pass_evidence_consistency(
        manifest, gate, report, expected_stage="development"
    )


def test_over_limit_bundle_is_not_promoted(tmp_path):
    run_dir = tmp_path / "run"

    def reject_promotion():
        raise experiment.ChronologicalExhaustionExperimentError("over time")

    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError, match="over time"
    ):
        experiment._seal_bundle(
            run_dir,
            {"report.json": b"{}\n"},
            before_promote=reject_promotion,
        )

    assert not run_dir.exists()
    assert not list(tmp_path.glob(".*.sealing"))


def test_regenerated_state_must_match_committed_parent_checkpoint(tmp_path):
    frame = _frame(pd.bdate_range("2017-01-02", periods=300))
    forecast = build_chronological_exhaustion_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    cutoff = frame.index[-1]
    checkpoint = experiment._checkpoint_from_forecast(
        forecast, cutoff=cutoff, learning_mode=CAUSAL_ONLINE_MODE
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_bytes(experiment._pretty_json_bytes(checkpoint))
    manifest_path = tmp_path / "stage_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    experiment._require_checkpoint_continuity(
        forecast,
        cutoff=cutoff,
        parent_manifest_path=manifest_path,
        checkpoint_filename=checkpoint_path.name,
    )

    checkpoint["experts"]["contextual"]["matured_count"] += 1
    checkpoint_path.write_bytes(experiment._pretty_json_bytes(checkpoint))
    with pytest.raises(
        experiment.ChronologicalExhaustionExperimentError,
        match="learner state does not match",
    ):
        experiment._require_checkpoint_continuity(
            forecast,
            cutoff=cutoff,
            parent_manifest_path=manifest_path,
            checkpoint_filename=checkpoint_path.name,
        )


def _validation_metrics(active_2022: float = 0.01):
    learner_periods = {
        str(year): {"active_log_edge": active_2022 if year == 2022 else 0.01}
        for year in range(2019, 2024)
    }
    learner = {
        "total_active_log_edge": 0.05,
        "periods": learner_periods,
        "cash_episode_count": 5,
        "mean_cash_episode_edge": 0.01,
        "median_cash_episode_edge": 0.01,
        "maximum_positive_episode_share": 0.30,
    }
    always = {
        "total_active_log_edge": 0.0,
        "comparison": {"relative_wealth_vs_aapl_buy_hold": 0.0},
    }
    return {
        name: {"learner": dict(learner), "always_long": dict(always)}
        for name, _ in experiment.COST_SCENARIOS
    }


def test_validation_gates_apply_to_both_costs_and_frozen_2022():
    passing = experiment.apply_validation_gates(_validation_metrics())
    assert passing["passed"] is True

    failing = _validation_metrics()
    failing["base_5bps"]["learner"]["periods"]["2022"][
        "active_log_edge"
    ] = -0.001
    report = experiment.apply_validation_gates(failing)
    assert report["passed"] is False
    assert "base_5bps_nonnegative_2022_active_log_edge" in report["failures"]


def _final_metrics(active_edge: float):
    learner = {
        "periods": {
            "2024": {"active_log_edge": active_edge},
            "2025": {"active_log_edge": active_edge},
            "2026_ytd": {"active_log_edge": active_edge},
        },
    }
    always = {
        "total_active_log_edge": 0.0,
        "comparison": {"relative_wealth_vs_aapl_buy_hold": 0.0},
    }
    return {
        name: {"learner": dict(learner), "always_long": dict(always)}
        for name, _ in experiment.COST_SCENARIOS
    }


def _lifetime_metrics(active_edge: float = 0.01):
    learner = {
        "comparison": {"relative_wealth_vs_aapl_buy_hold": 0.10},
        "periods": {
            str(year): {"active_log_edge": active_edge}
            for year in range(2005, 2026)
        },
    }
    always = {
        "total_active_log_edge": 0.0,
        "comparison": {"relative_wealth_vs_aapl_buy_hold": 0.0},
    }
    return {
        name: {"learner": dict(learner), "always_long": dict(always)}
        for name, _ in experiment.COST_SCENARIOS
    }


def test_final_gate_requires_preregistered_material_edge_not_rounding_noise():
    passing = experiment.apply_final_gates(
        _final_metrics(0.0011), _lifetime_metrics()
    )
    assert passing["strict_pass"] is True

    failing = experiment.apply_final_gates(
        _final_metrics(experiment.MIN_MATERIAL_ACTIVE_LOG_EDGE),
        _lifetime_metrics(),
    )
    assert failing["strict_pass"] is False
    assert (
        "strict_base_5bps_2024_material_active_log_edge"
        in failing["failures"]
    )


def test_final_gate_fails_closed_when_always_long_control_breaks():
    frozen = _final_metrics(0.01)
    frozen["base_5bps"]["always_long"]["total_active_log_edge"] = 0.01

    report = experiment.apply_final_gates(frozen, _lifetime_metrics())

    assert report["integrity_pass"] is False
    assert report["strict_pass"] is False
    assert report["passed"] is False
    assert "frozen_always_long_matches_buy_hold" in report["failures"]
