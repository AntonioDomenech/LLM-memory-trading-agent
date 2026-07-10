from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.direct_edge_experiment as experiment
from agent_benchmark.direct_edge_walkforward import (
    DIRECT_EDGE_CANDIDATES,
    candidate_cash_block_start_column,
    candidate_cash_target_column,
    five_session_cash_policy,
)


def _price_frame(*, include_2019: bool = False) -> pd.DataFrame:
    end = "2019-01-04" if include_2019 else "2018-12-31"
    index = pd.bdate_range("2004-01-02", end)
    position = np.arange(len(index), dtype=float)
    base = 50.0 * np.exp(position * 0.0002)
    return pd.DataFrame(
        {
            "aapl_open": base * 0.999,
            "aapl_close": base,
            "aapl_adj_close": base,
            "spy_adj_close": 100.0 * np.exp(position * 0.0001),
            "qqq_adj_close": 80.0 * np.exp(position * 0.00015),
        },
        index=index,
    )


def _context_frame(*, include_2019: bool = False) -> pd.DataFrame:
    end = "2019-01-04" if include_2019 else "2018-12-31"
    index = pd.bdate_range("2004-01-02", end)
    position = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "iwm_adj_close": 80.0 * np.exp(position * 0.0001),
            "vix_close": 20.0 + 2.0 * np.sin(position / 20.0),
            "tnx_close": 25.0 + 0.001 * position,
        },
        index=index,
    )


def _fake_predictions(index: pd.DatetimeIndex) -> pd.DataFrame:
    output = pd.DataFrame(index=index)
    output.index.name = "decision_date"
    output["price_only_cash_win_probability"] = 0.60
    output["price_only_expected_net_edge"] = 0.01
    output["price_only_baseline_cash_win_probability"] = 0.45
    output["price_only_baseline_mean_net_edge"] = 0.0
    output["market_sentiment_cash_win_probability"] = 0.61
    output["market_sentiment_expected_net_edge"] = 0.011
    output["market_sentiment_baseline_cash_win_probability"] = 0.45
    output["market_sentiment_baseline_mean_net_edge"] = 0.0
    output["price_features_ready"] = True
    output["sentiment_features_ready"] = True
    output["market_sentiment_used_price_fallback"] = False
    labels = np.ones(len(index), dtype=float)
    edges = np.full(len(index), 0.01)
    labels[-6:] = np.nan
    edges[-6:] = np.nan
    output["cash_beats_long_10bps"] = labels
    output["cash_active_log_edge_10bps"] = edges
    for family in ("price_only", "market_sentiment"):
        probability = output[f"{family}_cash_win_probability"].to_numpy(float)
        edge = output[f"{family}_expected_net_edge"].to_numpy(float)
        for candidate in DIRECT_EDGE_CANDIDATES:
            trigger = (
                (probability >= candidate.probability_gate)
                & (edge >= candidate.expected_edge_gate)
            )
            cash, starts = five_session_cash_policy(trigger)
            output[candidate_cash_target_column(family, candidate.candidate_id)] = cash
            output[
                candidate_cash_block_start_column(family, candidate.candidate_id)
            ] = starts
    return output


def _fake_ledger(frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    fill_index = frame.loc["2005-01-01":"2018-12-31"].index
    executed = target.shift(1).loc[fill_index].fillna(1.0).to_numpy(float)
    return pd.DataFrame(
        {
            "decision_date": fill_index.date.astype(str),
            "fill_date": fill_index.date.astype(str),
            "target_exposure": executed,
            "new_exposure_after_fill": executed,
            "holding_exposure_for_return": executed,
            "cash": np.where(executed == 0.0, 1000.0, 0.0),
            "shares": np.where(executed == 0.0, 0.0, 10.0),
            "margin_interest": 0.0,
            "trade_executed": False,
            "daily_return": 0.0,
        }
    )


def _score_payload(*, passed: bool, sentiment: bool, cost_bps: float) -> dict:
    metrics = {
        "minimum_fold_active_log_edge": 0.01 if not sentiment else 0.011,
        "total_active_log_edge": 0.10 if not sentiment else 0.11,
        "cash_episodes": 20,
        "brier_score": 0.20 if not sentiment else 0.19,
        "expected_edge_mae": 0.02 if not sentiment else 0.019,
    }
    return {
        "metrics": metrics,
        "gates": {"passed": passed, "checks": {"synthetic": passed}},
        "prediction_availability": {"oof_rows": 1},
        "cost_bps": cost_bps,
    }


def _rehash_artifact(run_dir: Path) -> None:
    manifest_path = run_dir / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["payload_sha256"]:
        manifest["payload_sha256"][name] = experiment._sha256_tagged(
            (run_dir / name).read_bytes()
        )
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = experiment._sha256_tagged(
        experiment._canonical_json_bytes(unsigned)
    )
    manifest_path.write_bytes(experiment._pretty_json_bytes(manifest))
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selection_manifest"] = manifest
    report_path.write_bytes(experiment._pretty_json_bytes(report))
    checksums = {
        path.name: experiment._sha256_hex(path.read_bytes())
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    (run_dir / "checksums.json").write_bytes(experiment._pretty_json_bytes(checksums))


def test_contract_has_eight_direct_candidates_and_explicit_source_exclusions():
    assert len(experiment.CANDIDATE_SPECS) == 8
    assert [item.candidate_id for item in experiment.CANDIDATE_SPECS[:4]] == [
        "price_only_p50_e0",
        "price_only_p55_e0",
        "price_only_p50_e25",
        "price_only_p55_e25",
    ]
    assert experiment.SOURCE_EXCLUSION_AUDITS["macro"]["included"] is False
    assert "vintage" in experiment.SOURCE_EXCLUSION_AUDITS["macro"]["finding"]
    assert experiment.SOURCE_EXCLUSION_AUDITS["news"]["included"] is False
    assert "timing" in experiment.SOURCE_EXCLUSION_AUDITS["news"]["finding"]


def test_any_injected_post_2018_price_or_context_row_is_rejected(tmp_path):
    with pytest.raises(experiment.DirectEdgeExperimentError, match="post-2018 price"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(include_2019=True),
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )
    with pytest.raises(experiment.DirectEdgeExperimentError, match="post-2018 context"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(),
            context_frame=_context_frame(include_2019=True),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )

    incomplete_future = _price_frame().copy()
    incomplete_future.loc[pd.Timestamp("2019-01-02")] = np.nan
    with pytest.raises(experiment.DirectEdgeExperimentError, match="post-2018 price"):
        experiment.run_development_from_inputs(
            price_frame=incomplete_future,
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )


def test_public_context_parquet_query_is_symbol_and_date_bounded(tmp_path):
    source = tmp_path / "context_daily.parquet"
    pd.DataFrame(
        {
            "date": ["2018-12-31", "2018-12-31", "2019-01-02"],
            "symbol": ["IWM", "SPY", "VIX"],
            "close": [100.0, 200.0, 30.0],
            "adj_close": [99.0, 199.0, 30.0],
        }
    ).to_parquet(source, index=False)
    frame, provenance = experiment.load_bounded_context_parquet(source)
    assert frame["symbol"].tolist() == ["IWM"]
    assert pd.to_datetime(frame["date"]).max() == pd.Timestamp("2018-12-31")
    assert provenance["physical_data_end"] == "2018-12-31"
    assert provenance["network_access"] is False
    assert provenance["source_file_sha256"].startswith("sha256:")


def _ablation_result(*, sentiment: bool) -> dict:
    return {
        "metrics": {
            "brier_score": 0.19 if sentiment else 0.20,
            "expected_edge_mae": 0.019 if sentiment else 0.020,
            "total_active_log_edge": 0.11 if sentiment else 0.10,
            "minimum_fold_active_log_edge": 0.011 if sentiment else 0.010,
        }
    }


def test_common_support_requires_all_four_improvements_at_both_costs():
    common = {}
    for family in ("price_only", "market_sentiment"):
        common[f"{family}_p50_e0"] = {
            scenario: _ablation_result(sentiment=family == "market_sentiment")
            for scenario, _ in experiment.COST_SCENARIOS
        }
    passed = experiment.common_support_ablation(common, gate_id="p50_e0")
    assert passed["passed"] is True
    failed = copy.deepcopy(common)
    failed["market_sentiment_p50_e0"]["stress_10bps"]["metrics"][
        "expected_edge_mae"
    ] = 0.021
    result = experiment.common_support_ablation(failed, gate_id="p50_e0")
    assert result["passed"] is False
    assert result["scenarios"]["stress_10bps"]["checks"][
        "strict_expected_edge_mae_improvement"
    ] is False


def _candidate_result(
    candidate_id: str,
    *,
    weakest: float,
    total: float,
    episodes: int,
    passed: bool = True,
) -> dict:
    metrics = {
        "minimum_fold_active_log_edge": weakest,
        "total_active_log_edge": total,
        "cash_episodes": episodes,
    }
    return {
        "candidate": {"candidate_id": candidate_id},
        "scenarios": {
            "base_5bps": {"metrics": dict(metrics)},
            "stress_10bps": {"metrics": dict(metrics)},
        },
        "passed": passed,
    }


def test_selection_uses_worst_fold_then_total_then_fewer_episodes_then_id():
    results = [
        _candidate_result("b", weakest=0.01, total=0.10, episodes=10),
        _candidate_result("a", weakest=0.011, total=0.09, episodes=20),
        _candidate_result("c", weakest=1.0, total=1.0, episodes=1, passed=False),
    ]
    assert experiment.select_development_candidate(results) == "a"
    assert experiment.select_development_candidate(
        [{**item, "passed": False} for item in results]
    ) is None


def test_timeout_fails_before_feature_or_artifact_work(tmp_path):
    readings = iter((0.0, experiment.RUN_TIME_LIMIT_SECONDS + 1.0))
    with pytest.raises(experiment.DirectEdgeExperimentTimeout, match="bounded input"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(),
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: next(readings),
            run_id="must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_rejected_run_seals_null_selection_both_costs_and_tamper_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    price = _price_frame()
    context = _context_frame()
    prediction_index = price.loc["2005-01-01":"2018-12-31"].index
    predictions = _fake_predictions(prediction_index)
    placeholder_features = pd.DataFrame(index=price.index)

    monkeypatch.setattr(
        experiment,
        "build_direct_edge_feature_label_frame",
        lambda price_frame, context_frame: placeholder_features,
    )
    monkeypatch.setattr(
        experiment,
        "build_pre2019_direct_edge_predictions",
        lambda feature_label: predictions,
    )
    monkeypatch.setattr(
        experiment,
        "_simulate",
        lambda frame, target, cost_bps: _fake_ledger(frame, target),
    )

    def failing_score(strategy, benchmark, *, prediction_inputs, cost_bps):
        del benchmark
        cash = prediction_inputs[6]
        starts = prediction_inputs[7]
        rebuilt = np.zeros(len(cash), dtype=float)
        for position in np.flatnonzero(starts == 1.0):
            rebuilt[position : position + 5] = 1.0
        assert np.array_equal(rebuilt, cash)
        # Family is observable from whether most rows are CASH in this stub;
        # the exact values do not matter because every gate is forced to fail.
        sentiment = bool((strategy["target_exposure"] == 0.0).mean() > 0.99)
        return _score_payload(passed=False, sentiment=sentiment, cost_bps=cost_bps)

    monkeypatch.setattr(experiment, "_score_strategy", failing_score)
    monkeypatch.setattr(
        experiment,
        "_refit_selected_models",
        lambda frame, selected, deadline: (
            {},
            {"performed": False, "reason": "no_development_candidate_passed"},
        ),
    )

    report = experiment.run_development_from_inputs(
        price_frame=price,
        context_frame=context,
        output_dir=tmp_path,
        clock=lambda: 0.0,
        run_id="synthetic-direct-edge",
    )
    run_dir = Path(report["artifact_dir"])
    assert report["selected_candidate_id"] is None
    assert report["development_pass"] is False
    verified = experiment.verify_development_artifact(run_dir)
    assert verified["verified"] is True
    assert verified["selected_candidate_id"] is None

    results = json.loads((run_dir / "candidate_results.json").read_text())
    assert len(results) == 8
    assert all(set(item["scenarios"]) == {"base_5bps", "stress_10bps"} for item in results)
    manifest = json.loads((run_dir / "selection_manifest.json").read_text())
    assert manifest["post_2018_market_data_accessed"] is False
    assert manifest["source_exclusion_audits"]["macro"]["included"] is False
    assert not list(run_dir.glob("final_*_model_through_2018.json"))
    targets = pd.read_csv(run_dir / "candidate_targets_2005_2018.csv")
    assert any(column.endswith("_cash_block_start") for column in targets.columns)

    path = run_dir / "candidate_results.json"
    original_results = path.read_bytes()
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(experiment.DirectEdgeExperimentError, match="checksum mismatch"):
        experiment.verify_development_artifact(run_dir)

    path.write_bytes(original_results)
    targets_path = run_dir / "candidate_targets_2005_2018.csv"
    targets = pd.read_csv(targets_path)
    target_column = "price_only_p50_e0_cash"
    targets.loc[0, target_column] = 1 - int(targets.loc[0, target_column])
    targets_path.write_bytes(
        experiment._frame_csv_bytes(targets, float_format="%.17g")
    )
    _rehash_artifact(run_dir)
    with pytest.raises(
        experiment.DirectEdgeExperimentError,
        match="full CASH target differs from prediction",
    ):
        experiment.verify_development_artifact(run_dir)

    now = [0.0]
    original_verify = experiment.verify_development_artifact

    def slow_verify(path):
        result = original_verify(path)
        now[0] = experiment.RUN_TIME_LIMIT_SECONDS + 1.0
        return result

    monkeypatch.setattr(experiment, "verify_development_artifact", slow_verify)
    with pytest.raises(
        experiment.DirectEdgeExperimentTimeout,
        match="completed development verification",
    ):
        experiment.run_development_from_inputs(
            price_frame=price,
            context_frame=context,
            output_dir=tmp_path,
            clock=lambda: now[0],
            run_id="late-direct-edge",
        )
    assert not (tmp_path / "late-direct-edge").exists()
    assert not list(tmp_path.glob(".late-direct-edge.*.sealing"))
