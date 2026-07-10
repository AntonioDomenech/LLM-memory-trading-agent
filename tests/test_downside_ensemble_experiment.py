from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.downside_ensemble_experiment as experiment
from agent_benchmark.downside_features import CFTC_FEATURE_COLUMNS, PRICE_FEATURE_COLUMNS
from agent_benchmark.downside_walkforward import (
    RISK_MULTIPLE_GATES,
    candidate_cash_target_column,
    five_session_cash_target,
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


def _fake_predictions(index: pd.DatetimeIndex) -> pd.DataFrame:
    output = pd.DataFrame(index=index)
    output.index.name = "decision_date"
    output["price_only_downside_probability"] = 0.20
    output["price_only_baseline_downside_probability"] = 0.10
    output["price_only_predicted_mean_clipped_return"] = 0.01
    output["price_cftc_downside_probability"] = 0.21
    output["price_cftc_effective_baseline_downside_probability"] = 0.10
    output["price_cftc_predicted_mean_clipped_return"] = 0.02
    output["price_features_ready"] = True
    output["cftc_features_finite"] = True
    output["price_cftc_used_fallback"] = False
    labels = np.zeros(len(index), dtype=float)
    labels[-6:] = np.nan
    output["crash_label_5session"] = labels
    for family in ("price_only", "price_cftc"):
        probability = output[f"{family}_downside_probability"].to_numpy(dtype=float)
        baseline_column = (
            "price_only_baseline_downside_probability"
            if family == "price_only"
            else "price_cftc_effective_baseline_downside_probability"
        )
        baseline = output[baseline_column].to_numpy(dtype=float)
        for gate in RISK_MULTIPLE_GATES:
            output[candidate_cash_target_column(family, gate)] = (
                five_session_cash_target(probability >= gate * baseline)
            )
    return output


def _fake_ledger(frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    fill_index = frame.loc["2005-01-01":"2018-12-31"].index
    executed = target.shift(1).loc[fill_index].to_numpy(dtype=float)
    return pd.DataFrame(
        {
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


def _passing_score(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    brier,
    cost_bps: float,
):
    del benchmark, brier
    cash_days = int((strategy["target_exposure"] == 0.0).sum())
    metrics = {
        "minimum_fold_active_log_edge": 0.01,
        "total_active_log_edge": 0.10,
        "brier_improvement": 0.05,
        "cash_days": cash_days,
    }
    return {
        "metrics": metrics,
        "gates": {"passed": True, "checks": {"synthetic": True}},
        "prediction_availability": {"oof_rows": 1},
        "cost_bps": cost_bps,
    }


def test_candidate_contract_uses_causal_relative_risk_and_records_rejected_preflight():
    assert len(experiment.CANDIDATE_SPECS) == 8
    assert [item.candidate_id for item in experiment.CANDIDATE_SPECS[:4]] == [
        "price_only_r110",
        "price_only_r120",
        "price_only_r130",
        "price_only_r140",
    ]
    assert all(
        "risk_multiple_gate" in item.to_dict()
        and item.to_dict()["predicted_mean_return_role"]
        == "diagnostic_only_not_a_trading_gate"
        for item in experiment.CANDIDATE_SPECS
    )
    assert (
        experiment.REJECTED_ABSOLUTE_TRIGGER_PREFLIGHT["observed_result"]
        == "zero_cash_targets_for_all_eight_candidates"
    )


def test_any_post_2018_price_or_cftc_report_is_rejected(tmp_path):
    with pytest.raises(experiment.DownsideEnsembleError, match="post-2018 price"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(include_2019=True),
            cot_records=(),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )

    from agent_benchmark.cftc_cot_policy import COTWeeklyRecord

    contaminated = COTWeeklyRecord(
        market="NASDAQ",
        report_date=pd.Timestamp("2019-01-01").date(),
        availability_date=pd.Timestamp("2019-01-09").date(),
        net_share=0.1,
    )
    with pytest.raises(experiment.DownsideEnsembleError, match="post-2018 CFTC"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(),
            cot_records=(contaminated,),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )


def _ablation_metrics(total: float, weakest: float) -> dict:
    return {"metrics": {"total_active_log_edge": total, "minimum_fold_active_log_edge": weakest}}


def test_common_support_ablation_requires_both_costs_and_weakest_fold():
    common = {}
    for family in ("price_only", "price_cftc"):
        candidate = f"{family}_r110"
        common[candidate] = {}
        for scenario, _ in experiment.COST_SCENARIOS:
            common[candidate][scenario] = _ablation_metrics(
                0.10 if family == "price_only" else 0.12,
                0.01 if family == "price_only" else 0.011,
            )
    passed = experiment.common_support_ablation(common, risk_multiple=1.10)
    assert passed["passed"] is True
    assert set(passed["scenarios"]) == {"base_5bps", "stress_10bps"}

    failed = copy.deepcopy(common)
    failed["price_cftc_r110"]["stress_10bps"] = _ablation_metrics(0.09, 0.011)
    result = experiment.common_support_ablation(failed, risk_multiple=1.10)
    assert result["passed"] is False
    assert (
        result["scenarios"]["stress_10bps"]["checks"][
            "strict_total_active_log_improvement"
        ]
        is False
    )


def _candidate_result(
    candidate_id: str,
    family: str,
    *,
    minimum_fold: float,
    total: float,
    brier: float,
    cash_days: int,
    passed: bool = True,
) -> dict:
    metrics = {
        "minimum_fold_active_log_edge": minimum_fold,
        "total_active_log_edge": total,
        "brier_improvement": brier,
        "cash_days": cash_days,
    }
    return {
        "candidate": {"candidate_id": candidate_id, "model_family": family},
        "scenarios": {
            "base_5bps": {"metrics": dict(metrics)},
            "stress_10bps": {"metrics": dict(metrics)},
        },
        "passed": passed,
    }


def test_selection_is_passing_only_and_uses_frozen_rank():
    results = [
        _candidate_result(
            "price_only_r110", "price_only", minimum_fold=0.01, total=0.10,
            brier=0.02, cash_days=50,
        ),
        _candidate_result(
            "price_cftc_r110", "price_cftc", minimum_fold=0.011, total=0.09,
            brier=0.03, cash_days=40,
        ),
        _candidate_result(
            "price_only_r120", "price_only", minimum_fold=0.50, total=1.0,
            brier=0.50, cash_days=1, passed=False,
        ),
    ]
    assert experiment.select_development_candidate(results) == "price_cftc_r110"
    assert experiment.select_development_candidate(
        [{**item, "passed": False} for item in results]
    ) is None


def test_timeout_is_fail_closed_before_feature_or_artifact_work(tmp_path):
    readings = iter((0.0, experiment.RUN_TIME_LIMIT_SECONDS + 1.0))
    with pytest.raises(experiment.DownsideEnsembleTimeout, match="bounded input"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(),
            cot_records=(),
            output_dir=tmp_path,
            clock=lambda: next(readings),
            run_id="must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_refit_is_repeated_byte_identically():
    index = pd.bdate_range("2016-01-04", "2018-12-31")
    position = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(index=index)
    for offset, name in enumerate(PRICE_FEATURE_COLUMNS + CFTC_FEATURE_COLUMNS):
        frame[name] = np.sin(position / (5.0 + offset)) + position * 0.0001
    returns = np.where((np.arange(len(index)) % 13) == 0, -0.06, 0.01)
    frame["aapl_forward_log_return_5"] = returns
    frame["crash_label_5session"] = (returns <= np.log(0.96)).astype(np.int8)
    frame["label_maturity_date"] = index
    frame["cftc_price_only_fallback"] = False
    deadline = experiment._Deadline(lambda: 0.0)
    first, metadata = experiment._refit_one_model(
        frame, model_family="price_only", deadline=deadline
    )
    second, _ = experiment._refit_one_model(
        frame.copy(), model_family="price_only", deadline=deadline
    )
    assert first.canonical_json() == second.canonical_json()
    assert metadata["refit_repeated_byte_identically"] is True
    assert metadata["training_latest_label_maturity"] == "2018-12-31"


def test_rejected_run_seals_null_selection_both_costs_and_exact_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    price = _price_frame()
    prediction_index = price.loc["2005-01-01":"2018-12-31"].index
    predictions = _fake_predictions(prediction_index)
    placeholder_features = pd.DataFrame(index=price.index)

    monkeypatch.setattr(
        experiment,
        "build_downside_feature_label_frame",
        lambda frame, cot_records: placeholder_features,
    )
    monkeypatch.setattr(
        experiment,
        "build_pre2019_walkforward_predictions",
        lambda frame: predictions,
    )
    monkeypatch.setattr(
        experiment,
        "_simulate",
        lambda frame, target, cost_bps: _fake_ledger(frame, target),
    )
    def failing_score(*args, **kwargs):
        result = _passing_score(*args, **kwargs)
        result["gates"]["passed"] = False
        result["gates"]["checks"]["synthetic"] = False
        return result

    monkeypatch.setattr(experiment, "_score_strategy", failing_score)
    monkeypatch.setattr(
        experiment,
        "_refit_selected_models",
        lambda frame, selected, deadline: (
            {},
            {
                "performed": False,
                "reason": "no_development_candidate_passed",
                "test_stub": True,
            },
        ),
    )

    report = experiment.run_development_from_inputs(
        price_frame=price,
        cot_records=(),
        output_dir=tmp_path,
        clock=lambda: 0.0,
        run_id="synthetic-sealed-run",
    )
    run_dir = Path(report["artifact_dir"])
    assert report["selected_candidate_id"] is None
    assert report["development_pass"] is False
    verification = experiment.verify_development_artifact(run_dir)
    assert verification["verified"] is True
    assert verification["selected_candidate_id"] is None
    assert verification["development_pass"] is False

    results = json.loads((run_dir / "candidate_results.json").read_text())
    assert len(results) == 8
    for candidate in results:
        assert set(candidate["scenarios"]) == {"base_5bps", "stress_10bps"}
    manifest = json.loads((run_dir / "selection_manifest.json").read_text())
    assert (
        manifest["rejected_absolute_trigger_preflight"]["observed_result"]
        == "zero_cash_targets_for_all_eight_candidates"
    )
    assert manifest["post_2018_market_data_accessed"] is False

    path = run_dir / "candidate_results.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(experiment.DownsideEnsembleError, match="checksum mismatch"):
        experiment.verify_development_artifact(run_dir)
