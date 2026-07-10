from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.regime_consensus_experiment as experiment
from agent_benchmark.regime_consensus_features import (
    ALL_HEADS_READY_COLUMN,
    HEAD_FEATURE_COLUMNS,
    HEAD_NAMES,
    HEAD_ORIENTATIONS,
    regime_head_readiness,
)
from agent_benchmark.regime_consensus_walkforward import (
    build_pre2019_regime_consensus_walkforward,
)
from agent_benchmark.regime_hmm import RegimeHeadModel


def _price_frame(*, include_2019: bool = False) -> pd.DataFrame:
    end = "2019-01-04" if include_2019 else "2018-12-31"
    index = pd.bdate_range("2000-01-03", end)
    position = np.arange(len(index), dtype=float)
    state = ((position // 90) % 2).astype(int)
    aapl_return = np.where(state == 0, 0.0008, -0.0004) + 0.0002 * np.sin(
        position / 7.0
    )
    qqq_return = np.where(state == 0, 0.0005, -0.0002) + 0.0001 * np.cos(
        position / 11.0
    )
    spy_return = 0.7 * qqq_return
    aapl = 50.0 * np.exp(np.cumsum(aapl_return))
    return pd.DataFrame(
        {
            "aapl_open": aapl * np.exp(0.0001 * np.sin(position / 5.0)),
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": 100.0 * np.exp(np.cumsum(spy_return)),
            "qqq_adj_close": 80.0 * np.exp(np.cumsum(qqq_return)),
        },
        index=index,
    )


def _context_frame(*, include_2019: bool = False) -> pd.DataFrame:
    end = "2019-01-04" if include_2019 else "2018-12-31"
    index = pd.bdate_range("2000-01-03", end)
    position = np.arange(len(index), dtype=float)
    state = ((position // 90) % 2).astype(int)
    iwm_return = np.where(state == 0, 0.0004, -0.0003) + 0.0001 * np.sin(
        position / 9.0
    )
    return pd.DataFrame(
        {
            "iwm_adj_close": 70.0 * np.exp(np.cumsum(iwm_return)),
            "vix_close": np.where(state == 0, 15.0, 29.0)
            + np.sin(position / 13.0),
            # An injected TNX value is physically erased by the frozen boundary.
            "tnx_close": 25.0 + position * 0.001,
        },
        index=index,
    )


def _actual_hmm_feature_frame() -> pd.DataFrame:
    """Long, separated regimes that exercise all 21 real HMM fits."""

    rng = np.random.default_rng(90210)
    dates = pd.bdate_range("2000-01-03", "2018-12-31", name="decision_date")
    latent = np.zeros(len(dates), dtype=np.int8)
    for position in range(1, len(dates)):
        latent[position] = (
            1 - latent[position - 1]
            if rng.random() < 0.02
            else latent[position - 1]
        )
    frame = pd.DataFrame(index=dates)
    for columns, orientations in zip(
        HEAD_FEATURE_COLUMNS, HEAD_ORIENTATIONS, strict=True
    ):
        for offset, (column, orientation) in enumerate(
            zip(columns, orientations, strict=True)
        ):
            frame[column] = (
                orientation * np.where(latent == 1, 1.5, -1.5)
                + rng.normal(scale=0.2, size=len(dates))
                + offset * 0.01
            )
    readiness = regime_head_readiness(frame)
    for column in readiness:
        frame[column] = readiness[column]

    entry = pd.Series(dates, index=dates).shift(-1)
    maturity = pd.Series(dates, index=dates).shift(-2)
    edge_10 = np.where(latent == 1, 0.006, -0.003) + rng.normal(
        scale=0.001, size=len(dates)
    )
    edge_10[-2:] = np.nan
    edge_5 = edge_10 + 0.001
    frame["label_entry_date"] = entry
    frame["label_maturity_date"] = maturity
    frame["label_entry_adjusted_open"] = np.where(entry.notna(), 100.0, np.nan)
    frame["label_exit_adjusted_open"] = np.where(maturity.notna(), 100.0, np.nan)
    frame["aapl_forward_log_return_1"] = -edge_10
    frame["cash_active_log_edge_5bps"] = edge_5
    frame["cash_active_log_edge_10bps"] = edge_10
    for column, values in (
        ("cash_beats_long_5bps", edge_5),
        ("cash_beats_long_10bps", edge_10),
    ):
        binary = pd.Series(
            (np.nan_to_num(values) > 0.0).astype(np.int8),
            index=dates,
            dtype="Int8",
        )
        binary.iloc[-2:] = pd.NA
        frame[column] = binary
    frame["label_available"] = np.isfinite(edge_10)
    frame["entry_fill_year"] = pd.to_datetime(entry).dt.year.astype("Int16")
    assert frame[ALL_HEADS_READY_COLUMN].all()
    return frame


def _candidate_result(
    candidate_id: str,
    *,
    weakest: float,
    total: float,
    brier: float,
    cash_days: int,
    passed: bool = True,
) -> dict:
    metrics = {
        "minimum_fold_active_log_edge": weakest,
        "total_active_log_edge": total,
        "brier_relative_improvement": brier,
        "cash_days": cash_days,
    }
    return {
        "candidate": {"candidate_id": candidate_id},
        "scenarios": {
            "base_5bps": {"metrics": dict(metrics)},
            "stress_10bps": {"metrics": dict(metrics)},
        },
        "passed": passed,
    }


def test_contract_has_two_candidates_and_all_four_explicit_exclusions() -> None:
    assert [item.candidate_id for item in experiment.CANDIDATE_SPECS] == [
        "p55_e5",
        "p60_e5",
    ]
    assert "TNX" not in experiment.CONTEXT_PARQUET_QUERY
    assert set(experiment.SOURCE_EXCLUSION_AUDITS) == {
        "macro",
        "news",
        "cftc",
        "tnx",
    }
    assert all(
        audit["included"] is False
        for audit in experiment.SOURCE_EXCLUSION_AUDITS.values()
    )


def test_actual_hmm_walkforward_completes_all_twenty_one_fits() -> None:
    result = build_pre2019_regime_consensus_walkforward(
        _actual_hmm_feature_frame()
    )
    assert len(result.model_states) == 21
    assert result.predictions["fold_id"].nunique() == 7
    assert result.predictions.index.max() <= pd.Timestamp("2018-12-31")
    assert set(item["head_name"] for item in result.model_states) == set(HEAD_NAMES)
    for record in result.model_states:
        restored = RegimeHeadModel.from_state(record["state"])
        assert restored.model_sha256 == record["model_sha256"]
    assert all(
        spec.cash_target_column(variant) in result.predictions
        for spec in experiment.CANDIDATE_SPECS
        for variant in ("regime_consensus", "aapl_only")
    )


def test_any_post_2018_row_is_rejected_even_when_values_are_missing(tmp_path: Path) -> None:
    with pytest.raises(experiment.RegimeConsensusExperimentError, match="post-2018 price"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(include_2019=True),
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )
    with pytest.raises(experiment.RegimeConsensusExperimentError, match="post-2018 context"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(),
            context_frame=_context_frame(include_2019=True),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )
    incomplete = _price_frame()
    incomplete.loc[pd.Timestamp("2019-01-02")] = np.nan
    with pytest.raises(experiment.RegimeConsensusExperimentError, match="post-2018 price"):
        experiment.run_development_from_inputs(
            price_frame=incomplete,
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )


def test_bounded_context_query_excludes_tnx_and_future_rows(tmp_path: Path) -> None:
    source = tmp_path / "context_daily.parquet"
    pd.DataFrame(
        {
            "date": ["2018-12-31", "2018-12-31", "2018-12-31", "2019-01-02"],
            "symbol": ["IWM", "VIX", "TNX", "IWM"],
            "close": [100.0, 20.0, 30.0, 101.0],
            "adj_close": [99.0, 20.0, 30.0, 100.0],
        }
    ).to_parquet(source, index=False)
    frame, provenance = experiment.load_bounded_context_parquet(source)
    assert frame["symbol"].tolist() == ["IWM", "VIX"]
    assert pd.to_datetime(frame["date"]).max() == pd.Timestamp("2018-12-31")
    assert provenance["allowed_symbols"] == ["IWM", "VIX"]
    assert provenance["network_access"] is False
    assert provenance["source_file_sha256"].startswith("sha256:")


def test_selection_uses_all_five_frozen_tie_break_keys() -> None:
    results = [
        _candidate_result(
            "p55_e5", weakest=0.01, total=0.10, brier=0.03, cash_days=10
        ),
        _candidate_result(
            "p60_e5", weakest=0.011, total=0.09, brier=0.02, cash_days=20
        ),
    ]
    assert experiment.select_development_candidate(results) == "p60_e5"
    tied_weakest = [
        _candidate_result(
            "p55_e5", weakest=0.01, total=0.10, brier=0.03, cash_days=10
        ),
        _candidate_result(
            "p60_e5", weakest=0.01, total=0.10, brier=0.04, cash_days=20
        ),
    ]
    assert experiment.select_development_candidate(tied_weakest) == "p60_e5"
    assert experiment.select_development_candidate(
        [{**item, "passed": False} for item in results]
    ) is None


def test_both_variants_use_identical_ready_support_and_keep_immature_suffix() -> None:
    index = pd.bdate_range("2018-12-24", periods=4, name="decision_date")
    predictions = pd.DataFrame(index=index)
    predictions["all_three_heads_ready"] = [False, True, True, True]
    predictions["causal_prevalence_probability_10bps"] = 0.4
    predictions["causal_training_mean_edge_10bps"] = -0.001
    for variant, probability in (("regime_consensus", 0.6), ("aapl_only", 0.55)):
        predictions[f"{variant}_cash_win_probability_10bps"] = probability
        predictions[f"{variant}_expected_edge_10bps"] = 0.002
    predictions.loc[index[0], "regime_consensus_cash_win_probability_10bps"] = np.nan
    predictions["cash_beats_long_10bps"] = [0.0, 1.0, 0.0, np.nan]
    predictions["cash_active_log_edge_10bps"] = [-0.1, 0.1, -0.1, np.nan]
    support = experiment._common_predictive_support(predictions)
    assert support.tolist() == [False, True, True, True]

    for variant in ("regime_consensus", "aapl_only"):
        inputs = experiment._prediction_inputs(
            predictions,
            variant=variant,
            cash_target=np.asarray([0, 1, 0, 0]),
            support=support,
        )
        assert inputs[-1].equals(index[1:])
        assert len(inputs[0]) == 3
        assert np.isnan(inputs[1][-1])
    with pytest.raises(experiment.RegimeConsensusExperimentError, match="must remain LONG"):
        experiment._prediction_inputs(
            predictions,
            variant="regime_consensus",
            cash_target=np.asarray([1, 1, 0, 0]),
            support=support,
        )


def test_timeout_happens_before_feature_or_artifact_work(tmp_path: Path) -> None:
    readings = iter((0.0, experiment.RUN_TIME_LIMIT_SECONDS + 1.0))
    with pytest.raises(experiment.RegimeConsensusExperimentTimeout, match="bounded input"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(),
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: next(readings),
            run_id="must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def _fake_computation(prices: pd.DataFrame) -> tuple:
    prediction_index = prices.loc["2005-01-03":"2018-12-31"].index
    predictions = pd.DataFrame(index=prediction_index)
    predictions.index.name = "decision_date"
    targets = pd.DataFrame(
        {"common_predictive_support": True}, index=prediction_index
    )
    targets.index.name = "decision_date"
    for spec in experiment.CANDIDATE_SPECS:
        for variant in ("regime_consensus", "aapl_only"):
            targets[spec.cash_target_column(variant)] = 0
    results = []
    for spec in experiment.CANDIDATE_SPECS:
        metrics = {
            "minimum_fold_active_log_edge": -0.01,
            "total_active_log_edge": -0.02,
            "brier_relative_improvement": -0.01,
            "cash_days": 0,
        }
        scenarios = {
            name: {"metrics": dict(metrics), "gates": {"passed": False}}
            for name, _ in experiment.COST_SCENARIOS
        }
        results.append(
            {
                "candidate": spec.to_dict(),
                "stage": "development_only",
                "scenarios": scenarios,
                "aapl_only_scenarios": scenarios,
                "both_cost_gates_passed": False,
                "paired_aapl_only_ablation": {"passed": False},
                "passed": False,
            }
        )
    ledger_index = prices.loc["2005-01-03":"2018-12-31"].index
    ledger = pd.DataFrame(
        {
            "decision_date": ledger_index.date.astype(str),
            "fill_date": ledger_index.date.astype(str),
            "target_exposure": 1.0,
        }
    )
    ledgers = {
        f"buy_hold_{name}.csv": ledger.copy()
        for name, _ in experiment.COST_SCENARIOS
    }
    for spec in experiment.CANDIDATE_SPECS:
        for variant in ("regime_consensus", "aapl_only"):
            for scenario, _ in experiment.COST_SCENARIOS:
                ledgers[
                    f"{variant}_{spec.candidate_id}_{scenario}_strategy.csv"
                ] = ledger.copy()
    states = [
        {"fold_id": f"f{position // 3}", "head_name": f"h{position % 3}", "state": {}}
        for position in range(21)
    ]
    return pd.DataFrame(index=prices.index), predictions, states, targets, results, ledgers


def test_rejected_run_end_to_end_seals_replays_and_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep this test at the run/atomic-seal/verifier boundary.  The focused
    # walk-forward suite exercises all real HMM fits; this fixture makes the
    # artifact lifecycle fast while still enforcing its exact six-value API.
    monkeypatch.setattr(
        experiment,
        "_compute_development",
        lambda prices, context, deadline: _fake_computation(prices),
    )
    report = experiment.run_development_from_inputs(
        price_frame=_price_frame(),
        context_frame=_context_frame(),
        output_dir=tmp_path,
        clock=lambda: 0.0,
        run_id="synthetic-regime-rejection",
    )
    run_dir = Path(report["artifact_dir"])
    assert report["selected_candidate_id"] is None
    assert report["development_pass"] is False
    assert report["reproducibility"]["model_calls"] == 0
    assert report["reproducibility"]["model_calls_semantics"] == (
        "external_or_llm_model_calls_only"
    )
    assert report["reproducibility"]["local_numeric_hmm_fits_before_seal"] == 21
    verified = experiment.verify_development_artifact(run_dir)
    assert verified["verified"] is True
    assert verified["selected_candidate_id"] is None
    manifest = json.loads((run_dir / "selection_manifest.json").read_text())
    assert manifest["hmm_fit_count"] == 21
    assert manifest["refit"]["performed"] is False
    assert manifest["post_2018_market_data_accessed"] is False
    assert not list(run_dir.glob("final_*_model_through_2018.json"))

    states = run_dir / "oof_hmm_states.json"
    states.write_bytes(states.read_bytes() + b"\n")
    with pytest.raises(
        experiment.RegimeConsensusExperimentError, match="checksum mismatch"
    ):
        experiment.verify_development_artifact(run_dir)
