from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.rare_loss_experiment as experiment
from agent_benchmark.rare_loss_features import (
    CORE_FEATURE_COLUMNS,
    RARE_LOSS_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
)


def _price_frame(*, include_2019: bool = False) -> pd.DataFrame:
    end = "2019-01-04" if include_2019 else "2018-12-31"
    index = pd.bdate_range("2004-01-02", end)
    position = np.arange(len(index), dtype=float)
    state = ((position // 90) % 2).astype(int)
    aapl_return = np.where(state == 0, 0.0008, -0.0004) + 0.0002 * np.sin(
        position / 7.0
    )
    qqq_return = np.where(state == 0, 0.0005, -0.0002) + 0.0001 * np.cos(
        position / 11.0
    )
    aapl = 50.0 * np.exp(np.cumsum(aapl_return))
    return pd.DataFrame(
        {
            "aapl_open": aapl * np.exp(0.0001 * np.sin(position / 5.0)),
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": 100.0 * np.exp(np.cumsum(0.7 * qqq_return)),
            "qqq_adj_close": 80.0 * np.exp(np.cumsum(qqq_return)),
        },
        index=index,
    )


def _context_frame(*, include_2019: bool = False) -> pd.DataFrame:
    end = "2019-01-04" if include_2019 else "2018-12-31"
    index = pd.bdate_range("2004-01-02", end)
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
            # The shared bounded-input adapter must physically erase this value.
            "tnx_close": 25.0 + position * 0.001,
        },
        index=index,
    )


def _candidate_result(
    candidate_id: str,
    *,
    weakest: float,
    total: float,
    severe_brier_improvement: float,
    completed_episodes: int,
    cash_days: int,
    passed: bool = True,
) -> dict:
    metrics = {
        "minimum_fold_active_log_edge": weakest,
        "total_active_log_edge": total,
        "severe_brier_relative_improvement": severe_brier_improvement,
        "completed_cash_episodes": completed_episodes,
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


def test_contract_metadata_matches_the_frozen_rare_loss_document() -> None:
    assert experiment.CONTRACT_VERSION == (
        "aapl-one-session-rare-loss-forest-development-v1"
    )
    assert experiment.RUN_TIME_LIMIT_SECONDS == 3600.0
    assert experiment.COST_SCENARIOS == (
        ("base_5bps", 5.0),
        ("stress_10bps", 10.0),
    )
    assert len(CORE_FEATURE_COLUMNS) == 15
    assert len(SENTIMENT_FEATURE_COLUMNS) == 5
    assert RARE_LOSS_FEATURE_COLUMNS == (
        *CORE_FEATURE_COLUMNS,
        *SENTIMENT_FEATURE_COLUMNS,
    )
    assert [item.to_dict() for item in experiment.CANDIDATE_SPECS] == [
        {
            "candidate_id": "tail975",
            "training_oob_severe_tail_quantile": 0.975,
            "severe_probability_minimum_multiple_of_causal_prevalence": 2.0,
            "ordinary_cash_win_probability_gate": 0.55,
            "expected_clipped_edge_10bps_gate": 0.001,
            "cash_decision_rows": 1,
        },
        {
            "candidate_id": "tail990",
            "training_oob_severe_tail_quantile": 0.99,
            "severe_probability_minimum_multiple_of_causal_prevalence": 2.0,
            "ordinary_cash_win_probability_gate": 0.55,
            "expected_clipped_edge_10bps_gate": 0.001,
            "cash_decision_rows": 1,
        },
    ]
    assert [
        (fold.first_entry_year, fold.last_entry_year)
        for fold in experiment.WALK_FORWARD_FOLDS
    ] == [(year, year + 1) for year in range(2005, 2019, 2)]
    assert experiment.MODEL_VARIANTS == ("full", "core")
    assert set(experiment.SOURCE_EXCLUSION_AUDITS) == {
        "news",
        "cftc",
        "macro_and_tnx",
        "llm",
    }
    assert all(
        audit["included"] is False
        for audit in experiment.SOURCE_EXCLUSION_AUDITS.values()
    )
    assert "docs/aapl_one_session_rare_loss_forest_v1.md" in experiment.SOURCE_FILES


def test_any_post_2018_row_is_refused_even_when_all_values_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(experiment.RareLossExperimentError, match="post-2018 price"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(include_2019=True),
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )
    with pytest.raises(experiment.RareLossExperimentError, match="post-2018 context"):
        experiment.run_development_from_inputs(
            price_frame=_price_frame(),
            context_frame=_context_frame(include_2019=True),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )
    incomplete = _price_frame()
    incomplete.loc[pd.Timestamp("2019-01-02")] = np.nan
    with pytest.raises(experiment.RareLossExperimentError, match="post-2018 price"):
        experiment.run_development_from_inputs(
            price_frame=incomplete,
            context_frame=_context_frame(),
            output_dir=tmp_path,
            clock=lambda: 0.0,
        )
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("tail975", "tail990", "expected"),
    (
        # Minimum-fold edge wins before every later key.
        ((0.02, 0.01, 0.01, 99, 99), (0.01, 1.0, 1.0, 1, 1), "tail975"),
        # Total edge is the second key.
        ((0.01, 0.10, 0.01, 1, 1), (0.01, 0.11, 0.0, 99, 99), "tail990"),
        # Severe-event Brier improvement is the third key.
        ((0.01, 0.10, 0.02, 1, 1), (0.01, 0.10, 0.03, 99, 99), "tail990"),
        # Fewer completed episodes is the fourth key.
        ((0.01, 0.10, 0.02, 10, 1), (0.01, 0.10, 0.02, 9, 99), "tail990"),
        # Fewer CASH days is the fifth key.
        ((0.01, 0.10, 0.02, 10, 11), (0.01, 0.10, 0.02, 10, 10), "tail990"),
        # Lexical candidate ID is the final key.
        ((0.01, 0.10, 0.02, 10, 10), (0.01, 0.10, 0.02, 10, 10), "tail975"),
    ),
)
def test_selection_uses_all_six_frozen_tie_break_keys(
    tail975: tuple[float, float, float, int, int],
    tail990: tuple[float, float, float, int, int],
    expected: str,
) -> None:
    results = [
        _candidate_result(
            candidate_id,
            weakest=values[0],
            total=values[1],
            severe_brier_improvement=values[2],
            completed_episodes=values[3],
            cash_days=values[4],
        )
        for candidate_id, values in (("tail975", tail975), ("tail990", tail990))
    ]
    assert experiment.select_development_candidate(results) == expected
    assert experiment.select_development_candidate(
        [{**item, "passed": False} for item in results]
    ) is None


def test_timeout_happens_before_feature_or_artifact_work(tmp_path: Path) -> None:
    readings = iter((0.0, experiment.RUN_TIME_LIMIT_SECONDS + 1.0))
    with pytest.raises(experiment.RareLossExperimentTimeout, match="bounded input"):
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
        for variant in experiment.MODEL_VARIANTS:
            targets[spec.cash_target_column(variant)] = 0

    results = []
    for spec in experiment.CANDIDATE_SPECS:
        metrics = {
            "minimum_fold_active_log_edge": -0.01,
            "total_active_log_edge": -0.02,
            "severe_brier_relative_improvement": -0.01,
            "completed_cash_episodes": 0,
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
                "core_ablation_scenarios": scenarios,
                "both_cost_gates_passed": False,
                "paired_core_ablation": {"passed": False},
                "passed": False,
            }
        )

    ledger_index = prices.loc["2005-01-03":"2018-12-31"].index
    dates = [date.date().isoformat() for date in ledger_index]
    ledger = pd.DataFrame(
        {
            "decision_date": dates,
            "fill_date": dates,
            "target_exposure": 1.0,
        }
    )
    ledgers = {
        f"buy_hold_{name}.csv": ledger.copy()
        for name, _ in experiment.COST_SCENARIOS
    }
    for spec in experiment.CANDIDATE_SPECS:
        for variant in experiment.MODEL_VARIANTS:
            for scenario, _ in experiment.COST_SCENARIOS:
                ledgers[
                    f"{variant}_{spec.candidate_id}_{scenario}_strategy.csv"
                ] = ledger.copy()
    states = [
        {
            "fold_id": experiment.WALK_FORWARD_FOLDS[position // 2].fold_id,
            "variant": experiment.MODEL_VARIANTS[position % 2],
            "model_state": {},
        }
        for position in range(14)
    ]
    return pd.DataFrame(index=prices.index), predictions, states, targets, results, ledgers


def test_rejected_run_end_to_end_seals_replays_and_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Forest correctness is covered by its focused suite. This fixture keeps
    # the artifact lifecycle real while replacing only the expensive 14 fits.
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
        run_id="synthetic-rare-loss-rejection",
    )
    run_dir = Path(report["artifact_dir"])
    assert report["selected_candidate_id"] is None
    assert report["development_pass"] is False
    assert report["reproducibility"]["model_calls"] == 0
    assert report["reproducibility"]["local_numeric_forest_fits_before_seal"] == 14

    verified = experiment.verify_development_artifact(run_dir)
    assert verified["verified"] is True
    assert verified["selected_candidate_id"] is None
    manifest = json.loads((run_dir / "selection_manifest.json").read_text())
    assert manifest["development_forest_fit_count"] == 14
    assert manifest["refit"]["performed"] is False
    assert manifest["post_2018_market_data_accessed"] is False
    assert not list(run_dir.glob("final_*_through_2018.json"))
    assert not list(tmp_path.glob(".*.sealing"))

    states = run_dir / "oof_forest_states.json"
    states.write_bytes(states.read_bytes() + b"\n")
    with pytest.raises(experiment.RareLossExperimentError, match="checksum mismatch"):
        experiment.verify_development_artifact(run_dir)
