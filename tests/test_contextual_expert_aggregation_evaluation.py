from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.contextual_expert_aggregation_evaluation as evaluation
from agent_benchmark import contextual_expert_aggregation_ledger as ledger
from agent_benchmark.contextual_expert_aggregation_evaluation import (
    ABLATION_COMPARISON_NAMES,
    BASE_COST_NAME,
    CONFIRMATION_ACCOUNT_YEARS,
    CONFIRMATION_INTEGRITY_CHECKS,
    CONFIRMATION_YEARS,
    COST_BPS,
    COST_NAMES,
    DEVELOPMENT_FOLDS,
    DEVELOPMENT_INTEGRITY_CHECKS,
    DEVELOPMENT_YEARS,
    FIXED_COMPARATOR_NAMES,
    MIN_ACTIVE_LOG_EDGE,
    STRESS_COST_NAME,
    STRICT_INCREMENTAL_EDGE,
    ContextualExpertAggregationEvaluationError,
    apply_confirmation_gates,
    apply_development_gates,
    require_report_status,
    summarize_complete_episodes,
    summarize_policy_ledgers,
    summarize_xor_differences,
)


def _market(
    years: Sequence[int], *, episodes_per_year: int = 3
) -> tuple[pd.Series, pd.Series]:
    dates: list[pd.Timestamp] = []
    prices: list[float] = []
    close_targets: list[int] = []
    negative_years = {2008, 2022}
    for year in years:
        year_dates = pd.date_range(
            f"{year}-01-03", periods=2 * episodes_per_year + 2, freq="14D"
        )
        year_prices = [100.0]
        year_targets: list[int] = []
        for _ in range(episodes_per_year):
            year_targets.extend((0, 1))
            year_prices.extend((100.0, 98.0))
        year_targets.extend((1, 1))
        year_prices.append(90.0 if year in negative_years else 100.0)
        dates.extend(year_dates.tolist())
        prices.extend(year_prices)
        close_targets.extend(year_targets)
    index = pd.DatetimeIndex(dates)
    return (
        pd.Series(prices, index=index, dtype=float),
        pd.Series(close_targets, index=index, dtype=int),
    )


def _run(
    opens: pd.Series,
    targets: pd.Series,
    *,
    policy_name: str,
    cost_name: str,
) -> pd.DataFrame:
    return ledger.run_continuous_ledger(
        opens,
        targets,
        policy_name=policy_name,
        cost_bps=COST_BPS[cost_name],
    ).ledger


def _episodes(frame: pd.DataFrame, *, cost_name: str) -> pd.DataFrame:
    start = ledger.AccountState.initial(
        policy_name=str(frame.iloc[0]["policy_name"]),
        cost_bps=COST_BPS[cost_name],
    )
    extraction = ledger.extract_cash_episodes(frame, start_state=start)
    assert extraction.unresolved.empty
    return extraction.complete


def _xor(
    primary: pd.DataFrame, comparator: pd.DataFrame, *, cost_name: str
) -> pd.DataFrame:
    primary_start = ledger.AccountState.initial(
        policy_name=str(primary.iloc[0]["policy_name"]),
        cost_bps=COST_BPS[cost_name],
    )
    comparator_start = ledger.AccountState.initial(
        policy_name=str(comparator.iloc[0]["policy_name"]),
        cost_bps=COST_BPS[cost_name],
    )
    extraction = ledger.extract_signed_xor_episodes(
        primary,
        comparator,
        primary_start_state=primary_start,
        comparator_start_state=comparator_start,
    )
    assert extraction.unresolved.empty
    return extraction.complete


def _rehash_artifact_row(
    frame: pd.DataFrame,
    *,
    row_index: int,
    columns: Sequence[str],
    hash_column: str,
) -> None:
    row = frame.iloc[row_index].to_dict()
    payload = {name: row[name] for name in columns if name != hash_column}
    frame.at[frame.index[row_index], hash_column] = evaluation._canonical_sha256(
        payload, field="adversarial test row"
    )


def _confused_value(original: object, kind: str) -> object:
    if kind == "mmdd_date":
        return pd.Timestamp(original).strftime("%m/%d/%Y")
    if kind == "datetime_string":
        return f"{original}T00:00:00"
    if kind == "integer":
        return int(float(original))
    if kind == "numeric_string":
        return str(original)
    if kind == "numpy_float":
        return np.float64(original)
    if kind == "float_count":
        return float(original)
    if kind == "numpy_integer":
        return np.int64(original)
    if kind == "boolean":
        return bool(original)
    raise AssertionError(f"unknown adversarial value kind: {kind}")


def _policy(
    opens: pd.Series,
    targets: pd.Series,
    *,
    policy_name: str,
    cost_name: str,
) -> dict[str, pd.DataFrame]:
    strategy = _run(
        opens, targets, policy_name=policy_name, cost_name=cost_name
    )
    benchmark = _run(
        opens,
        pd.Series(1, index=targets.index, dtype=int),
        policy_name="buy_hold",
        cost_name=cost_name,
    )
    return {
        "strategy_ledger": strategy,
        "benchmark_ledger": benchmark,
        "complete_episodes": _episodes(strategy, cost_name=cost_name),
    }


def _pairwise(
    opens: pd.Series,
    primary_targets: pd.Series,
    comparator_targets: pd.Series,
    *,
    comparator_name: str,
    cost_name: str,
) -> dict[str, pd.DataFrame]:
    primary = _run(
        opens, primary_targets, policy_name="learner", cost_name=cost_name
    )
    comparator = _run(
        opens,
        comparator_targets,
        policy_name=comparator_name,
        cost_name=cost_name,
    )
    benchmark = _run(
        opens,
        pd.Series(1, index=primary_targets.index, dtype=int),
        policy_name="buy_hold",
        cost_name=cost_name,
    )
    return {
        "strategy_ledger": comparator,
        "benchmark_ledger": benchmark,
        "complete_episodes": _episodes(comparator, cost_name=cost_name),
        "learner_minus_comparator_xor": _xor(
            primary, comparator, cost_name=cost_name
        ),
    }


def _evidence(years: Sequence[int]):
    opens, learner_targets = _market(years)
    long_targets = pd.Series(1, index=learner_targets.index, dtype=int)
    ablation_targets = learner_targets.copy()
    ablation_targets.loc[ablation_targets.index.year >= 2019] = 1
    learner = {
        cost: _policy(
            opens,
            learner_targets,
            policy_name="learner",
            cost_name=cost,
        )
        for cost in COST_NAMES
    }
    fixed = {
        cost: {
            name: _pairwise(
                opens,
                learner_targets,
                long_targets,
                comparator_name=name,
                cost_name=cost,
            )
            for name in FIXED_COMPARATOR_NAMES
        }
        for cost in COST_NAMES
    }
    ablations = {
        cost: {
            name: _pairwise(
                opens,
                learner_targets,
                ablation_targets,
                comparator_name=name,
                cost_name=cost,
            )
            for name in ABLATION_COMPARISON_NAMES
        }
        for cost in COST_NAMES
    }
    return opens, learner_targets, learner, fixed, ablations


def _development_integrity() -> dict[str, bool]:
    return {name: True for name in DEVELOPMENT_INTEGRITY_CHECKS}


def _confirmation_integrity() -> dict[str, bool]:
    return {name: True for name in CONFIRMATION_INTEGRITY_CHECKS}


def test_episode_and_xor_summaries_derive_entry_open_year() -> None:
    opens, targets = _market(CONFIRMATION_ACCOUNT_YEARS)
    primary = _run(
        opens, targets, policy_name="learner", cost_name=STRESS_COST_NAME
    )
    comparator = _run(
        opens,
        pd.Series(1, index=targets.index, dtype=int),
        policy_name="comparator",
        cost_name=STRESS_COST_NAME,
    )
    episodes = summarize_complete_episodes(
        _episodes(primary, cost_name=STRESS_COST_NAME),
        years=CONFIRMATION_YEARS,
        cost_bps=10.0,
    )
    differences = summarize_xor_differences(
        _xor(primary, comparator, cost_name=STRESS_COST_NAME),
        years=CONFIRMATION_YEARS,
        cost_bps=10.0,
    )
    assert episodes["count"] == differences["count"] == 15
    assert set(episodes["year_edges"]) == {
        str(year) for year in CONFIRMATION_YEARS
    }
    assert differences["positive_year_count"] == 5
    assert differences["orientation_values"] == [
        "primary_cash_comparator_long"
    ]


def test_maximal_multi_fill_cash_and_xor_runs_are_supported() -> None:
    dates = pd.date_range("2005-01-03", periods=6, freq="D")
    opens = pd.Series([100, 100, 99, 98, 100, 100], index=dates, dtype=float)
    primary_targets = pd.Series([0, 0, 1, 1, 1, 1], index=dates, dtype=int)
    comparator_targets = pd.Series(1, index=dates, dtype=int)
    primary = _run(
        opens,
        primary_targets,
        policy_name="learner",
        cost_name=STRESS_COST_NAME,
    )
    comparator = _run(
        opens,
        comparator_targets,
        policy_name="comparator",
        cost_name=STRESS_COST_NAME,
    )
    episodes = _episodes(primary, cost_name=STRESS_COST_NAME)
    differences = _xor(primary, comparator, cost_name=STRESS_COST_NAME)
    assert len(episodes) == len(differences) == 1
    assert int(episodes.iloc[0]["cash_fill_observations"]) == 2
    assert int(differences.iloc[0]["xor_fill_observations"]) == 2


def test_direct_xor_flip_splits_into_two_exact_directional_runs() -> None:
    dates = pd.date_range("2005-01-03", periods=6, freq="D")
    opens = pd.Series([100, 99, 98, 97, 96, 95], index=dates, dtype=float)
    primary_targets = pd.Series([0, 1, 1, 1, 1, 1], index=dates, dtype=int)
    comparator_targets = pd.Series([1, 0, 1, 1, 1, 1], index=dates, dtype=int)
    primary = _run(
        opens,
        primary_targets,
        policy_name="learner",
        cost_name=STRESS_COST_NAME,
    )
    comparator = _run(
        opens,
        comparator_targets,
        policy_name="comparator",
        cost_name=STRESS_COST_NAME,
    )

    differences = _xor(primary, comparator, cost_name=STRESS_COST_NAME)

    assert differences["orientation"].tolist() == [
        ledger.PRIMARY_CASH_COMPARATOR_LONG,
        ledger.PRIMARY_LONG_COMPARATOR_CASH,
    ]
    assert "mixed" not in differences["orientation"].tolist()
    assert differences.iloc[0]["exit_fill_date"] == differences.iloc[1][
        "entry_fill_date"
    ]
    summary = summarize_xor_differences(
        differences, years=(2005,), cost_bps=COST_BPS[STRESS_COST_NAME]
    )
    assert summary["count"] == 2
    assert summary["orientation_values"] == sorted(
        [
            ledger.PRIMARY_CASH_COMPARATOR_LONG,
            ledger.PRIMARY_LONG_COMPARATOR_CASH,
        ]
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("cost_log_edge", 0.0),
        ("net_active_log_edge", 999.0),
        ("entry_fill_date", "1900-01-01"),
        ("exit_fill_date", "2005-02-01"),
        ("entry_reference_price", 999.0),
        ("entry_sell_fill_price", 999.0),
        ("cash_fill_observations", 0),
        ("episode_id", "not-a-canonical-digest"),
        ("episode_sha256", "sha256:" + "0" * 64),
    ],
)
def test_episode_schema_date_and_cost_tamper_fail(
    column: str, value: object
) -> None:
    opens, targets = _market(DEVELOPMENT_YEARS)
    frame = _run(
        opens, targets, policy_name="learner", cost_name=BASE_COST_NAME
    )
    episodes = _episodes(frame, cost_name=BASE_COST_NAME)
    episodes.loc[0, column] = value
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        summarize_complete_episodes(
            episodes, years=DEVELOPMENT_YEARS, cost_bps=5.0
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("orientation", "mixed"),
        ("entry_fill_date", "1900-01-01"),
        ("net_signed_log_edge", 1.0),
        ("transition_cost_component", 1.0),
        ("xor_fill_observations", 0),
        ("xor_id", "not-a-canonical-digest"),
        ("xor_sha256", "sha256:" + "0" * 64),
    ],
)
def test_xor_orientation_entry_date_and_edge_tamper_fail(
    column: str, value: object
) -> None:
    opens, targets = _market(CONFIRMATION_ACCOUNT_YEARS)
    primary = _run(
        opens, targets, policy_name="learner", cost_name=STRESS_COST_NAME
    )
    comparator = _run(
        opens,
        pd.Series(1, index=targets.index, dtype=int),
        policy_name="comparator",
        cost_name=STRESS_COST_NAME,
    )
    differences = _xor(primary, comparator, cost_name=STRESS_COST_NAME)
    differences.loc[0, column] = value
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        summarize_xor_differences(
            differences, years=CONFIRMATION_YEARS, cost_bps=10.0
        )


@pytest.mark.parametrize(
    "column,kind",
    [
        ("entry_fill_date", "mmdd_date"),
        ("exit_fill_date", "datetime_string"),
        ("entry_reference_price", "integer"),
        ("entry_reference_price", "numeric_string"),
        ("entry_reference_price", "numpy_float"),
        ("entry_reference_price", "boolean"),
        ("cash_fill_observations", "float_count"),
        ("cash_fill_observations", "numeric_string"),
        ("cash_fill_observations", "numpy_integer"),
        ("cash_fill_observations", "boolean"),
    ],
)
def test_self_rehashed_episode_type_confusion_fails(
    column: str, kind: str
) -> None:
    opens, targets = _market(DEVELOPMENT_YEARS)
    strategy = _run(
        opens, targets, policy_name="learner", cost_name=BASE_COST_NAME
    )
    episodes = _episodes(strategy, cost_name=BASE_COST_NAME)
    original = episodes.at[episodes.index[0], column]
    episodes[column] = episodes[column].astype(object)
    episodes.at[episodes.index[0], column] = _confused_value(original, kind)
    _rehash_artifact_row(
        episodes,
        row_index=0,
        columns=ledger.EPISODE_COLUMNS,
        hash_column="episode_sha256",
    )

    with pytest.raises(ContextualExpertAggregationEvaluationError):
        summarize_complete_episodes(
            episodes, years=DEVELOPMENT_YEARS, cost_bps=5.0
        )


@pytest.mark.parametrize(
    "column,kind",
    [
        ("entry_fill_date", "mmdd_date"),
        ("exit_fill_date", "datetime_string"),
        ("raw_market_component", "integer"),
        ("raw_market_component", "numeric_string"),
        ("raw_market_component", "numpy_float"),
        ("raw_market_component", "boolean"),
        ("xor_fill_observations", "float_count"),
        ("xor_fill_observations", "numeric_string"),
        ("xor_fill_observations", "numpy_integer"),
        ("xor_fill_observations", "boolean"),
    ],
)
def test_self_rehashed_xor_type_confusion_fails(column: str, kind: str) -> None:
    opens, targets = _market(CONFIRMATION_ACCOUNT_YEARS)
    primary = _run(
        opens, targets, policy_name="learner", cost_name=STRESS_COST_NAME
    )
    comparator = _run(
        opens,
        pd.Series(1, index=targets.index, dtype=int),
        policy_name="comparator",
        cost_name=STRESS_COST_NAME,
    )
    differences = _xor(primary, comparator, cost_name=STRESS_COST_NAME)
    original = differences.at[differences.index[0], column]
    differences[column] = differences[column].astype(object)
    differences.at[differences.index[0], column] = _confused_value(original, kind)
    _rehash_artifact_row(
        differences,
        row_index=0,
        columns=ledger.XOR_COLUMNS,
        hash_column="xor_sha256",
    )

    with pytest.raises(ContextualExpertAggregationEvaluationError):
        summarize_xor_differences(
            differences, years=CONFIRMATION_YEARS, cost_bps=10.0
        )


def test_reporting_year_and_cost_type_confusion_fails() -> None:
    opens, targets = _market(DEVELOPMENT_YEARS)
    strategy = _run(
        opens, targets, policy_name="learner", cost_name=BASE_COST_NAME
    )
    episodes = _episodes(strategy, cost_name=BASE_COST_NAME)
    bad_years = (
        ("2005",),
        (np.int64(2005),),
        (True,),
        (2006, 2005),
        (2005, 2005),
    )
    for years in bad_years:
        with pytest.raises(ContextualExpertAggregationEvaluationError):
            summarize_complete_episodes(episodes, years=years, cost_bps=5.0)
    for cost_bps in (5, "5.0", np.float64(5.0), True):
        with pytest.raises(ContextualExpertAggregationEvaluationError):
            summarize_complete_episodes(
                episodes, years=DEVELOPMENT_YEARS, cost_bps=cost_bps
            )


@pytest.mark.parametrize(
    "threshold,left_small_count,middle_small_count",
    [
        (MIN_ACTIVE_LOG_EDGE, 5, 8),
        (STRICT_INCREMENTAL_EDGE, 4, 13),
    ],
)
def test_fsum_strict_boundaries_resist_cancellation(
    threshold: float,
    left_small_count: int,
    middle_small_count: int,
) -> None:
    small = threshold / 28.0
    values = (
        [small] * left_small_count
        + [-1.0]
        + [small] * middle_small_count
        + [1.0]
        + [small] * (28 - left_small_count - middle_small_count)
    )
    assert len(values) == 30
    assert math.fsum(values) == threshold
    assert float(np.sum(np.asarray(values, dtype=float))) > threshold
    exact = evaluation._stats(values, prefix="strict-boundary")
    assert exact["sum"] == threshold
    assert exact["mean"] == threshold / len(values)
    assert not exact["sum"] > threshold

    just_above = values.copy()
    just_above[-1] += math.nextafter(threshold, math.inf) - threshold
    above = evaluation._stats(just_above, prefix="strict-boundary-above")
    assert above["sum"] == math.nextafter(threshold, math.inf)
    assert above["sum"] > threshold


def test_policy_ledger_replay_and_episode_reconciliation() -> None:
    opens, targets = _market(CONFIRMATION_ACCOUNT_YEARS)
    policy = _policy(
        opens,
        targets,
        policy_name="learner",
        cost_name=STRESS_COST_NAME,
    )
    summary = summarize_policy_ledgers(
        **policy,
        reporting_years=CONFIRMATION_YEARS,
        account_years=CONFIRMATION_ACCOUNT_YEARS,
        cost_bps=10.0,
    )
    assert summary["full_account_episode_reconciled"] is True
    assert abs(summary["full_account_episode_reconciliation_error"]) <= 1e-10
    assert summary["negative_aapl_years"] == [2022]
    assert summary["negative_aapl_year_edge_sum"] > 0.0
    assert summary["full_account_active_log_edge"] > summary["reporting_active_log_edge"]


def test_full_account_strict_threshold_uses_atomic_fsum_not_terminal_equity() -> None:
    dates = pd.DatetimeIndex(
        ["2005-01-03", "2005-01-04", "2005-01-05", "2005-01-06"]
    )
    opens = pd.Series(
        [100.0, 100.0, 99.79022027920416, 99.79022027920416],
        index=dates,
        dtype=float,
    )
    targets = pd.Series([0, 1, 1, 1], index=dates, dtype=int)
    policy = _policy(
        opens,
        targets,
        policy_name="learner",
        cost_name=STRESS_COST_NAME,
    )
    summary = summarize_policy_ledgers(
        **policy,
        reporting_years=(2005,),
        account_years=(2005,),
        cost_bps=10.0,
    )
    terminal_edge = math.log(
        float(policy["strategy_ledger"].iloc[-1]["equity"])
        / float(policy["benchmark_ledger"].iloc[-1]["equity"])
    )

    assert terminal_edge > STRICT_INCREMENTAL_EDGE
    assert summary["entry_attributed_full_account_active_log_edge"] < (
        STRICT_INCREMENTAL_EDGE
    )
    assert summary["full_account_active_log_edge"] == summary[
        "entry_attributed_full_account_active_log_edge"
    ]
    assert not summary["full_account_active_log_edge"] > STRICT_INCREMENTAL_EDGE


@pytest.mark.parametrize(
    "column,value",
    [
        ("daily_return", 0.01),
        ("equity", 1.0),
        ("cash_after_fill", 1.0),
        ("shares_after_fill", 1.0),
        ("post_fill_exposure", 1),
        ("margin_interest", 0.01),
    ],
)
def test_canonical_ledger_arithmetic_and_hash_tamper_fail(
    column: str, value: object
) -> None:
    opens, targets = _market(DEVELOPMENT_YEARS)
    policy = _policy(
        opens, targets, policy_name="learner", cost_name=BASE_COST_NAME
    )
    policy["strategy_ledger"].loc[3, column] = value
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        summarize_policy_ledgers(
            **policy,
            reporting_years=DEVELOPMENT_YEARS,
            account_years=DEVELOPMENT_YEARS,
            cost_bps=5.0,
            folds=DEVELOPMENT_FOLDS,
        )


def test_extra_schema_and_duplicate_episode_fail() -> None:
    opens, targets = _market(DEVELOPMENT_YEARS)
    policy = _policy(
        opens, targets, policy_name="learner", cost_name=BASE_COST_NAME
    )
    policy["strategy_ledger"]["extra"] = 1
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        summarize_policy_ledgers(
            **policy,
            reporting_years=DEVELOPMENT_YEARS,
            account_years=DEVELOPMENT_YEARS,
            cost_bps=5.0,
            folds=DEVELOPMENT_FOLDS,
        )
    policy = _policy(
        opens, targets, policy_name="learner", cost_name=BASE_COST_NAME
    )
    policy["complete_episodes"] = pd.concat(
        [policy["complete_episodes"], policy["complete_episodes"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        summarize_policy_ledgers(
            **policy,
            reporting_years=DEVELOPMENT_YEARS,
            account_years=DEVELOPMENT_YEARS,
            cost_bps=5.0,
            folds=DEVELOPMENT_FOLDS,
        )


def test_development_gates_pass_only_from_raw_evidence() -> None:
    _, _, learner, fixed, _ = _evidence(DEVELOPMENT_YEARS)
    report = apply_development_gates(
        learner,
        fixed_comparator_evidence=fixed,
        integrity=_development_integrity(),
    )
    assert report["passed"] is True
    require_report_status(report, expected_pass=True)
    assert not any(
        name.endswith("negative_aapl_years_exist") for name in report["checks"]
    )


@pytest.mark.parametrize("kind", ["missing", "extra", "false"])
def test_development_integrity_inventory_is_exact(kind: str) -> None:
    _, _, learner, fixed, _ = _evidence(DEVELOPMENT_YEARS)
    integrity = _development_integrity()
    if kind == "missing":
        integrity.pop(next(iter(integrity)))
    elif kind == "extra":
        integrity["proof"] = True
    else:
        integrity[next(iter(integrity))] = False
    if kind == "false":
        assert apply_development_gates(
            learner, fixed_comparator_evidence=fixed, integrity=integrity
        )["passed"] is False
    else:
        with pytest.raises(ContextualExpertAggregationEvaluationError):
            apply_development_gates(
                learner, fixed_comparator_evidence=fixed, integrity=integrity
            )


def test_development_strict_active_and_comparator_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, learner, fixed, _ = _evidence(DEVELOPMENT_YEARS)
    initial = apply_development_gates(
        learner,
        fixed_comparator_evidence=fixed,
        integrity=_development_integrity(),
    )
    observed = initial["policy_summaries"][BASE_COST_NAME][
        "reporting_active_log_edge"
    ]
    monkeypatch.setattr(evaluation, "MIN_ACTIVE_LOG_EDGE", observed)
    report = apply_development_gates(
        learner,
        fixed_comparator_evidence=fixed,
        integrity=_development_integrity(),
    )
    assert report["passed"] is False
    monkeypatch.setattr(evaluation, "MIN_ACTIVE_LOG_EDGE", 0.001)
    initial = apply_development_gates(
        learner,
        fixed_comparator_evidence=fixed,
        integrity=_development_integrity(),
    )
    monkeypatch.setattr(
        evaluation,
        "STRICT_INCREMENTAL_EDGE",
        initial["best_fixed_comparator"]["learner_minus_comparator"],
    )
    assert apply_development_gates(
        learner,
        fixed_comparator_evidence=fixed,
        integrity=_development_integrity(),
    )["passed"] is False


def test_confirmation_gates_and_all_three_ablations_pass() -> None:
    _, _, learner, fixed, ablations = _evidence(CONFIRMATION_ACCOUNT_YEARS)
    report = apply_confirmation_gates(
        learner,
        fixed_comparator_evidence=fixed,
        ablation_evidence=ablations,
        integrity=_confirmation_integrity(),
    )
    assert report["passed"] is True
    require_report_status(report, expected_pass=True)
    assert not any(
        name.endswith("negative_aapl_years_exist") for name in report["checks"]
    )
    for cost in COST_NAMES:
        assert sum(
            value > 0.0
            for value in report["learner_minus_union_year_edges"][cost].values()
        ) == 5


def test_each_confirmation_ablation_requires_exact_through_2018_prefix() -> None:
    opens, learner_targets, learner, fixed, ablations = _evidence(
        CONFIRMATION_ACCOUNT_YEARS
    )
    bad_targets = learner_targets.copy()
    bad_targets.loc[bad_targets.index.year >= 2019] = 1
    pre_cutoff_cash = np.flatnonzero(
        (bad_targets.index.year <= 2018) & (bad_targets.to_numpy(dtype=int) == 0)
    )
    assert pre_cutoff_cash.size > 0
    bad_targets.iloc[int(pre_cutoff_cash[0])] = 1

    for comparison_name in ABLATION_COMPARISON_NAMES:
        tampered = copy.deepcopy(ablations)
        for cost_name in COST_NAMES:
            tampered[cost_name][comparison_name] = _pairwise(
                opens,
                learner_targets,
                bad_targets,
                comparator_name=comparison_name,
                cost_name=cost_name,
            )
        with pytest.raises(
            ContextualExpertAggregationEvaluationError,
            match="through-2018 account/action prefix",
        ):
            apply_confirmation_gates(
                learner,
                fixed_comparator_evidence=fixed,
                ablation_evidence=tampered,
                integrity=_confirmation_integrity(),
            )


def test_confirmation_uses_full_account_comparator_not_reporting_only() -> None:
    _, _, learner, fixed, ablations = _evidence(CONFIRMATION_ACCOUNT_YEARS)
    report = apply_confirmation_gates(
        learner,
        fixed_comparator_evidence=fixed,
        ablation_evidence=ablations,
        integrity=_confirmation_integrity(),
    )
    summary = report["policy_summaries"][STRESS_COST_NAME]
    observed = report["fixed_comparator_differences"][STRESS_COST_NAME][
        "always_long"
    ]
    assert observed == pytest.approx(summary["full_account_active_log_edge"])
    assert observed > summary["reporting_active_log_edge"]


def test_pairwise_entry_years_come_from_xor_runs_not_subtracted_cash_episodes() -> None:
    opens, learner_targets = _market(CONFIRMATION_ACCOUNT_YEARS)
    comparator_targets = pd.Series(1, index=learner_targets.index, dtype=int)
    years = comparator_targets.index.year.to_numpy(dtype=int)
    entry_decision = int(np.flatnonzero(years == 2018)[-2])
    exit_decision = int(np.flatnonzero(years == 2019)[2])
    comparator_targets.iloc[entry_decision : exit_decision + 1] = 0
    primary = evaluation._evaluate_policy_evidence(
        _policy(
            opens,
            learner_targets,
            policy_name="learner",
            cost_name=STRESS_COST_NAME,
        ),
        cost_name=STRESS_COST_NAME,
        stage="confirmation",
        field="cross-year primary",
    )
    pairwise = evaluation._evaluate_pairwise_evidence(
        _pairwise(
            opens,
            learner_targets,
            comparator_targets,
            comparator_name="cross_year_comparator",
            cost_name=STRESS_COST_NAME,
        ),
        primary=primary,
        cost_name=STRESS_COST_NAME,
        stage="confirmation",
        field="cross-year pairwise",
    )

    subtracted_cash_reporting = float(
        primary["summary"]["reporting_active_log_edge"]
        - pairwise["policy"]["summary"]["reporting_active_log_edge"]
    )
    xor_reporting = float(pairwise["xor_summary"]["sum"])
    assert not np.isclose(subtracted_cash_reporting, xor_reporting, atol=1e-12)
    assert pairwise["reporting_incremental_edge"] == xor_reporting


def test_confirmation_integrity_inventory_is_exact() -> None:
    _, _, learner, fixed, ablations = _evidence(CONFIRMATION_ACCOUNT_YEARS)
    integrity = _confirmation_integrity()
    integrity.pop("development_checkpoint_continuity_exact")
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_confirmation_gates(
            learner,
            fixed_comparator_evidence=fixed,
            ablation_evidence=ablations,
            integrity=integrity,
        )


def test_union_delta_is_recomputed_from_canonical_xor_rows() -> None:
    _, _, learner, fixed, ablations = _evidence(CONFIRMATION_ACCOUNT_YEARS)
    tampered = copy.deepcopy(fixed)
    frame = tampered[BASE_COST_NAME]["exact_union_cash"][
        "learner_minus_comparator_xor"
    ]
    frame.loc[0, "net_signed_log_edge"] += 0.01
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_confirmation_gates(
            learner,
            fixed_comparator_evidence=tampered,
            ablation_evidence=ablations,
            integrity=_confirmation_integrity(),
        )


def test_self_rehashed_xor_transition_tamper_fails_canonical_gate_replay() -> None:
    _, _, learner, fixed, ablations = _evidence(CONFIRMATION_ACCOUNT_YEARS)
    tampered = copy.deepcopy(fixed)
    frame = tampered[BASE_COST_NAME]["exact_union_cash"][
        "learner_minus_comparator_xor"
    ]
    frame.loc[0, "transition_cost_component"] += 0.001
    frame.loc[0, "net_signed_log_edge"] += 0.001
    row = frame.iloc[0].to_dict()
    payload = {name: row[name] for name in ledger.XOR_COLUMNS[:-1]}
    frame.loc[0, "xor_sha256"] = evaluation._canonical_sha256(
        payload, field="test XOR row"
    )

    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_confirmation_gates(
            learner,
            fixed_comparator_evidence=tampered,
            ablation_evidence=ablations,
            integrity=_confirmation_integrity(),
        )


def test_ablation_xor_cost_identity_and_exact_stage_years_fail_closed() -> None:
    _, _, learner, fixed, ablations = _evidence(CONFIRMATION_ACCOUNT_YEARS)
    tampered = copy.deepcopy(ablations)
    frame = tampered[STRESS_COST_NAME]["full_minus_global_only"][
        "learner_minus_comparator_xor"
    ]
    frame.loc[0, "entry_fill_date"] = "1900-01-01"
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_confirmation_gates(
            learner,
            fixed_comparator_evidence=fixed,
            ablation_evidence=tampered,
            integrity=_confirmation_integrity(),
        )
    id_tampered = copy.deepcopy(ablations)
    id_tampered[STRESS_COST_NAME]["full_minus_global_only"][
        "learner_minus_comparator_xor"
    ].loc[0, "xor_id"] = "sha256:" + "0" * 64
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_confirmation_gates(
            learner,
            fixed_comparator_evidence=fixed,
            ablation_evidence=id_tampered,
            integrity=_confirmation_integrity(),
        )
    shortened = copy.deepcopy(learner)
    for key in ("strategy_ledger", "benchmark_ledger"):
        shortened[BASE_COST_NAME][key] = shortened[BASE_COST_NAME][key].loc[
            shortened[BASE_COST_NAME][key]["fill_date"].str[:4] != "2023"
        ].reset_index(drop=True)
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_confirmation_gates(
            shortened,
            fixed_comparator_evidence=fixed,
            ablation_evidence=ablations,
            integrity=_confirmation_integrity(),
        )


def test_valid_but_different_action_stream_between_costs_is_rejected() -> None:
    opens, targets, learner, fixed, _ = _evidence(DEVELOPMENT_YEARS)
    alternate = targets.copy()
    alternate.iloc[0] = 1
    learner[STRESS_COST_NAME] = _policy(
        opens,
        alternate,
        policy_name="learner",
        cost_name=STRESS_COST_NAME,
    )
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_development_gates(
            learner,
            fixed_comparator_evidence=fixed,
            integrity=_development_integrity(),
        )


def test_always_long_control_must_equal_same_ledger_benchmark() -> None:
    opens, targets, learner, fixed, ablations = _evidence(
        CONFIRMATION_ACCOUNT_YEARS
    )
    tampered = copy.deepcopy(fixed)
    tampered[BASE_COST_NAME]["always_long"] = _pairwise(
        opens,
        targets,
        targets,
        comparator_name="always_long",
        cost_name=BASE_COST_NAME,
    )
    with pytest.raises(ContextualExpertAggregationEvaluationError):
        apply_confirmation_gates(
            learner,
            fixed_comparator_evidence=tampered,
            ablation_evidence=ablations,
            integrity=_confirmation_integrity(),
        )


def test_report_diagnostic_tamper_is_rejected() -> None:
    _, _, learner, fixed, _ = _evidence(DEVELOPMENT_YEARS)
    report = apply_development_gates(
        learner,
        fixed_comparator_evidence=fixed,
        integrity=_development_integrity(),
    )
    for field, value in (
        ("passed_count", 0),
        ("total_count", 999),
        ("failed_checks", ["fabricated"]),
    ):
        bad = copy.deepcopy(report)
        bad[field] = value
        with pytest.raises(ContextualExpertAggregationEvaluationError):
            require_report_status(bad, expected_pass=True)


def test_no_acquisition_imports_are_present() -> None:
    source = open(evaluation.__file__, encoding="utf-8").read().lower()
    for forbidden in (
        "import yfinance",
        "import requests",
        "import urllib",
        "import socket",
        "from yfinance",
        "from requests",
    ):
        assert forbidden not in source
