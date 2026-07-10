from __future__ import annotations

from datetime import date, timedelta

import pytest

from agent_benchmark.cftc_cot_policy import (
    CFTC_COT_VARIANTS,
    COTPolicyVariant,
    COTWeeklyRecord,
    evaluate_all_cftc_cot_variants,
    evaluate_cftc_cot_policy,
    get_cftc_cot_variant,
)


def _weekly_records(
    rows: list[tuple[float, float, float]],
    *,
    first_report: date = date(2010, 1, 5),
) -> list[COTWeeklyRecord]:
    records: list[COTWeeklyRecord] = []
    for index, (nasdaq, sp500, vix) in enumerate(rows):
        report_date = first_report + timedelta(days=7 * index)
        availability_date = report_date + timedelta(days=8)
        records.extend(
            [
                COTWeeklyRecord("NASDAQ", report_date, availability_date, nasdaq),
                COTWeeklyRecord("SP500", report_date, availability_date, sp500),
                COTWeeklyRecord("VIX", report_date, availability_date, vix),
            ]
        )
    return records


def _variable_history(count: int) -> list[tuple[float, float, float]]:
    # Each individual signal has non-zero variance, while the NASDAQ-SP500
    # divergence varies independently enough to be standardized.
    return [
        (
            -0.08 + (index % 7) * 0.02,
            -0.03 + (index % 5) * 0.015,
            -0.12 + (index % 6) * 0.025,
        )
        for index in range(count)
    ]


def test_policy_family_contains_exactly_the_four_predeclared_variants():
    assert [(item.lookback_releases, item.z_threshold) for item in CFTC_COT_VARIANTS] == [
        (26, 0.75),
        (26, 1.25),
        (52, 0.75),
        (52, 1.25),
    ]
    assert len({item.variant_id for item in CFTC_COT_VARIANTS}) == 4

    with pytest.raises(ValueError, match="unknown CFTC COT variant"):
        get_cftc_cot_variant("cot_13w_z050")
    with pytest.raises(ValueError, match="frozen declaration"):
        get_cftc_cot_variant(COTPolicyVariant("cot_26w_z075", 20, 0.75))


def test_two_of_three_vote_moves_to_cash_and_never_uses_leverage_or_shorting():
    rows = _variable_history(26)
    # NASDAQ is abnormally low, its divergence from SP500 is abnormally low,
    # and VIX is ordinary: exactly two risk-off votes.
    rows.append((-0.50, 0.00, -0.06))
    records = _weekly_records(rows)
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=records[-1].availability_date,
        variant="cot_26w_z075",
    )

    assert decision.status == "ready"
    assert decision.stance == "CASH"
    assert decision.action == "CASH_ALL"
    assert decision.target_exposure == 0.0
    assert decision.cash_votes == 2
    assert [signal.triggered for signal in decision.signals] == [True, True, False]
    assert all(signal.history_count == 26 for signal in decision.signals)


def test_one_vote_keeps_the_long_baseline():
    rows = _variable_history(26)
    rows.append((-0.50, -0.50, -0.06))  # NASDAQ low, divergence and VIX ordinary.
    records = _weekly_records(rows)
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=records[-1].availability_date,
        variant="cot_26w_z075",
    )

    assert decision.cash_votes == 1
    assert decision.stance == "LONG"
    assert decision.action == "BUY_ALL"
    assert decision.target_exposure == 1.0


def test_vix_high_is_the_third_declared_vote():
    rows = _variable_history(26)
    rows.append((-0.50, -0.50, 0.50))  # NASDAQ low plus VIX high.
    records = _weekly_records(rows)
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=records[-1].availability_date,
        variant="cot_26w_z125",
    )

    assert [signal.direction for signal in decision.signals] == ["low", "low", "high"]
    assert [signal.triggered for signal in decision.signals] == [True, False, True]
    assert decision.cash_votes == 2
    assert decision.target_exposure == 0.0


def test_current_release_is_excluded_from_its_own_zscore_reference_window():
    history = _variable_history(26)
    records = _weekly_records([*history, (-0.50, 0.00, -0.06)])
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=records[-1].availability_date,
        variant="cot_26w_z075",
    )

    nasdaq = decision.signals[0]
    expected_mean = sum(row[0] for row in history) / 26
    assert nasdaq.history_count == 26
    assert nasdaq.history_mean == pytest.approx(expected_mean)
    assert nasdaq.current_value == pytest.approx(-0.50)
    assert nasdaq.history_mean != pytest.approx(
        (sum(row[0] for row in history) - 0.50) / 27
    )


def test_future_releases_cannot_change_an_earlier_decision():
    rows = [*_variable_history(26), (-0.50, 0.00, -0.06)]
    records = _weekly_records(rows)
    as_of = records[-1].availability_date
    original = evaluate_cftc_cot_policy(
        records,
        decision_date=as_of,
        variant="cot_26w_z075",
    )

    future = _weekly_records(
        [(1000.0, -1000.0, 1000.0)],
        first_report=records[-1].report_date + timedelta(days=7),
    )
    with_future = evaluate_cftc_cot_policy(
        [*future, *reversed(records)],
        decision_date=as_of,
        variant="cot_26w_z075",
    )

    assert with_future == original
    assert with_future.to_dict() == original.to_dict()


def test_release_is_usable_at_14_days_but_fails_long_without_signals_at_15_days():
    rows = [*_variable_history(26), (-0.50, 0.00, -0.06)]
    records = _weekly_records(rows)
    latest_availability = records[-1].availability_date

    boundary = evaluate_cftc_cot_policy(
        records,
        decision_date=latest_availability + timedelta(days=14),
        variant="cot_26w_z075",
    )
    stale = evaluate_cftc_cot_policy(
        records,
        decision_date=latest_availability + timedelta(days=15),
        variant="cot_26w_z075",
    )

    assert boundary.status == "ready"
    assert boundary.cash_votes == 2
    assert boundary.stance == "CASH"
    assert len(boundary.signals) == 3

    assert stale.status == "stale_release"
    assert stale.cash_votes == 0
    assert stale.stance == "LONG"
    assert stale.action == "BUY_ALL"
    assert stale.target_exposure == 1.0
    assert stale.signals == ()
    assert stale.current_availability_date == latest_availability


def test_incomplete_release_is_never_used():
    records = _weekly_records(_variable_history(27))
    current_report = records[-1].report_date + timedelta(days=7)
    current_availability = current_report + timedelta(days=8)
    records.extend(
        [
            COTWeeklyRecord("NASDAQ", current_report, current_availability, -100.0),
            COTWeeklyRecord("SP500", current_report, current_availability, 100.0),
            # VIX is deliberately absent, so this is not a usable three-market release.
        ]
    )
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=current_availability,
        variant="cot_26w_z075",
    )

    assert decision.current_report_date == records[-3].report_date
    assert decision.complete_releases_available == 27


@pytest.mark.parametrize("lookback", [26, 52])
def test_full_declared_lookback_is_required_before_any_cash_call(lookback):
    rows = [*_variable_history(lookback - 1), (-100.0, 100.0, 100.0)]
    records = _weekly_records(rows)
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=records[-1].availability_date,
        variant=f"cot_{lookback}w_z075",
    )

    assert decision.status == "insufficient_history"
    assert decision.cash_votes == 0
    assert decision.stance == "LONG"
    assert decision.target_exposure == 1.0
    assert all(signal.status == "insufficient_history" for signal in decision.signals)


def test_zero_variance_reference_history_abstains_instead_of_inventing_a_zscore():
    records = _weekly_records([(0.1, 0.0, -0.1)] * 26 + [(-1.0, 0.0, 1.0)])
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=records[-1].availability_date,
        variant="cot_26w_z075",
    )

    assert decision.status == "ready"
    assert decision.cash_votes == 0
    assert decision.stance == "LONG"
    assert all(signal.status == "zero_variance" for signal in decision.signals)
    assert all(signal.z_score is None for signal in decision.signals)


def test_no_complete_release_defaults_to_long_with_auditable_diagnostics():
    report_date = date(2018, 1, 2)
    decision = evaluate_cftc_cot_policy(
        [COTWeeklyRecord("NASDAQ", report_date, report_date + timedelta(days=8), 0.1)],
        decision_date=report_date + timedelta(days=8),
        variant="cot_26w_z075",
    )

    assert decision.status == "no_complete_release"
    assert decision.current_report_date is None
    assert decision.signals == ()
    assert decision.target_exposure == 1.0
    assert decision.to_dict()["action"] == "BUY_ALL"


def test_same_day_releases_are_not_treated_as_previously_available_history():
    records = _weekly_records(_variable_history(26))
    tied_availability = records[-1].availability_date
    next_report = records[-1].report_date + timedelta(days=7)
    records.extend(
        [
            COTWeeklyRecord("NASDAQ", next_report, tied_availability, -1.0),
            COTWeeklyRecord("SP500", next_report, tied_availability, 1.0),
            COTWeeklyRecord("VIX", next_report, tied_availability, 1.0),
        ]
    )
    decision = evaluate_cftc_cot_policy(
        records,
        decision_date=tied_availability,
        variant="cot_26w_z075",
    )

    assert decision.current_report_date == next_report
    assert decision.status == "insufficient_history"
    assert decision.prior_releases_used == 25


def test_input_order_does_not_change_deterministic_output():
    records = _weekly_records([*_variable_history(52), (-0.50, 0.00, 0.50)])
    forward = evaluate_all_cftc_cot_variants(
        records,
        decision_date=records[-1].availability_date,
    )
    reverse = evaluate_all_cftc_cot_variants(
        reversed(records),
        decision_date=records[-1].availability_date,
    )

    assert forward == reverse
    assert len(forward) == 4
    assert [item.variant_id for item in forward] == [
        item.variant_id for item in CFTC_COT_VARIANTS
    ]
    assert all(item.target_exposure in {0.0, 1.0} for item in forward)


def test_record_validation_rejects_ambiguous_or_non_point_in_time_input():
    report_date = date(2017, 1, 3)
    with pytest.raises(ValueError, match="availability_date cannot be before"):
        COTWeeklyRecord("NASDAQ", report_date, report_date - timedelta(days=1), 0.1)
    with pytest.raises(ValueError, match="market must be one of"):
        COTWeeklyRecord("DOW", report_date, report_date, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        COTWeeklyRecord("NASDAQ", report_date, report_date, float("nan"))
    with pytest.raises(ValueError, match="ISO-8601 date"):
        COTWeeklyRecord("NASDAQ", "2017-01-03-not-a-time", report_date, 0.1)

    duplicate = COTWeeklyRecord("NASDAQ", report_date, report_date, 0.1)
    with pytest.raises(ValueError, match="duplicate normalized COT record"):
        evaluate_cftc_cot_policy(
            [duplicate, duplicate],
            decision_date=report_date,
            variant="cot_26w_z075",
        )
