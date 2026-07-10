from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.cftc_cot import (
    CFTC_LEGACY_DATASET_PAGE,
    COTDownload,
    COTRecord,
    COTRetrievalMetadata,
    anomaly_calendar_sha256,
    build_socrata_count_url,
    build_socrata_csv_url,
)
from agent_benchmark.cftc_cot_experiment import (
    CONFIRMATION_END,
    COT_STRUCTURAL_COVERAGE_CONTRACT,
    CFTCExperimentError,
    ACTIVE_EDGE_WIN_TOLERANCE,
    DEVELOPMENT_END,
    DEVELOPMENT_GATES,
    DEVELOPMENT_PRICE_START,
    PRICE_SESSION_CALENDAR_ID,
    _assert_stage_frame_bounds,
    _committed_artifact_snapshot,
    _confirmation_completion_path,
    _finish_confirmation_access,
    _initialize_byte_exact_artifact_directory,
    _month_end_rolling_win_rate,
    _parse_bounded_download,
    _verify_complete_artifact_directory,
    _write_checksums,
    _ledger_csv_bytes,
    _save_ledger_set,
    _simulate_continuous,
    audit_cot_response_coverage,
    audit_price_response_coverage,
    apply_stage_gates,
    build_policy_target,
    expected_pre_2024_nyse_sessions,
    reserve_confirmation_access,
    score_continuous_ledgers,
    select_development_variant,
    to_policy_records,
)
from agent_benchmark.cftc_cot_policy import COTWeeklyRecord


def _price_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.linspace(100.0, 120.0, len(index))
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": values,
            "qqq_adj_close": values,
        },
        index=index,
    )


def _cot_record(code: str, report_date: date, net: int = 100) -> COTRecord:
    return COTRecord(
        source_id=f"{report_date:%Y%m%d}-{code}",
        market_name="TEST",
        contract_code=code,
        report_date=report_date,
        open_interest=1000,
        noncommercial_long=300 + net,
        noncommercial_short=300,
    )


def _cot_csv(records: list[COTRecord]) -> bytes:
    lines = [
        '"id","market_and_exchange_names","report_date_as_yyyy_mm_dd",'
        '"cftc_contract_market_code","open_interest_all",'
        '"noncomm_positions_long_all","noncomm_positions_short_all"'
    ]
    lines.extend(
        (
            f'"{item.source_id}","TEST","{item.report_date.isoformat()}T00:00:00.000",'
            f'"{item.contract_code}","{item.open_interest}",'
            f'"{item.noncommercial_long}","{item.noncommercial_short}"'
        )
        for item in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bounded_cot_download(
    records: list[COTRecord],
    *,
    start: str,
    end: str,
    official_row_count: int,
) -> COTDownload:
    raw = _cot_csv(records)
    count_proof = f'"row_count"\n"{official_row_count}"\n'.encode("utf-8")
    return COTDownload(
        raw_csv=raw,
        count_proof_csv=count_proof,
        metadata=COTRetrievalMetadata(
            dataset_id="6dca-aqww",
            dataset_page=CFTC_LEGACY_DATASET_PAGE,
            source_url=build_socrata_csv_url(start, end),
            requested_start_date=start,
            requested_end_date=end,
            contract_codes=("13874A", "209742", "1170E1"),
            retrieved_at_utc="2026-01-01T00:00:00Z",
            raw_response_sha256=f"sha256:{hashlib.sha256(raw).hexdigest()}",
            raw_size_bytes=len(raw),
            http_status=200,
            content_type="text/csv",
            count_source_url=build_socrata_count_url(start, end),
            count_response_sha256=(
                f"sha256:{hashlib.sha256(count_proof).hexdigest()}"
            ),
            count_size_bytes=len(count_proof),
            count_http_status=200,
            count_content_type="text/csv",
            anomaly_calendar_sha256=anomaly_calendar_sha256(),
        ),
    )


def test_stage_boundaries_are_fixed_before_2024():
    assert DEVELOPMENT_PRICE_START == "1999-12-31"
    assert DEVELOPMENT_END == "2018-12-31"
    assert CONFIRMATION_END == "2023-12-31"
    assert DEVELOPMENT_GATES["minimum_annual_win_rate"] == 0.55


def test_committed_artifact_snapshot_uses_captured_git_blobs(tmp_path):
    repo = tmp_path / "repo"
    run_dir = repo / "e" / "sealed-run"
    _initialize_byte_exact_artifact_directory(run_dir)
    artifact = run_dir / "report.json"
    sealed_bytes = b'{"sealed": true}\r\n'
    artifact.write_bytes(sealed_bytes)

    commands = (
        ("init",),
        ("config", "user.email", "tests@example.invalid"),
        ("config", "user.name", "CFTC Test"),
        ("config", "core.autocrlf", "true"),
        ("add", "."),
        ("commit", "-m", "seal fixture"),
    )
    for command in commands:
        subprocess.run(
            ["git", *command],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # A mutable worktree replacement and an untracked extra file must neither
    # replace nor expand the captured committed artifact set.
    artifact.write_bytes(b"temporary-replacement")
    (run_dir / "untracked.bin").write_bytes(b"not part of the seal")

    snapshot = _committed_artifact_snapshot(repo, run_dir, head_commit=head)

    assert snapshot == {
        ".gitattributes": b"* -text\n",
        "report.json": sealed_bytes,
    }


def test_complete_artifact_directory_rejects_post_checksum_mutation(tmp_path):
    run_dir = tmp_path / "run"
    _initialize_byte_exact_artifact_directory(run_dir)
    payload = run_dir / "payload.csv"
    payload.write_bytes(b"date,value\n2000-01-03,1\n")
    _write_checksums(run_dir)
    expected = {".gitattributes", "payload.csv", "checksums.json"}

    snapshot = _verify_complete_artifact_directory(
        run_dir,
        expected_files=expected,
    )
    assert set(snapshot) == expected

    payload.write_bytes(b"date,value\n2000-01-03,2\n")
    with pytest.raises(CFTCExperimentError, match="checksum mismatch"):
        _verify_complete_artifact_directory(run_dir, expected_files=expected)


def test_price_payload_is_rejected_if_it_physically_contains_a_later_row():
    frame = _price_frame(pd.DatetimeIndex(["2004-01-02", "2019-01-02"]))
    with pytest.raises(CFTCExperimentError, match="outside its physical"):
        _assert_stage_frame_bounds(
            frame,
            start="2004-01-01",
            end=DEVELOPMENT_END,
            stage="development",
        )

    truncated = _price_frame(pd.bdate_range("2008-01-02", "2018-12-31"))
    with pytest.raises(CFTCExperimentError, match="starts materially after"):
        _assert_stage_frame_bounds(
            truncated,
            start="2004-01-01",
            end=DEVELOPMENT_END,
            stage="development",
        )

    sparse_dates = pd.DatetimeIndex(
        ["1999-12-31", "2000-01-03", "2018-12-28", "2018-12-31"]
    )
    with pytest.raises(CFTCExperimentError, match="incomplete internal coverage"):
        _assert_stage_frame_bounds(
            _price_frame(sparse_dates),
            start=DEVELOPMENT_PRICE_START,
            end=DEVELOPMENT_END,
            stage="development",
        )


def test_exact_price_session_calendar_includes_known_pre_2024_closures():
    warmup = expected_pre_2024_nyse_sessions(
        start=DEVELOPMENT_PRICE_START, end="2000-01-03"
    )
    assert [value.date().isoformat() for value in warmup] == [
        "1999-12-31",
        "2000-01-03",
    ]

    september_2001 = expected_pre_2024_nyse_sessions(
        start="2001-09-10", end="2001-09-17"
    )
    assert [value.date().isoformat() for value in september_2001] == [
        "2001-09-10",
        "2001-09-17",
    ]

    sandy = expected_pre_2024_nyse_sessions(start="2012-10-26", end="2012-10-31")
    assert [value.date().isoformat() for value in sandy] == [
        "2012-10-26",
        "2012-10-31",
    ]

    juneteenth = expected_pre_2024_nyse_sessions(start="2022-06-17", end="2022-06-21")
    assert [value.date().isoformat() for value in juneteenth] == [
        "2022-06-17",
        "2022-06-21",
    ]

    assert len(expected_pre_2024_nyse_sessions(start="2000-01-01", end="2000-12-31")) == 252
    assert len(expected_pre_2024_nyse_sessions(start="2001-01-01", end="2001-12-31")) == 248
    assert len(expected_pre_2024_nyse_sessions(start="2012-01-01", end="2012-12-31")) == 250
    assert len(expected_pre_2024_nyse_sessions(start="2023-01-01", end="2023-12-31")) == 250


def test_exact_price_coverage_accepts_the_complete_expected_sequence():
    expected = expected_pre_2024_nyse_sessions(start="2019-01-01", end="2023-12-31")
    audit = audit_price_response_coverage(
        _price_frame(expected),
        start="2019-01-01",
        end="2023-12-31",
        stage="confirmation",
    )
    assert audit["passed"] is True
    assert audit["session_calendar_id"] == PRICE_SESSION_CALENDAR_ID
    assert audit["sessions"] == audit["expected_sessions"]
    assert audit["missing_sessions"] == 0
    assert audit["unexpected_sessions"] == 0
    assert audit["observed_date_sequence_sha256"] == audit["expected_date_sequence_sha256"]


def test_price_coverage_rejects_scattered_matched_omissions():
    expected = expected_pre_2024_nyse_sessions(start="2019-01-01", end="2023-12-31")
    # Simulate the same isolated rows disappearing from AAPL, SPY, and QQQ.
    # This keeps the old weekday ratio well above 94% and never creates a
    # greater-than-seven-day gap, but must fail exact session coverage.
    omitted = expected[25:-25:50]
    observed = expected.difference(omitted)
    with pytest.raises(
        CFTCExperimentError,
        match=rf"incomplete internal coverage \({len(omitted)} missing, 0 unexpected",
    ):
        audit_price_response_coverage(
            _price_frame(observed),
            start="2019-01-01",
            end="2023-12-31",
            stage="confirmation",
        )


def test_price_coverage_rejects_equal_count_session_substitution():
    expected = expected_pre_2024_nyse_sessions(start="2023-01-01", end="2023-01-10")
    observed = expected.drop(pd.Timestamp("2023-01-03")).append(
        pd.DatetimeIndex(["2023-01-02"])
    ).sort_values()
    assert len(observed) == len(expected)
    with pytest.raises(
        CFTCExperimentError,
        match=r"incomplete internal coverage \(1 missing, 1 unexpected",
    ):
        audit_price_response_coverage(
            _price_frame(observed),
            start="2023-01-01",
            end="2023-01-10",
            stage="confirmation",
        )


def test_exact_price_calendar_fails_closed_outside_sealed_pre_2024_range():
    with pytest.raises(CFTCExperimentError, match="only sealed 1999-12-31 through 2023"):
        expected_pre_2024_nyse_sessions(start="2023-12-29", end="2024-01-02")


def test_cftc_response_metadata_and_rows_must_match_the_sealed_bounds():
    raw = (
        '"id","market_and_exchange_names","report_date_as_yyyy_mm_dd",'
        '"cftc_contract_market_code","open_interest_all",'
        '"noncomm_positions_long_all","noncomm_positions_short_all"\n'
        '"one","TEST","2019-01-08T00:00:00.000","13874A","1000","300","200"\n'
    ).encode()
    count_proof = b'"row_count"\n"1"\n'
    metadata = COTRetrievalMetadata(
        dataset_id="6dca-aqww",
        dataset_page=CFTC_LEGACY_DATASET_PAGE,
        source_url=build_socrata_csv_url("1997-09-16", DEVELOPMENT_END),
        requested_start_date="1997-09-16",
        requested_end_date=DEVELOPMENT_END,
        contract_codes=("13874A", "209742", "1170E1"),
        retrieved_at_utc="2026-01-01T00:00:00Z",
        raw_response_sha256=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        raw_size_bytes=len(raw),
        http_status=200,
        content_type="text/csv",
        count_source_url=build_socrata_count_url("1997-09-16", DEVELOPMENT_END),
        count_response_sha256=f"sha256:{hashlib.sha256(count_proof).hexdigest()}",
        count_size_bytes=len(count_proof),
        count_http_status=200,
        count_content_type="text/csv",
        anomaly_calendar_sha256=anomaly_calendar_sha256(),
    )
    with pytest.raises(ValueError, match="outside the expected response bounds"):
        _parse_bounded_download(
            COTDownload(raw_csv=raw, count_proof_csv=count_proof, metadata=metadata),
            expected_start="1997-09-16",
            expected_end=DEVELOPMENT_END,
        )

    wrong_hash = COTDownload(
        raw_csv=raw,
        count_proof_csv=count_proof,
        metadata=COTRetrievalMetadata(
            **{
                **metadata.as_dict(),
                "contract_codes": tuple(metadata.contract_codes),
                "raw_response_sha256": "sha256:false",
            }
        ),
    )
    with pytest.raises(CFTCExperimentError, match="hash does not match"):
        _parse_bounded_download(
            wrong_hash,
            expected_start="1997-09-16",
            expected_end=DEVELOPMENT_END,
        )


def test_cftc_coverage_fails_closed_on_sparse_or_missing_contract_history():
    rows = [_cot_record("13874A", date(2018, 12, 18))]
    with pytest.raises(CFTCExperimentError, match="failed required coverage"):
        audit_cot_response_coverage(
            rows,
            expected_start="1997-09-16",
            expected_end=DEVELOPMENT_END,
        )


def test_cftc_coverage_accepts_official_early_vix_structural_gaps():
    all_dates = [
        value.date()
        for value in pd.date_range("2004-07-27", "2018-12-31", freq="7D")
    ]
    # The official pre-2019 VIX series has 711 rows, about 94.4% weekly
    # coverage, and a longest 168-day gap. Reproduce that structural shape
    # while keeping the other two contracts weekly.
    omitted_indexes = set(range(100, 123)) | {200 + 20 * index for index in range(19)}
    vix_dates = [value for index, value in enumerate(all_dates) if index not in omitted_indexes]
    assert len(all_dates) == 753
    assert len(vix_dates) == 711
    records = [
        _cot_record(code, report_date)
        for code, dates in (
            ("13874A", all_dates),
            ("209742", all_dates),
            ("1170E1", vix_dates),
        )
        for report_date in dates
    ]

    audit = audit_cot_response_coverage(
        records,
        expected_start="2004-07-27",
        expected_end="2018-12-31",
    )

    assert audit["passed"] is True
    vix = audit["contracts"]["1170E1"]
    assert vix["coverage_ratio"] == pytest.approx(711 / 753)
    assert vix["maximum_internal_gap_days"] == 168
    assert vix["structural_coverage_contract"] == dict(
        COT_STRUCTURAL_COVERAGE_CONTRACT["1170E1"]
    )

    offset_records: list[COTRecord] = []
    first = date(2020, 1, 7)
    for index in range(12):
        for code, offset in (("13874A", 0), ("209742", 1), ("1170E1", 2)):
            offset_records.append(
                _cot_record(code, first + timedelta(days=7 * index + offset))
            )
    with pytest.raises(CFTCExperimentError, match="three_market_report_date_alignment"):
        audit_cot_response_coverage(
            offset_records,
            expected_start="2020-01-01",
            expected_end="2020-03-31",
        )


@pytest.mark.parametrize("omission_pattern", ["common_week", "scattered_weeks"])
def test_official_count_proof_catches_omissions_that_heuristic_coverage_misses(
    omission_pattern,
):
    reports = [date(2014, 1, 7) + timedelta(days=7 * index) for index in range(260)]
    all_records = [
        _cot_record(code, report)
        for report in reports
        for code in ("13874A", "209742", "1170E1")
    ]
    if omission_pattern == "common_week":
        missing = {(code, reports[100]) for code in ("13874A", "209742", "1170E1")}
    else:
        missing = {
            (code, reports[100 + index])
            for index, code in enumerate(("13874A", "209742", "1170E1"))
        }
    incomplete = [
        item
        for item in all_records
        if (item.contract_code, item.report_date) not in missing
    ]

    # Even the stricter structural checks accept one row missing from a
    # five-year response; the exact independent count proof must not.
    assert audit_cot_response_coverage(
        incomplete,
        expected_start="2014-01-01",
        expected_end="2018-12-31",
    )["passed"] is True

    download = _bounded_cot_download(
        incomplete,
        start="2014-01-01",
        end="2018-12-31",
        official_row_count=len(all_records),
    )
    with pytest.raises(CFTCExperimentError, match="does not match.*official"):
        _parse_bounded_download(
            download,
            expected_start="2014-01-01",
            expected_end="2018-12-31",
        )


def test_adapter_maps_only_the_three_fixed_markets_and_rejects_post_2023():
    report = date(2023, 12, 19)
    records = to_policy_records(
        [
            _cot_record("13874A", report),
            _cot_record("209742", report),
            _cot_record("1170E1", report),
        ]
    )
    assert {item.market for item in records} == {"SP500", "NASDAQ", "VIX"}
    assert all(item.availability_date == date(2023, 12, 27) for item in records)

    with pytest.raises(CFTCExperimentError, match="post-2023"):
        to_policy_records([_cot_record("13874A", date(2024, 1, 2))])


def test_policy_target_is_binary_and_refuses_records_beyond_phase_cap():
    first = date(2010, 1, 5)
    records: list[COTWeeklyRecord] = []
    for index in range(27):
        report = first + timedelta(days=7 * index)
        available = report + timedelta(days=8)
        records.extend(
            [
                COTWeeklyRecord("NASDAQ", report, available, -0.05 + index * 0.001),
                COTWeeklyRecord("SP500", report, available, 0.01 + index * 0.0005),
                COTWeeklyRecord("VIX", report, available, -0.10 + index * 0.001),
            ]
        )
    prices = _price_frame(pd.bdate_range("2010-07-01", "2010-08-31"))
    target, diagnostics = build_policy_target(
        prices,
        records,
        variant_id="cot_26w_z075",
        maximum_decision_date=DEVELOPMENT_END,
    )
    assert set(target.unique()).issubset({0.0, 1.0})
    assert len(diagnostics) == len(prices)

    future = COTWeeklyRecord(
        "NASDAQ", date(2019, 1, 8), date(2019, 1, 16), 0.0
    )
    with pytest.raises(CFTCExperimentError, match="exceeds the phase"):
        build_policy_target(
            prices,
            [*records, future],
            variant_id="cot_26w_z075",
            maximum_decision_date=DEVELOPMENT_END,
        )


def _synthetic_ledgers() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2005-01-03", "2018-12-31")
    count = len(dates)
    benchmark_returns = np.full(count, 0.0001)
    strategy_returns = np.full(count, 0.0002)
    cash = np.zeros(count, dtype=bool)
    # Twenty separate three-session cash episodes: enough activity, under 2%.
    for start in range(100, 2100, 100):
        cash[start : start + 3] = True
    target = np.where(cash, 0.0, 1.0)
    entered = cash & ~pd.Series(cash).shift(1, fill_value=False).to_numpy()
    exited = ~cash & pd.Series(cash).shift(1, fill_value=False).to_numpy()
    traded = entered | exited
    base = {
        "fill_date": dates.date.astype(str),
        "target_exposure": target,
        "new_exposure_after_fill": target,
        "holding_exposure_for_return": target,
        "cash": np.where(cash, 1000.0, 0.0),
        "shares": np.where(cash, 0.0, 10.0),
        "margin_interest": np.zeros(count),
        "trade_executed": traded,
    }
    strategy = pd.DataFrame({**base, "daily_return": strategy_returns})
    benchmark = pd.DataFrame(
        {
            **base,
            "target_exposure": np.ones(count),
            "new_exposure_after_fill": np.ones(count),
            "holding_exposure_for_return": np.ones(count),
            "cash": np.zeros(count),
            "shares": np.full(count, 10.0),
            "daily_return": benchmark_returns,
        }
    )
    return strategy, benchmark


def test_development_metrics_use_active_logs_and_pass_all_robustness_gates():
    strategy, benchmark = _synthetic_ledgers()
    metrics = score_continuous_ledgers(strategy, benchmark, stage="development")
    gate = apply_stage_gates(metrics, stage="development")

    assert metrics["total_active_log_edge"] > 0.0
    assert metrics["rolling_252_session_month_end_win_rate"] == 1.0
    assert metrics["rolling_756_session_month_end_win_rate"] == 1.0
    assert metrics["cash_days"] == 60
    assert metrics["cash_episodes"] == 20
    assert metrics["positive_folds"] == 4
    assert metrics["active_log_edge_without_best_year"] > 0.0
    assert "global_financial_crisis_2008" in metrics["fixed_downside_windows"]
    assert set(metrics["annual_strategy_returns"]) == set(
        metrics["annual_aapl_buy_hold_returns"]
    )
    assert gate["passed"] is True


def test_scoring_rejects_nonfinite_returns_and_does_not_count_initial_cash():
    strategy, benchmark = _synthetic_ledgers()
    corrupted = benchmark.copy()
    corrupted.loc[10, "daily_return"] = float("nan")
    with pytest.raises(CFTCExperimentError, match="non-finite"):
        score_continuous_ledgers(strategy, corrupted, stage="development")

    always_long = strategy.copy()
    always_long["target_exposure"] = 1.0
    always_long["new_exposure_after_fill"] = 1.0
    always_long["holding_exposure_for_return"] = 1.0
    always_long["cash"] = 0.0
    always_long["shares"] = 10.0
    metrics = score_continuous_ledgers(always_long, benchmark, stage="development")
    assert metrics["cash_days"] == 0
    assert metrics["cash_episodes"] == 0


def test_confirmation_ledger_reuses_a_prior_decision_row_and_counts_real_cash_interval():
    index = pd.DatetimeIndex(
        ["2018-12-31", "2019-01-02", "2019-01-03", "2019-01-04", "2019-01-07"]
    )
    frame = _price_frame(index)
    target = pd.Series([1.0, 0.0, 1.0, 1.0, 1.0], index=index)
    strategy, _, metrics = _simulate_continuous(
        frame,
        target,
        stage="confirmation",
        slippage_bps=5.0,
    )
    assert len(strategy) > 0
    assert metrics["cash_days"] == 1
    assert metrics["cash_episodes"] == 1

    without_warmup = frame.loc["2019-01-02":]
    with pytest.raises(ValueError, match="causal warm-up"):
        _simulate_continuous(
            without_warmup,
            target.loc[without_warmup.index],
            stage="confirmation",
            slippage_bps=5.0,
        )


def test_development_ledger_uses_1999_warmup_and_starts_account_in_2000():
    index = pd.DatetimeIndex(
        ["1999-12-31", "2000-01-03", "2000-01-04", "2000-01-05"]
    )
    frame = _price_frame(index)
    target = pd.Series(1.0, index=index)

    strategy, _, _ = _simulate_continuous(
        frame,
        target,
        stage="development",
        slippage_bps=5.0,
    )

    assert strategy["fill_date"].iloc[0] == "2000-01-03"


def test_development_gate_rejects_economically_immaterial_epsilon():
    strategy, benchmark = _synthetic_ledgers()
    metrics = score_continuous_ledgers(strategy, benchmark, stage="development")
    metrics["total_active_log_edge"] = 1e-12
    gate = apply_stage_gates(metrics, stage="development")
    assert gate["checks"]["material_total_active_log_edge"] is False
    assert gate["passed"] is False


def test_win_rates_do_not_count_floating_point_dust():
    index = pd.bdate_range("2010-01-01", periods=800)
    dust = pd.Series(0.0, index=index)
    dust.iloc[100] = ACTIVE_EDGE_WIN_TOLERANCE / 10.0

    win_rate, observations = _month_end_rolling_win_rate(dust, 252)

    assert observations > 0
    assert win_rate == 0.0


def test_selection_uses_only_passing_variants_and_the_frozen_tie_break():
    def result(variant_id: str, weakest: float, total: float, cash_days: int, passed=True):
        return {
            "variant": {"variant_id": variant_id},
            "passed": passed,
            "scenarios": {
                "stress_10bps": {
                    "metrics": {
                        "minimum_fold_active_log_edge": weakest,
                        "total_active_log_edge": total,
                        "cash_days": cash_days,
                    }
                }
            },
        }

    selected = select_development_variant(
        [
            result("cot_26w_z075", 0.01, 0.20, 50),
            result("cot_52w_z075", 0.02, 0.10, 60),
            result("cot_26w_z125", 1.0, 1.0, 1, passed=False),
        ]
    )
    assert selected == "cot_52w_z075"
    assert select_development_variant([result("cot_26w_z075", 0, 0, 0, False)]) is None


def test_confirmation_access_is_atomic_and_one_shot(tmp_path):
    registry = tmp_path / "confirmation.json"
    first = reserve_confirmation_access(
        registry,
        development_run_id="dev-1",
        development_manifest_sha256="sha256:manifest",
        selected_variant_id="cot_26w_z075",
    )
    assert first["status"] == "reserved_before_data_access"
    assert registry.exists()
    with pytest.raises(CFTCExperimentError, match="already exists"):
        reserve_confirmation_access(
            registry,
            development_run_id="dev-1",
            development_manifest_sha256="sha256:manifest",
            selected_variant_id="cot_26w_z075",
        )


def test_confirmation_completion_preserves_immutable_reservation(tmp_path):
    registry = tmp_path / "confirmation.json"
    reservation = reserve_confirmation_access(
        registry,
        development_run_id="dev-a",
        development_manifest_sha256="sha256:abc",
        selected_variant_id="cot_26w_z075",
    )
    reserved_bytes = registry.read_bytes()

    finished = _finish_confirmation_access(
        registry,
        reservation,
        run_id="confirmation-a",
        confirmation_pass=True,
        confirmation_manifest_sha256="sha256:manifest",
        artifact_checksums_sha256="sha256:checksums",
    )

    assert registry.read_bytes() == reserved_bytes
    assert finished["status"] == "completed"
    assert finished["confirmation_pass"] is True
    completion = _confirmation_completion_path(registry)
    assert completion.exists()
    assert json.loads(completion.read_text(encoding="utf-8"))["access_id"] == reservation[
        "access_id"
    ]


def test_confirmation_completion_refuses_replaced_reservation(tmp_path):
    registry = tmp_path / "confirmation.json"
    reservation = reserve_confirmation_access(
        registry,
        development_run_id="dev-a",
        development_manifest_sha256="sha256:abc",
        selected_variant_id="cot_26w_z075",
    )
    replacement = b'{"access_id":"different-concurrent-access"}\n'
    registry.write_bytes(replacement)

    with pytest.raises(CFTCExperimentError, match="changed before completion"):
        _finish_confirmation_access(
            registry,
            reservation,
            run_id="confirmation-a",
            confirmation_pass=True,
            confirmation_manifest_sha256="sha256:manifest",
            artifact_checksums_sha256="sha256:checksums",
        )

    assert registry.read_bytes() == replacement
    assert not _confirmation_completion_path(registry).exists()


def test_confirmation_cannot_be_reserved_again_if_completion_survives(tmp_path):
    registry = tmp_path / "confirmation.json"
    reservation = reserve_confirmation_access(
        registry,
        development_run_id="dev-a",
        development_manifest_sha256="sha256:abc",
        selected_variant_id="cot_26w_z075",
    )
    _finish_confirmation_access(
        registry,
        reservation,
        run_id="confirmation-a",
        confirmation_pass=False,
        confirmation_manifest_sha256="sha256:manifest",
        artifact_checksums_sha256="sha256:checksums",
    )
    registry.unlink()

    with pytest.raises(CFTCExperimentError, match="previously accessed"):
        reserve_confirmation_access(
            registry,
            development_run_id="dev-b",
            development_manifest_sha256="sha256:def",
            selected_variant_id="cot_52w_z125",
        )

    assert not registry.exists()
    assert _confirmation_completion_path(registry).exists()


def test_saved_ledger_bytes_match_the_readback_verifier_contract(tmp_path):
    frame = pd.DataFrame({"fill_date": ["2018-01-02"], "value": [1.23456789]})
    _save_ledger_set(
        tmp_path,
        variant_id="cot_26w_z075",
        ledgers={"test": frame},
    )
    assert (
        tmp_path / "cot_26w_z075_test.csv"
    ).read_bytes() == _ledger_csv_bytes(frame)
