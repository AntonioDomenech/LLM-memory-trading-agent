from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from agent_benchmark.cftc_cot import (
    CFTC_LEGACY_FUTURES_ONLY_CSV_URL,
    CFTC_LEGACY_FUTURES_ONLY_DATASET_ID,
    COTDataError,
    COTFeature,
    COTRecord,
    EARLIEST_SUPPORTED_REPORT_DATE,
    OFFICIAL_ANOMALY_CALENDAR,
    POST_2023_GUARD_DATE,
    SUPPORTED_CONTRACT_CODES,
    Post2023AccessError,
    anomaly_calendar_sha256,
    apply_official_anomaly_calendar,
    build_socrata_count_params,
    build_socrata_count_url,
    build_socrata_csv_url,
    build_socrata_params,
    feature_history_as_of,
    fetch_legacy_cot_csv,
    latest_features_as_of,
    official_anomaly_for,
    parse_legacy_cot_csv,
    parse_socrata_count_proof_csv,
    trailing_net_share_zscore,
    validate_request_bounds,
)


CSV_HEADER = (
    '"id","market_and_exchange_names","report_date_as_yyyy_mm_dd",'
    '"cftc_contract_market_code","open_interest_all",'
    '"noncomm_positions_long_all","noncomm_positions_short_all"\n'
)


def _csv_row(
    source_id: str,
    report_date: str,
    code: str,
    *,
    open_interest: str = "1000",
    long: str = "300",
    short: str = "100",
    market_name: str = "TEST MARKET",
) -> str:
    return (
        f'"{source_id}","{market_name}","{report_date}T00:00:00.000",'
        f'"{code}","{open_interest}","{long}","{short}"\n'
    )


def _record(
    code: str,
    report_date: date,
    *,
    source_id: str | None = None,
    open_interest: int = 1000,
    long: int = 300,
    short: int = 100,
) -> COTRecord:
    return COTRecord(
        source_id=source_id or f"{report_date:%Y%m%d}-{code}",
        market_name="TEST MARKET",
        contract_code=code,
        report_date=report_date,
        open_interest=open_interest,
        noncommercial_long=long,
        noncommercial_short=short,
    )


def test_fixed_official_dataset_and_contract_codes():
    assert CFTC_LEGACY_FUTURES_ONLY_DATASET_ID == "6dca-aqww"
    assert CFTC_LEGACY_FUTURES_ONLY_CSV_URL == (
        "https://publicreporting.cftc.gov/resource/6dca-aqww.csv"
    )
    assert SUPPORTED_CONTRACT_CODES == ("13874A", "209742", "1170E1")
    assert EARLIEST_SUPPORTED_REPORT_DATE == date(1997, 9, 16)
    assert POST_2023_GUARD_DATE == date(2023, 12, 31)


def test_query_is_bounded_to_fixed_ids_fields_dates_and_order():
    params = build_socrata_params("2000-01-01", "2023-12-31")

    for code in SUPPORTED_CONTRACT_CODES:
        assert f"'{code}'" in params["$where"]
    assert "2000-01-01T00:00:00.000" in params["$where"]
    assert "2024-01-01T00:00:00.000" in params["$where"]
    assert params["$select"].split(",") == [
        "id",
        "market_and_exchange_names",
        "report_date_as_yyyy_mm_dd",
        "cftc_contract_market_code",
        "open_interest_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
    ]
    assert params["$order"].startswith("report_date_as_yyyy_mm_dd ASC")
    assert params["$limit"] == "50000"

    parsed = parse_qs(urlparse(build_socrata_csv_url("2000-01-01", "2000-01-31")).query)
    assert parsed["$where"] == [build_socrata_params("2000-01-01", "2000-01-31")["$where"]]

    count_params = build_socrata_count_params("2000-01-01", "2000-01-31")
    assert count_params == {
        "$select": "count(*) AS row_count",
        "$where": build_socrata_params("2000-01-01", "2000-01-31")["$where"],
        "$limit": "1",
    }
    parsed_count = parse_qs(
        urlparse(build_socrata_count_url("2000-01-01", "2000-01-31")).query
    )
    assert parsed_count == {key: [value] for key, value in count_params.items()}


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("1997-09-15", "2000-01-01", "precedes"),
        ("2000-01-02", "2000-01-01", "on or before"),
    ],
)
def test_invalid_query_bounds_are_rejected(start, end, message):
    with pytest.raises(COTDataError, match=message):
        validate_request_bounds(start, end)


def test_post_2023_query_requires_explicit_opt_in():
    with pytest.raises(Post2023AccessError, match="allow_post_2023=True"):
        build_socrata_params("2023-12-01", "2024-01-02")
    with pytest.raises(Post2023AccessError, match="allow_post_2023=True"):
        build_socrata_count_params("2023-12-01", "2024-01-02")

    params = build_socrata_params(
        "2023-12-01", "2024-01-02", allow_post_2023=True
    )
    assert "2024-01-03T00:00:00.000" in params["$where"]


def test_fetch_preserves_byte_exact_hash_and_retrieval_metadata():
    raw = (CSV_HEADER + _csv_row("row-1", "2023-12-26", "13874A")).encode()
    count_raw = b'"row_count"\n"1"\n'

    class FakeResponse:
        def __init__(self, *, content, url):
            self.content = content
            self.status_code = 200
            self.url = url
            self.headers = {"content-type": "text/csv; charset=utf-8"}

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if kwargs["params"]["$select"] == "count(*) AS row_count":
                return FakeResponse(
                    content=count_raw,
                    url=build_socrata_count_url("2023-12-26", "2023-12-26"),
                )
            return FakeResponse(
                content=raw,
                url=build_socrata_csv_url("2023-12-26", "2023-12-26"),
            )

    session = FakeSession()
    download = fetch_legacy_cot_csv(
        "2023-12-26",
        "2023-12-26",
        session=session,
        retrieved_at=datetime(2024, 1, 2, 12, 30, tzinfo=timezone.utc),
    )

    assert download.raw_csv == raw
    assert download.count_proof_csv == count_raw
    assert download.metadata.raw_response_sha256 == f"sha256:{hashlib.sha256(raw).hexdigest()}"
    assert download.metadata.raw_size_bytes == len(raw)
    assert download.metadata.retrieved_at_utc == "2024-01-02T12:30:00Z"
    assert download.metadata.contract_codes == SUPPORTED_CONTRACT_CODES
    assert download.metadata.requested_start_date == "2023-12-26"
    assert download.metadata.requested_end_date == "2023-12-26"
    assert download.metadata.http_status == 200
    assert download.metadata.content_type == "text/csv; charset=utf-8"
    assert download.metadata.count_source_url == build_socrata_count_url(
        "2023-12-26", "2023-12-26"
    )
    assert download.metadata.count_response_sha256 == (
        f"sha256:{hashlib.sha256(count_raw).hexdigest()}"
    )
    assert download.metadata.count_size_bytes == len(count_raw)
    assert download.metadata.count_http_status == 200
    assert download.metadata.count_content_type == "text/csv; charset=utf-8"
    assert download.metadata.anomaly_calendar_sha256 == anomaly_calendar_sha256()
    assert session.calls[0][0] == CFTC_LEGACY_FUTURES_ONLY_CSV_URL
    assert session.calls[0][1]["params"] == build_socrata_params(
        "2023-12-26", "2023-12-26"
    )
    assert session.calls[0][1]["timeout"] == 30.0
    assert session.calls[1][0] == CFTC_LEGACY_FUTURES_ONLY_CSV_URL
    assert session.calls[1][1]["params"] == build_socrata_count_params(
        "2023-12-26", "2023-12-26"
    )
    assert parse_socrata_count_proof_csv(download.count_proof_csv) == 1


def test_fetch_rejects_naive_retrieval_timestamp():
    class FakeResponse:
        content = CSV_HEADER.encode()
        status_code = 200
        url = CFTC_LEGACY_FUTURES_ONLY_CSV_URL
        headers = {}

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return FakeResponse()

    session = FakeSession()
    with pytest.raises(COTDataError, match="timezone-aware"):
        fetch_legacy_cot_csv(
            "2023-01-01",
            "2023-01-31",
            session=session,
            retrieved_at=datetime(2024, 1, 1, 12, 0),
        )
    assert session.calls == 0


def test_offline_parser_computes_release_and_net_share_features():
    records = parse_legacy_cot_csv(
        CSV_HEADER + _csv_row("row-1", "2023-12-26", "13874A")
    )

    assert len(records) == 1
    record = records[0]
    assert record.report_date == date(2023, 12, 26)
    assert record.available_date == date(2024, 1, 3)
    assert record.noncommercial_net == 200
    assert record.noncommercial_net_share == pytest.approx(0.2)
    assert record.usable is True


def test_parser_rejects_post_2023_rows_unless_explicitly_allowed():
    raw = CSV_HEADER + _csv_row("row-1", "2024-01-02", "13874A")
    with pytest.raises(Post2023AccessError, match="allow_post_2023=True"):
        parse_legacy_cot_csv(raw)

    records = parse_legacy_cot_csv(raw, allow_post_2023=True)
    assert records[0].report_date == date(2024, 1, 2)


def test_parser_enforces_request_specific_response_bounds():
    raw = CSV_HEADER + _csv_row("row-1", "2019-01-08", "13874A")
    with pytest.raises(COTDataError, match="outside the expected response bounds"):
        parse_legacy_cot_csv(
            raw,
            expected_start_date="1997-09-16",
            expected_end_date="2018-12-31",
        )
    with pytest.raises(COTDataError, match="provided together"):
        parse_legacy_cot_csv(raw, expected_start_date="1997-09-16")


@pytest.mark.parametrize(
    "raw,match",
    [
        (CSV_HEADER + _csv_row("row-1", "2023-01-03", "UNKNOWN"), "Unsupported"),
        (
            CSV_HEADER + _csv_row("row-1", "2023-01-03", "13874A", open_interest="0"),
            "greater than zero",
        ),
        (
            CSV_HEADER + _csv_row("row-1", "2023-01-03", "13874A", long="not-a-number"),
            "must be an integer",
        ),
        (
            CSV_HEADER
            + _csv_row("row-1", "2023-01-03", "13874A")
            + _csv_row("row-2", "2023-01-03", "13874A"),
            "duplicate",
        ),
        (
            CSV_HEADER
            + _csv_row("row-1", "2023-01-03-not-a-timestamp", "13874A"),
            "must be an ISO date",
        ),
    ],
)
def test_parser_fails_closed_on_invalid_payloads(raw, match):
    with pytest.raises(COTDataError, match=match):
        parse_legacy_cot_csv(raw)


def test_parser_requires_every_selected_source_field():
    raw = '"id","cftc_contract_market_code"\n"one","13874A"\n'
    with pytest.raises(COTDataError, match="missing required fields"):
        parse_legacy_cot_csv(raw)


def test_official_anomaly_calendar_covers_selected_contract_corrections_and_outages():
    vix_correction = official_anomaly_for("1170e1", "2009-11-24")
    assert vix_correction is not None
    assert vix_correction.anomaly_id == "cftc-vix-corrections-2009"
    assert official_anomaly_for("13874A", "2009-11-24") is None

    assert official_anomaly_for("1170E1", "2012-10-30").kind == "stale_positions"
    assert official_anomaly_for("209742", "2011-06-21") is not None
    for code in SUPPORTED_CONTRACT_CODES:
        assert official_anomaly_for(code, "2012-11-27") is not None
        assert official_anomaly_for(code, "2019-01-08").kind == "delayed_publication"
        assert official_anomaly_for(code, "2019-02-19").kind == "delayed_publication"
        assert official_anomaly_for(code, "2019-02-26") is None
        assert official_anomaly_for(code, "2019-03-26").kind == "corrected_after_publication"
        assert official_anomaly_for(code, "2023-02-14").kind == "delayed_publication"
        assert official_anomaly_for(code, "2025-10-07").kind == "delayed_publication"
    assert official_anomaly_for("13874A", "2023-03-21") is None
    assert all(anomaly.action == "drop" for anomaly in OFFICIAL_ANOMALY_CALENDAR)
    assert anomaly_calendar_sha256().startswith("sha256:")
    assert len(anomaly_calendar_sha256()) == 71


def test_parser_tags_and_cleaner_audits_anomalous_rows():
    raw = (
        CSV_HEADER
        + _csv_row("bad", "2019-03-26", "13874A")
        + _csv_row("good", "2019-04-02", "13874A")
    )
    parsed = parse_legacy_cot_csv(raw)
    assert parsed[0].usable is False
    assert parsed[0].anomaly_id == "cftc-reporting-firm-correction-2019-03-26"

    cleaned = apply_official_anomaly_calendar(parsed)
    assert [record.source_id for record in cleaned.records] == ["good"]
    assert [item.source_id for item in cleaned.exclusions] == ["bad"]
    assert cleaned.exclusions[0].source_url.startswith("https://www.cftc.gov/")


def test_no_feature_is_visible_before_conservative_release_date():
    records = (_record("13874A", date(2023, 12, 5)),)

    assert latest_features_as_of(records, "2023-12-12")["13874A"] is None
    available = latest_features_as_of(records, "2023-12-13")["13874A"]
    assert available is not None
    assert available.available_date == date(2023, 12, 13)
    assert available.age_since_available_days == 0


def test_feature_is_neutralized_after_fourteen_days_of_carry():
    records = (_record("13874A", date(2023, 12, 5)),)

    assert latest_features_as_of(records, "2023-12-27")["13874A"] is not None
    assert latest_features_as_of(records, "2023-12-28")["13874A"] is None


def test_anomalous_latest_week_is_dropped_instead_of_revised_history_leaking():
    records = (
        _record("13874A", date(2019, 3, 19), source_id="prior"),
        _record("13874A", date(2019, 3, 26), source_id="corrected"),
    )

    latest = latest_features_as_of(records, "2019-04-05")["13874A"]
    assert latest is not None
    assert latest.report_date == date(2019, 3, 19)


def test_feature_history_contains_only_released_non_anomalous_observations():
    records = (
        _record("13874A", date(2023, 12, 5), source_id="released"),
        _record("13874A", date(2023, 12, 12), source_id="future"),
        _record("209742", date(2023, 12, 5), source_id="other-market"),
    )

    history = feature_history_as_of(records, "13874A", "2023-12-19")
    assert [item.report_date for item in history] == [date(2023, 12, 5)]


def test_feature_access_after_2023_also_requires_explicit_opt_in():
    records = (_record("13874A", date(2023, 12, 26)),)
    with pytest.raises(Post2023AccessError, match="allow_post_2023=True"):
        latest_features_as_of(records, "2024-01-03")

    feature = latest_features_as_of(
        records, "2024-01-03", allow_post_2023=True
    )["13874A"]
    assert feature is not None
    assert feature.report_date == date(2023, 12, 26)


def test_direct_post_2023_record_cannot_bypass_parser_guard():
    records = (_record("13874A", date(2024, 1, 2)),)
    with pytest.raises(Post2023AccessError, match="allow_post_2023=True"):
        feature_history_as_of(records, "13874A", "2023-12-31")


def test_trailing_zscore_uses_prior_weeks_and_excludes_current_value():
    def feature(day: int, share: float) -> COTFeature:
        return COTFeature(
            contract_code="13874A",
            market_key="sp500_emini",
            report_date=date(2023, 1, day),
            available_date=date(2023, 1, day),
            age_since_available_days=0,
            noncommercial_net=int(share * 1000),
            noncommercial_net_share=share,
        )

    history = (feature(3, 0.1), feature(10, 0.2), feature(17, 0.4))
    assert trailing_net_share_zscore(history, lookback_weeks=2) == pytest.approx(5.0)
    assert trailing_net_share_zscore(history[:2], lookback_weeks=2) is None


def test_trailing_zscore_neutralizes_constant_baseline_and_rejects_short_lookback():
    repeated = tuple(
        COTFeature(
            contract_code="13874A",
            market_key="sp500_emini",
            report_date=date(2023, 1, 3 + index * 7),
            available_date=date(2023, 1, 3 + index * 7),
            age_since_available_days=0,
            noncommercial_net=100,
            noncommercial_net_share=0.1,
        )
        for index in range(3)
    )
    assert trailing_net_share_zscore(repeated, lookback_weeks=2) is None
    with pytest.raises(COTDataError, match="at least 2"):
        trailing_net_share_zscore(repeated, lookback_weeks=1)
