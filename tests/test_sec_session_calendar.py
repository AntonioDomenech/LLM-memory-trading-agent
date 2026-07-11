from __future__ import annotations

from bisect import insort

import pytest

from agent_benchmark.cftc_cot_experiment import expected_pre_2024_nyse_sessions
from agent_benchmark.sec_point_in_time import SecPointInTimeError
from agent_benchmark.sec_session_calendar import (
    CALENDAR_END,
    CALENDAR_ID,
    CALENDAR_START,
    EXPECTED_SESSIONS,
    EXPECTED_MARKET_HISTORY_SESSIONS,
    LEGACY_CALENDAR_END,
    LEGACY_CALENDAR_ID,
    LEGACY_CALENDAR_START,
    LEGACY_EXPECTED_SESSIONS,
    MARKET_HISTORY_CALENDAR_ID,
    MARKET_HISTORY_CALENDAR_START,
    SPECIAL_CLOSURES,
    validate_aapl_session_calendar,
    validate_legacy_aapl_session_calendar,
    validate_market_history_session_calendar,
)


LEGACY_SESSION_COUNT = 6_295
LEGACY_NEWLINE_SHA256 = (
    "sha256:46d40710c48883b43f4666c8d0d1ba0a1d34b308eb2a3cb152f86853f3bb7520"
)
CURRENT_SESSION_COUNT = 6_669
CURRENT_NEWLINE_SHA256 = (
    "sha256:f3ea99a9fcbe99187701acb030cd6fd0f6dc2c7be53528ef7f5adb13a1c5fdd3"
)
CURRENT_CANONICAL_JSON_SHA256 = (
    "sha256:e0550f12f98d7e0cf38d6797ae0d0e410bb3f9006793830ed75e0f753ccc5cf9"
)
MARKET_HISTORY_SESSION_COUNT = 7_173
MARKET_HISTORY_NEWLINE_SHA256 = (
    "sha256:df335e3d907a517e4072bbbc005433d7ff4cb930631740849b412700989d7ae6"
)
MARKET_HISTORY_CANONICAL_JSON_SHA256 = (
    "sha256:e9d37d63d158f8a3b6de58ef81970b27bcac9edb6a9c7b9e2f3ffe477d2f3032"
)


def test_legacy_v1_identity_bounds_count_and_hashes_are_immutable() -> None:
    assert LEGACY_CALENDAR_ID == "nyse_full_sessions_2000_01_01_2025_01_10_v1"
    assert LEGACY_CALENDAR_START.isoformat() == "2000-01-01"
    assert LEGACY_CALENDAR_END.isoformat() == "2025-01-10"
    assert len(LEGACY_EXPECTED_SESSIONS) == LEGACY_SESSION_COUNT
    assert LEGACY_EXPECTED_SESSIONS[0] == "2000-01-03"
    assert LEGACY_EXPECTED_SESSIONS[-1] == "2025-01-10"

    evidence = validate_legacy_aapl_session_calendar(LEGACY_EXPECTED_SESSIONS)
    assert evidence == {
        "calendar_id": LEGACY_CALENDAR_ID,
        "start": "2000-01-01",
        "end": "2025-01-10",
        "session_count": LEGACY_SESSION_COUNT,
        "sessions_sha256": LEGACY_NEWLINE_SHA256,
        "sessions": list(LEGACY_EXPECTED_SESSIONS),
    }


def test_current_v2_identity_bounds_count_and_hashes_are_exact() -> None:
    assert CALENDAR_ID == "nyse_trading_session_dates_2000_01_01_2026_07_10_v2"
    assert CALENDAR_START == LEGACY_CALENDAR_START
    assert CALENDAR_END.isoformat() == "2026-07-10"
    assert len(EXPECTED_SESSIONS) == CURRENT_SESSION_COUNT
    assert EXPECTED_SESSIONS[0] == "2000-01-03"
    assert EXPECTED_SESSIONS[-1] == "2026-07-10"

    evidence = validate_aapl_session_calendar(EXPECTED_SESSIONS)
    assert evidence == {
        "calendar_id": CALENDAR_ID,
        "start": "2000-01-01",
        "end": "2026-07-10",
        "session_count": CURRENT_SESSION_COUNT,
        "sessions_sha256": CURRENT_NEWLINE_SHA256,
        "sessions_canonical_json_sha256": CURRENT_CANONICAL_JSON_SHA256,
        "sessions": list(EXPECTED_SESSIONS),
    }


def test_v2_is_an_exact_append_only_extension_of_legacy_v1() -> None:
    prefix_length = len(LEGACY_EXPECTED_SESSIONS)
    assert EXPECTED_SESSIONS[:prefix_length] == LEGACY_EXPECTED_SESSIONS
    assert EXPECTED_SESSIONS[prefix_length : prefix_length + 3] == (
        "2025-01-13",
        "2025-01-14",
        "2025-01-15",
    )
    assert EXPECTED_SESSIONS[-1] == "2026-07-10"


def test_market_feature_calendar_exactly_supplies_pre_2000_lookback() -> None:
    assert MARKET_HISTORY_CALENDAR_ID.endswith("1998_01_01_2026_07_10_v1")
    assert MARKET_HISTORY_CALENDAR_START.isoformat() == "1998-01-01"
    assert len(EXPECTED_MARKET_HISTORY_SESSIONS) == MARKET_HISTORY_SESSION_COUNT
    assert EXPECTED_MARKET_HISTORY_SESSIONS[0] == "1998-01-02"
    assert EXPECTED_MARKET_HISTORY_SESSIONS[-1] == "2026-07-10"
    assert EXPECTED_MARKET_HISTORY_SESSIONS[-len(EXPECTED_SESSIONS) :] == EXPECTED_SESSIONS
    assert sum(value < "2000-01-01" for value in EXPECTED_MARKET_HISTORY_SESSIONS) == 504
    evidence = validate_market_history_session_calendar(
        EXPECTED_MARKET_HISTORY_SESSIONS
    )
    assert evidence["sessions_sha256"] == MARKET_HISTORY_NEWLINE_SHA256
    assert (
        evidence["sessions_canonical_json_sha256"]
        == MARKET_HISTORY_CANONICAL_JSON_SHA256
    )


def test_known_holidays_and_special_closures_are_not_session_dates() -> None:
    assert CALENDAR_START.isoformat() not in EXPECTED_SESSIONS
    assert "2001-09-11" not in EXPECTED_SESSIONS
    assert "2012-10-29" not in EXPECTED_SESSIONS
    assert "2024-06-19" not in EXPECTED_SESSIONS
    assert "2025-01-09" not in EXPECTED_SESSIONS
    assert "2025-07-04" not in EXPECTED_SESSIONS
    assert "2025-12-25" not in EXPECTED_SESSIONS
    assert "2026-01-01" not in EXPECTED_SESSIONS
    assert "2026-04-03" not in EXPECTED_SESSIONS
    assert "2026-06-19" not in EXPECTED_SESSIONS
    assert "2026-07-03" not in EXPECTED_SESSIONS
    assert {value.isoformat() for value in SPECIAL_CLOSURES}.isdisjoint(
        EXPECTED_SESSIONS
    )


def test_known_early_close_dates_remain_trading_session_dates() -> None:
    # This artifact freezes trading dates, not intraday hours.  Removing an
    # early-close date would therefore be a calendar corruption.
    assert "2024-11-29" in LEGACY_EXPECTED_SESSIONS
    assert "2024-12-24" in LEGACY_EXPECTED_SESSIONS
    assert "2025-07-03" in EXPECTED_SESSIONS
    assert "2025-11-28" in EXPECTED_SESSIONS
    assert "2025-12-24" in EXPECTED_SESSIONS


def test_validation_rejects_truncation_reordering_and_wrong_version() -> None:
    with pytest.raises(SecPointInTimeError, match="do not match"):
        validate_aapl_session_calendar(EXPECTED_SESSIONS[:-1])

    swapped = list(EXPECTED_SESSIONS)
    swapped[100], swapped[101] = swapped[101], swapped[100]
    with pytest.raises(SecPointInTimeError, match="do not match"):
        validate_aapl_session_calendar(swapped)

    with pytest.raises(SecPointInTimeError, match="v2"):
        validate_aapl_session_calendar(LEGACY_EXPECTED_SESSIONS)
    with pytest.raises(SecPointInTimeError, match="v1"):
        validate_legacy_aapl_session_calendar(EXPECTED_SESSIONS)


def test_validation_rejects_inserting_a_closed_holiday() -> None:
    mutated = list(EXPECTED_SESSIONS)
    insort(mutated, "2026-07-03")
    with pytest.raises(SecPointInTimeError, match="do not match"):
        validate_aapl_session_calendar(mutated)


def test_validation_rejects_removing_an_early_close_session_date() -> None:
    mutated = list(EXPECTED_SESSIONS)
    mutated.remove("2025-11-28")
    with pytest.raises(SecPointInTimeError, match="do not match"):
        validate_aapl_session_calendar(mutated)


def test_validation_rejects_text_and_bytes_instead_of_a_date_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        validate_aapl_session_calendar("2026-07-10")
    with pytest.raises(TypeError, match="sequence"):
        validate_legacy_aapl_session_calendar(b"2025-01-10")


def test_calendar_matches_independently_sealed_pre_2024_calendar() -> None:
    existing = expected_pre_2024_nyse_sessions(
        start="2000-01-01", end="2023-12-31"
    )
    expected = tuple(value.date().isoformat() for value in existing)
    observed = tuple(value for value in EXPECTED_SESSIONS if value <= "2023-12-31")
    assert observed == expected
