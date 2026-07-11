from __future__ import annotations

import pytest

from agent_benchmark.cftc_cot_experiment import expected_pre_2024_nyse_sessions
from agent_benchmark.sec_point_in_time import SecPointInTimeError
from agent_benchmark.sec_session_calendar import (
    CALENDAR_END,
    CALENDAR_START,
    EXPECTED_SESSIONS,
    SPECIAL_CLOSURES,
    validate_aapl_session_calendar,
)


def test_frozen_calendar_has_expected_bounds_and_known_closures() -> None:
    assert EXPECTED_SESSIONS[0] == "2000-01-03"
    assert EXPECTED_SESSIONS[-1] == CALENDAR_END.isoformat()
    assert CALENDAR_START.isoformat() not in EXPECTED_SESSIONS
    assert "2001-09-11" not in EXPECTED_SESSIONS
    assert "2012-10-29" not in EXPECTED_SESSIONS
    assert "2024-06-19" not in EXPECTED_SESSIONS
    assert "2025-01-09" not in EXPECTED_SESSIONS
    assert {value.isoformat() for value in SPECIAL_CLOSURES}.isdisjoint(
        EXPECTED_SESSIONS
    )


def test_validation_is_exact_and_hashes_complete_sequence() -> None:
    evidence = validate_aapl_session_calendar(EXPECTED_SESSIONS)
    assert evidence["session_count"] == len(EXPECTED_SESSIONS)
    assert evidence["sessions"] == list(EXPECTED_SESSIONS)
    assert str(evidence["sessions_sha256"]).startswith("sha256:")

    with pytest.raises(SecPointInTimeError, match="do not match"):
        validate_aapl_session_calendar(EXPECTED_SESSIONS[:-1])
    swapped = list(EXPECTED_SESSIONS)
    swapped[100], swapped[101] = swapped[101], swapped[100]
    with pytest.raises(SecPointInTimeError, match="do not match"):
        validate_aapl_session_calendar(swapped)


def test_calendar_matches_independently_sealed_pre_2024_calendar() -> None:
    existing = expected_pre_2024_nyse_sessions(
        start="2000-01-01", end="2023-12-31"
    )
    expected = tuple(value.date().isoformat() for value in existing)
    observed = tuple(value for value in EXPECTED_SESSIONS if value <= "2023-12-31")
    assert observed == expected
