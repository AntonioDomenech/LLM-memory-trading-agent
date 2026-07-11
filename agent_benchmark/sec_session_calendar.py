"""Versioned NYSE trading-session-date calendars for the SEC audit.

This module freezes *dates* on which the NYSE had a trading session.  It does
not attest to the intraday opening or closing time of any session.  In
particular, an early-close date is still present as a trading-session date.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
import hashlib
import json
from typing import Sequence

from .sec_point_in_time import SecPointInTimeError


# The legacy v1 calendar is part of already-sealed SEC-audit evidence.  Keep
# its identity, bounds, closures, generated sequence, and validator immutable.
LEGACY_CALENDAR_ID = "nyse_full_sessions_2000_01_01_2025_01_10_v1"
LEGACY_CALENDAR_START = date(2000, 1, 1)
LEGACY_CALENDAR_END = date(2025, 1, 10)

# v2 is the current calendar used by new audits.  It is an append-only
# extension of v1 through the final requested 2026-YTD session.
CALENDAR_ID = "nyse_trading_session_dates_2000_01_01_2026_07_10_v2"
CALENDAR_START = LEGACY_CALENDAR_START
CALENDAR_END = date(2026, 7, 10)

# A separate market-feature calendar supplies pre-2000 lookback sessions.  It
# never changes the filing-universe start or the frozen experiment score dates.
MARKET_HISTORY_CALENDAR_ID = (
    "nyse_trading_session_dates_1998_01_01_2026_07_10_v1"
)
MARKET_HISTORY_CALENDAR_START = date(1998, 1, 1)

SPECIAL_CLOSURES = frozenset(
    date.fromisoformat(value)
    for value in (
        "2001-09-11",
        "2001-09-12",
        "2001-09-13",
        "2001-09-14",
        "2004-06-11",
        "2007-01-02",
        "2012-10-29",
        "2012-10-30",
        "2018-12-05",
        "2025-01-09",
    )
)


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(
        days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1)
    )


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nearest_weekday(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _holidays(year: int) -> set[date]:
    new_year = date(year, 1, 1)
    observed_new_year = (
        new_year + timedelta(days=1) if new_year.weekday() == 6 else new_year
    )
    values = {
        observed_new_year,
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _nearest_weekday(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _nearest_weekday(date(year, 12, 25)),
    }
    if year >= 2022:
        values.add(_nearest_weekday(date(year, 6, 19)))
    values.update(value for value in SPECIAL_CLOSURES if value.year == year)
    return values


def _session_dates(*, start: date, end: date) -> tuple[str, ...]:
    holidays: set[date] = set()
    for year in range(start.year, end.year + 1):
        holidays.update(_holidays(year))
    result: list[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(result)


def expected_legacy_nyse_sessions() -> tuple[str, ...]:
    """Return the immutable legacy-v1 trading-session-date sequence."""

    return _session_dates(start=LEGACY_CALENDAR_START, end=LEGACY_CALENDAR_END)


LEGACY_EXPECTED_SESSIONS = expected_legacy_nyse_sessions()


def expected_nyse_sessions() -> tuple[str, ...]:
    """Return the current v2 trading-session-date sequence."""

    return _session_dates(start=CALENDAR_START, end=CALENDAR_END)


EXPECTED_SESSIONS = expected_nyse_sessions()


def expected_market_history_sessions() -> tuple[str, ...]:
    """Return exact session dates used only for lagged market features."""

    return _session_dates(start=MARKET_HISTORY_CALENDAR_START, end=CALENDAR_END)


EXPECTED_MARKET_HISTORY_SESSIONS = expected_market_history_sessions()


def _newline_payload(values: Sequence[str]) -> bytes:
    return "".join(f"{value}\n" for value in values).encode("ascii")


def _canonical_json_payload(values: Sequence[str]) -> bytes:
    return json.dumps(
        list(values),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _calendar_evidence(
    values: tuple[str, ...],
    *,
    calendar_id: str,
    start: date,
    end: date,
    include_canonical_json_sha256: bool,
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "calendar_id": calendar_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "session_count": len(values),
        "sessions_sha256": (
            f"sha256:{hashlib.sha256(_newline_payload(values)).hexdigest()}"
        ),
        "sessions": list(values),
    }
    if include_canonical_json_sha256:
        evidence["sessions_canonical_json_sha256"] = (
            "sha256:"
            f"{hashlib.sha256(_canonical_json_payload(values)).hexdigest()}"
        )
    return evidence


def _validate_calendar_values(
    values: Sequence[str],
    *,
    expected: tuple[str, ...],
    error_label: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("AAPL session calendar must be a sequence of ISO dates")
    observed = tuple(values)
    if observed != expected:
        raise SecPointInTimeError(
            f"AAPL sessions do not match the frozen {error_label} calendar"
        )
    return observed


def validate_legacy_aapl_session_calendar(
    values: Sequence[str],
) -> dict[str, object]:
    """Validate and describe the exact legacy v1 calendar."""

    observed = _validate_calendar_values(
        values,
        expected=LEGACY_EXPECTED_SESSIONS,
        error_label="NYSE trading-session-date v1",
    )
    return _calendar_evidence(
        observed,
        calendar_id=LEGACY_CALENDAR_ID,
        start=LEGACY_CALENDAR_START,
        end=LEGACY_CALENDAR_END,
        include_canonical_json_sha256=False,
    )


def validate_aapl_session_calendar(values: Sequence[str]) -> dict[str, object]:
    """Validate and describe the exact current v2 calendar."""

    observed = _validate_calendar_values(
        values,
        expected=EXPECTED_SESSIONS,
        error_label="NYSE trading-session-date v2",
    )
    return _calendar_evidence(
        observed,
        calendar_id=CALENDAR_ID,
        start=CALENDAR_START,
        end=CALENDAR_END,
        include_canonical_json_sha256=True,
    )


def validate_market_history_session_calendar(
    values: Sequence[str],
) -> dict[str, object]:
    """Validate the exact prehistory-plus-experiment market-feature calendar."""

    observed = _validate_calendar_values(
        values,
        expected=EXPECTED_MARKET_HISTORY_SESSIONS,
        error_label="NYSE market-feature trading-session-date v1",
    )
    return _calendar_evidence(
        observed,
        calendar_id=MARKET_HISTORY_CALENDAR_ID,
        start=MARKET_HISTORY_CALENDAR_START,
        end=CALENDAR_END,
        include_canonical_json_sha256=True,
    )


__all__ = [
    "CALENDAR_END",
    "CALENDAR_ID",
    "CALENDAR_START",
    "EXPECTED_SESSIONS",
    "EXPECTED_MARKET_HISTORY_SESSIONS",
    "LEGACY_CALENDAR_END",
    "LEGACY_CALENDAR_ID",
    "LEGACY_CALENDAR_START",
    "LEGACY_EXPECTED_SESSIONS",
    "MARKET_HISTORY_CALENDAR_ID",
    "MARKET_HISTORY_CALENDAR_START",
    "SPECIAL_CLOSURES",
    "expected_legacy_nyse_sessions",
    "expected_nyse_sessions",
    "expected_market_history_sessions",
    "validate_aapl_session_calendar",
    "validate_legacy_aapl_session_calendar",
    "validate_market_history_session_calendar",
]
