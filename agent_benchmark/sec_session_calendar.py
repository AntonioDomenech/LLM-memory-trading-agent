"""Frozen NYSE full-session calendar used by the SEC availability audit."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
import hashlib
from typing import Sequence

from .sec_point_in_time import SecPointInTimeError


CALENDAR_ID = "nyse_full_sessions_2000_01_01_2025_01_10_v1"
CALENDAR_START = date(2000, 1, 1)
CALENDAR_END = date(2025, 1, 10)
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


def expected_nyse_sessions() -> tuple[str, ...]:
    holidays: set[date] = set()
    for year in range(CALENDAR_START.year, CALENDAR_END.year + 1):
        holidays.update(_holidays(year))
    result: list[str] = []
    current = CALENDAR_START
    while current <= CALENDAR_END:
        if current.weekday() < 5 and current not in holidays:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(result)


EXPECTED_SESSIONS = expected_nyse_sessions()


def validate_aapl_session_calendar(values: Sequence[str]) -> dict[str, object]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("AAPL session calendar must be a sequence of ISO dates")
    observed = tuple(values)
    if observed != EXPECTED_SESSIONS:
        raise SecPointInTimeError(
            "AAPL sessions do not match the frozen NYSE 2000-2025 calendar"
        )
    payload = "".join(f"{value}\n" for value in observed).encode("ascii")
    return {
        "calendar_id": CALENDAR_ID,
        "start": CALENDAR_START.isoformat(),
        "end": CALENDAR_END.isoformat(),
        "session_count": len(observed),
        "sessions_sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "sessions": list(observed),
    }


__all__ = [
    "CALENDAR_END",
    "CALENDAR_ID",
    "CALENDAR_START",
    "EXPECTED_SESSIONS",
    "SPECIAL_CLOSURES",
    "expected_nyse_sessions",
    "validate_aapl_session_calendar",
]
