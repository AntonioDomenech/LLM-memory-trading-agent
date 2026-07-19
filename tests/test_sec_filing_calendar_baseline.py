from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent_benchmark.sec_filing_calendar_baseline import (
    MANIFEST_BYTE_COUNT,
    MANIFEST_ROWS,
    MANIFEST_SHA256,
    _sha256,
    build_filing_target,
    load_filing_manifest,
)


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=70)
    values = pd.Series(range(100, 170), index=dates, dtype=float)
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": values,
            "qqq_adj_close": values,
        },
        index=dates,
    )


def _event(frame: pd.DataFrame, position: int, sequence: int) -> dict[str, object]:
    day = frame.index[position].date().isoformat()
    return {
        "sequence": sequence,
        "accession_number": f"accession-{sequence}",
        "form": "10-Q",
        "filing_date": (frame.index[position] - pd.Timedelta(days=1)).date().isoformat(),
        "availability_session": day,
    }


def test_target_sells_t_plus_1_and_buys_t_plus_21() -> None:
    frame = _frame()
    target, schedule = build_filing_target(frame, [_event(frame, 5, 1)])
    assert (target.iloc[5:25] == 0.0).all()
    assert target.iloc[4] == 1.0
    assert target.iloc[25] == 1.0
    row = schedule.iloc[0]
    assert row["entry_open"] == frame.index[6].date().isoformat()
    assert row["exit_open"] == frame.index[26].date().isoformat()
    assert bool(row["complete_by_2018"])


def test_active_episode_cannot_be_extended() -> None:
    frame = _frame()
    events = [
        _event(frame, 5, 1),
        _event(frame, 5, 2),
        _event(frame, 24, 3),
        _event(frame, 25, 4),
        _event(frame, 26, 5),
    ]
    target, schedule = build_filing_target(frame, events)
    assert schedule["scheduled"].tolist() == [True, False, False, False, True]
    assert (target.iloc[5:25] == 0.0).all()
    assert target.iloc[25] == 1.0
    assert (target.iloc[26:46] == 0.0).all()
    assert target.iloc[46] == 1.0


def test_real_private_manifest_matches_preregistration_when_available() -> None:
    private = Path(
        r"C:\Users\anton\Documents\LLM-memory-trading-agent\data"
        r"\aapl_sec_gemma_lean_evidence_v3_8\development\parse_receipts"
    )
    if not private.is_dir():
        return
    rows, payload = load_filing_manifest(private)
    assert len(rows) == MANIFEST_ROWS
    assert len(payload) == MANIFEST_BYTE_COUNT
    assert _sha256(payload) == MANIFEST_SHA256
    assert rows[0]["availability_session"] == "2000-02-02"
    assert rows[-1]["availability_session"] == "2018-11-06"
