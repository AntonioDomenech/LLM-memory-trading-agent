from __future__ import annotations

import copy
import math
import random
from typing import Any

import pandas as pd
import pytest

from agent_benchmark.chronological_exhaustion_expert import (
    build_fixed_expert_signals,
)
from agent_benchmark.sec_gemma_online_risk_overlay_baseline import (
    SecGemmaOnlineRiskOverlayBaselineError,
    build_baseline_input_row,
    build_frozen_baseline_signal_batch,
    validate_frozen_baseline_signal_batch,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BASELINE_SOURCE_SHA256,
    canonical_sha256,
)


def _rows(count: int = 180) -> list[dict[str, Any]]:
    sessions = pd.bdate_range("2000-01-03", periods=count)
    aapl_close = [220.0 - 0.2 * position for position in range(count)]
    spy_close = [500.0 - 0.4 * position for position in range(count)]
    qqq_close = [400.0 - 0.35 * position for position in range(count)]
    result = []
    for position, stamp in enumerate(sessions):
        result.append(
            build_baseline_input_row(
                session=stamp.strftime("%Y-%m-%d"),
                aapl_raw_open=aapl_close[position],
                aapl_raw_close=aapl_close[position],
                aapl_adjusted_close=aapl_close[position],
                spy_adjusted_close=spy_close[position],
                qqq_adjusted_close=qqq_close[position],
                market_evidence_row_sha256=f"{position + 1:064x}",
            )
        )
    return result


def _rehash(row: dict[str, Any]) -> None:
    body = {
        key: value
        for key, value in row.items()
        if key != "baseline_input_row_sha256"
    }
    row["baseline_input_row_sha256"] = canonical_sha256(body)


def _pinned_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aapl_open": [
                float.fromhex(row["aapl_raw_open_hex"]) for row in rows
            ],
            "aapl_close": [
                float.fromhex(row["aapl_raw_close_hex"]) for row in rows
            ],
            "aapl_adj_close": [
                float.fromhex(row["aapl_adjusted_close_hex"]) for row in rows
            ],
            "spy_adj_close": [
                float.fromhex(row["spy_adjusted_close_hex"]) for row in rows
            ],
            "qqq_adj_close": [
                float.fromhex(row["qqq_adjusted_close_hex"]) for row in rows
            ],
        },
        index=pd.DatetimeIndex([row["session"] for row in rows]),
    )


def _assert_matches_pinned(rows: list[dict[str, Any]]) -> None:
    batch = build_frozen_baseline_signal_batch(rows)
    pinned = build_fixed_expert_signals(_pinned_frame(rows))
    for name in (
        "contextual_raw_signal",
        "contextual_virtual_signal",
        "weak_trend_raw_signal",
        "weak_trend_virtual_signal",
        "unfiltered_union_candidate_signal",
        "unfiltered_union_signal",
    ):
        assert [
            row[name] for row in batch["diagnostic_rows"]
        ] == pinned[name].astype(bool).tolist()


def _randomized_rows(
    *,
    seed: int,
    count: int,
    quantile_ties: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sessions = pd.bdate_range("2000-01-03", periods=count)
    aapl = 200.0
    spy = 500.0
    qqq = 400.0
    result: list[dict[str, Any]] = []
    for position, stamp in enumerate(sessions):
        if quantile_ties:
            intraday = (0.0, 0.0, 0.01, -0.01)[position % 4]
            aapl *= 1.0 + (0.0001, -0.0001, 0.0)[position % 3]
            spy *= 1.0 + (-0.001 if position % 11 < 7 else 0.001)
            qqq *= 1.0 + (-0.0012 if position % 13 < 8 else 0.001)
        else:
            intraday = rng.choice(
                (0.0, 0.0, 0.002, -0.002, rng.uniform(-0.03, 0.05))
            )
            aapl *= math.exp(rng.uniform(-0.02, 0.02))
            spy *= math.exp(rng.uniform(-0.012, 0.012))
            qqq *= math.exp(rng.uniform(-0.015, 0.015))
        result.append(
            build_baseline_input_row(
                session=stamp.strftime("%Y-%m-%d"),
                aapl_raw_open=aapl / (1.0 + intraday),
                aapl_raw_close=aapl,
                aapl_adjusted_close=aapl,
                spy_adjusted_close=spy,
                qqq_adjusted_close=qqq,
                market_evidence_row_sha256=canonical_sha256(
                    {"seed": seed, "position": position}
                ),
            )
        )
    return result


def test_adapter_uses_exact_pinned_baseline_and_only_raw_union_signal() -> None:
    rows = _rows()
    batch = build_frozen_baseline_signal_batch(rows)

    assert batch["baseline_source_sha256"] == BASELINE_SOURCE_SHA256
    assert batch["forbidden_actionable_target_used"] is False
    assert len(batch["signal_rows"]) == len(rows)
    assert set(batch["signal_rows"][0]) == {
        "session",
        "unfiltered_union_signal",
        "baseline_signal_sha256",
    }
    assert validate_frozen_baseline_signal_batch(
        batch,
        expected_baseline_batch_sha256=batch["baseline_batch_sha256"],
        input_rows=rows,
    ) == batch["baseline_batch_sha256"]


def test_known_exhaustion_event_is_projected_after_completed_close() -> None:
    rows = _rows()
    position = 130
    close = float.fromhex(rows[position]["aapl_raw_close_hex"])
    rows[position]["aapl_raw_open_hex"] = (close / 1.20).hex()
    _rehash(rows[position])
    batch = build_frozen_baseline_signal_batch(rows)

    assert batch["diagnostic_rows"][position][
        "unfiltered_union_candidate_signal"
    ] is True
    assert batch["signal_rows"][position][
        "unfiltered_union_signal"
    ] is True


def test_pure_baseline_matches_pinned_fixture_exactly() -> None:
    rows = _rows(220)
    for position in (130, 180):
        close = float.fromhex(rows[position]["aapl_raw_close_hex"])
        rows[position]["aapl_raw_open_hex"] = (close / 1.20).hex()
        _rehash(rows[position])
    _assert_matches_pinned(rows)


@pytest.mark.parametrize("count", (125, 126, 127, 180, 260))
@pytest.mark.parametrize("seed", range(8))
def test_pure_baseline_matches_pinned_randomized_boundaries(
    count: int,
    seed: int,
) -> None:
    _assert_matches_pinned(
        _randomized_rows(
            seed=seed,
            count=count,
            quantile_ties=False,
        )
    )


@pytest.mark.parametrize("count", (125, 126, 127, 180, 260))
def test_pure_baseline_matches_pinned_quantile_ties(count: int) -> None:
    _assert_matches_pinned(
        _randomized_rows(
            seed=0,
            count=count,
            quantile_ties=True,
        )
    )


def test_pure_and_pinned_baselines_both_reject_nan_prices() -> None:
    rows = _rows(127)
    rows[-1]["aapl_adjusted_close_hex"] = math.nan.hex()
    _rehash(rows[-1])

    with pytest.raises(
        SecGemmaOnlineRiskOverlayBaselineError,
        match="positive finite",
    ):
        build_frozen_baseline_signal_batch(rows)
    with pytest.raises(ValueError, match="finite"):
        build_fixed_expert_signals(_pinned_frame(rows))


def test_appending_future_rows_cannot_rewrite_completed_prefix_signals() -> None:
    rows = _rows(220)
    position = 130
    close = float.fromhex(rows[position]["aapl_raw_close_hex"])
    rows[position]["aapl_raw_open_hex"] = (close / 1.20).hex()
    _rehash(rows[position])
    prefix = build_frozen_baseline_signal_batch(rows[:180])
    full = build_frozen_baseline_signal_batch(rows)

    assert full["signal_rows"][:180] == prefix["signal_rows"]
    assert full["diagnostic_rows"][:180] == prefix["diagnostic_rows"]


def test_future_price_mutation_does_not_change_prior_signal() -> None:
    rows = _rows(220)
    baseline = build_frozen_baseline_signal_batch(rows)
    changed = copy.deepcopy(rows)
    changed[-1]["aapl_raw_open_hex"] = (
        float.fromhex(changed[-1]["aapl_raw_open_hex"]) * 3.0
    ).hex()
    _rehash(changed[-1])
    replayed = build_frozen_baseline_signal_batch(changed)

    assert replayed["signal_rows"][:-1] == baseline["signal_rows"][:-1]


def test_invalid_or_tampered_market_input_fails_closed() -> None:
    rows = _rows()
    rows[4]["aapl_raw_open_hex"] = math.nan.hex()
    _rehash(rows[4])

    with pytest.raises(
        SecGemmaOnlineRiskOverlayBaselineError,
        match="positive finite",
    ):
        build_frozen_baseline_signal_batch(rows)

    rows = _rows()
    rows[4]["aapl_raw_open_hex"] = 1.0.hex()
    with pytest.raises(
        SecGemmaOnlineRiskOverlayBaselineError,
        match="self-hash",
    ):
        build_frozen_baseline_signal_batch(rows)
