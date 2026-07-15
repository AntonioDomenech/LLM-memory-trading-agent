from __future__ import annotations

import ast
import copy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agent_benchmark.contextual_expert_aggregation_replay as replay_module
from agent_benchmark.chronological_exhaustion_expert import (
    canonicalize_one_session_signals,
)
from agent_benchmark.contextual_expert_aggregation import (
    CAUSAL_ONLINE_MODE,
    EXPERT_NAMES,
    FROZEN_CUTOFF_MODE,
    FULL_MODE,
    GLOBAL_ONLY_MODE,
    LIFETIME_ONLY_MODE,
    SCALE_NAMES,
)
from agent_benchmark.contextual_expert_aggregation_replay import (
    COMPARATOR_TARGET_COLUMNS,
    CONFIRMATION_ARM_ORDER,
    FROZEN_2018_ARM,
    GLOBAL_ONLY_ARM,
    LIFETIME_ONLY_ARM,
    ONLINE_FULL_ARM,
    ReplayCheckpoint,
    build_fixed_opportunity_frame,
    canonical_market_frame,
    continue_from_checkpoint,
    fork_confirmation_arms,
    prove_replay_prefix,
    replay_from_empty,
    verify_resume_equivalence,
)


def _market_frame(
    index: pd.DatetimeIndex,
    *,
    adjusted_opens: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    count = len(index)
    opens = (
        np.full(count, 100.0, dtype=float)
        if adjusted_opens is None
        else np.asarray(adjusted_opens, dtype=float)
    )
    if len(opens) != count:
        raise ValueError("adjusted_opens length mismatch")
    trend = np.arange(count, dtype=float)
    return pd.DataFrame(
        {
            "aapl_open": opens,
            "aapl_close": np.full(count, 100.0),
            "aapl_adj_close": np.full(count, 100.0),
            "spy_adj_close": 100.0 + trend * 0.02,
            "qqq_adj_close": 100.0 + trend * 0.03,
        },
        index=pd.DatetimeIndex(index, name="date"),
    )


def _fixed_signal_builder(
    *,
    contextual_raw_dates: set[str] | None = None,
    weak_raw_dates: set[str] | None = None,
):
    contextual_dates = contextual_raw_dates or set()
    weak_dates = weak_raw_dates or set()

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        index = pd.DatetimeIndex(frame.index, name="date")
        contextual_raw = pd.Series(
            [date.date().isoformat() in contextual_dates for date in index],
            index=index,
            dtype=bool,
        )
        weak_raw = pd.Series(
            [date.date().isoformat() in weak_dates for date in index],
            index=index,
            dtype=bool,
        )
        contextual = canonicalize_one_session_signals(contextual_raw)
        weak = canonicalize_one_session_signals(weak_raw)
        candidate = (contextual | weak).astype(bool)
        union = canonicalize_one_session_signals(candidate)
        length = len(index)
        result = pd.DataFrame(
            {
                "aapl_intraday_return": np.zeros(length),
                "contextual_prior_intraday_percentile": np.zeros(length),
                "contextual_spy_return_10": np.zeros(length),
                "contextual_qqq_return_10": np.zeros(length),
                "contextual_ready": np.ones(length, dtype=bool),
                "contextual_raw_signal": contextual_raw,
                "contextual_virtual_signal": contextual,
                "contextual_virtual_signal_blocked": contextual_raw & ~contextual,
                "contextual_virtual_signal_pending_stage_outcome": np.zeros(
                    length, dtype=bool
                ),
                "weak_trend_prior_intraday_percentile": np.zeros(length),
                "weak_trend_spy_return_20": np.zeros(length),
                "weak_trend_qqq_return_20": np.zeros(length),
                "weak_trend_aapl_sma_20": np.full(length, 100.0),
                "weak_trend_ready": np.ones(length, dtype=bool),
                "weak_trend_raw_signal": weak_raw,
                "weak_trend_virtual_signal": weak,
                "weak_trend_virtual_signal_blocked": weak_raw & ~weak,
                "weak_trend_virtual_signal_pending_stage_outcome": np.zeros(
                    length, dtype=bool
                ),
                "unfiltered_union_candidate_signal": candidate,
                "unfiltered_union_signal": union,
                "unfiltered_union_signal_blocked": candidate & ~union,
                "unfiltered_union_signal_pending_stage_outcome": np.zeros(
                    length, dtype=bool
                ),
                # Deliberately false everywhere: replay must not use this tail
                # availability field to delete an accepted opportunity.
                "stage_outcome_available": np.zeros(length, dtype=bool),
                "always_long_target_exposure": np.ones(length),
                "unfiltered_contextual_target_exposure": np.ones(length),
                "unfiltered_weak_trend_target_exposure": np.ones(length),
                "unfiltered_union_target_exposure": np.ones(length),
            },
            index=index,
        )
        return result

    return build


def _patch_signals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contextual_positions: list[int],
    weak_positions: list[int] | None,
    index: pd.DatetimeIndex,
) -> None:
    contextual = {index[position].date().isoformat() for position in contextual_positions}
    weak = {
        index[position].date().isoformat()
        for position in (weak_positions or [])
    }
    monkeypatch.setattr(
        replay_module,
        "build_fixed_expert_signals",
        _fixed_signal_builder(
            contextual_raw_dates=contextual,
            weak_raw_dates=weak,
        ),
    )


def _rehash_checkpoint_payload(payload: dict) -> dict:
    state = payload["model_payload"]["state"]
    payload["model_state_sha256"] = replay_module._sha256_hex(state)
    payload["pending_state_sha256"] = replay_module._sha256_hex(
        state["pending_lessons"]
    )
    payload["market_history_tail_sha256"] = replay_module._sha256_hex(
        state["market_history"]
    )
    payload["signal_cooldown_tail_sha256"] = replay_module._sha256_hex(
        payload["signal_cooldown_tail"]
    )
    without_digest = {
        key: value for key, value in payload.items() if key != "digest_sha256"
    }
    payload["digest_sha256"] = replay_module._sha256_hex(without_digest)
    return payload


def test_fixed_frame_uses_union_gated_individual_comparators(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=4, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[0, 1, 2],
        weak_positions=[1],
        index=index,
    )
    fixed = build_fixed_opportunity_frame(_market_frame(index))

    assert fixed["contextual_virtual_signal"].tolist() == [True, False, True, False]
    assert fixed["unfiltered_union_candidate_signal"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert fixed["unfiltered_union_signal"].tolist() == [True, False, True, False]
    assert not any("stage_outcome" in column for column in fixed.columns)
    assert fixed["fixed_union_cash_target_exposure"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert fixed["fixed_contextual_only_target_exposure"].tolist() == [
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    # The weak expert contributed to the union-suppressed candidate at row 1,
    # but an individual comparator cannot resurrect a non-opportunity.
    assert fixed["fixed_weak_trend_only_target_exposure"].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_tail_opportunity_is_pending_and_long_action_still_learns(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=3, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[0],
        weak_positions=[],
        index=index,
    )
    first = replay_from_empty(_market_frame(index[:1]))
    assert first.forecast.iloc[0]["action"] == "LONG"
    assert first.forecast.iloc[0]["pending_lesson_count"] == 1
    assert first.checkpoint.pending_lessons[0]["sessions_until_maturity"] == 2

    completed = continue_from_checkpoint(
        _market_frame(index[1:]),
        first.checkpoint,
        historical_prefix=_market_frame(index[:1]),
    )
    assert len(completed.matured_lessons) == 1
    assert completed.matured_lessons.iloc[0]["signal_date"] == index[0].date().isoformat()
    assert completed.forecast.iloc[-1]["matured_lesson_count"] == 1
    assert completed.forecast.iloc[-1]["admitted_lesson_count"] == 1


def test_t_plus_two_update_happens_before_same_close_decision(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=3, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[0, 1, 2],
        weak_positions=[],
        index=index,
    )
    result = replay_from_empty(
        _market_frame(index, adjusted_opens=[100.0, 120.0, 100.0])
    )

    assert result.forecast.iloc[0]["action"] == "LONG"
    maturity_row = result.forecast.iloc[2]
    assert maturity_row["matured_on_close_count"] == 1
    assert maturity_row["matured_admitted"] is True or bool(
        maturity_row["matured_admitted"]
    )
    assert maturity_row["canonical_union_opportunity"]
    assert maturity_row["Q"] > 0.5
    assert maturity_row["action"] == "CASH"


def test_full_replay_equals_checkpoint_resume_across_cooldown_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=150, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[8, 9, 10, 127, 128, 129, 130, 145],
        weak_positions=[40, 41, 90],
        index=index,
    )
    opens = 100.0 + np.sin(np.arange(len(index))) * 5.0
    frame = _market_frame(index, adjusted_opens=opens)
    full = replay_from_empty(frame)
    prefix = replay_from_empty(frame.iloc[:128])
    resumed = continue_from_checkpoint(
        frame.iloc[128:],
        prefix.checkpoint,
        historical_prefix=frame.iloc[:128],
    )

    proof = verify_resume_equivalence(full, resumed)
    proof.require()
    assert full.checkpoint.to_dict() == resumed.checkpoint.to_dict()
    assert resumed.opportunity_frame.iloc[0]["unfiltered_union_signal"] is False or not bool(
        resumed.opportunity_frame.iloc[0]["unfiltered_union_signal"]
    )
    assert len(prefix.checkpoint.market_feature_tail) == 126
    assert len(prefix.checkpoint.signal_cooldown_tail) == 126
    assert set(prefix.checkpoint.cooldown_predecessor) == {
        "contextual_virtual_signal",
        "weak_trend_virtual_signal",
        "unfiltered_union_signal",
    }


def test_real_signal_prefix_is_invariant_to_later_stage_tail():
    index = pd.bdate_range("2001-01-01", periods=150, name="date")
    trend = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "aapl_open": 100.0 + trend * 0.01,
            "aapl_close": 100.0 + trend * 0.012,
            "aapl_adj_close": 100.0 + trend * 0.011,
            "spy_adj_close": 100.0 + trend * 0.02,
            "qqq_adj_close": 100.0 + trend * 0.03,
        },
        index=index,
    )
    truncated = build_fixed_opportunity_frame(frame.iloc[:140])
    longer = build_fixed_opportunity_frame(frame)
    direct = replay_module.build_fixed_expert_signals(frame)
    assert longer.loc[:, replay_module.FIXED_CAUSAL_SIGNAL_COLUMNS].equals(
        direct.loc[:, replay_module.FIXED_CAUSAL_SIGNAL_COLUMNS]
    )
    assert truncated.equals(longer.iloc[:140])
    assert not any("stage_outcome" in column for column in truncated.columns)
    assert tuple(column for column in COMPARATOR_TARGET_COLUMNS if column in truncated) == COMPARATOR_TARGET_COLUMNS

    full = replay_from_empty(frame)
    prefix = replay_from_empty(frame.iloc[:140])
    resumed = continue_from_checkpoint(
        frame.iloc[140:],
        prefix.checkpoint,
        historical_prefix=frame.iloc[:140],
    )
    verify_resume_equivalence(full, resumed).require()
    assert full.opportunity_frame.loc[resumed.opportunity_frame.index].equals(
        resumed.opportunity_frame
    )
    assert full.forecast.loc[resumed.forecast.index].equals(resumed.forecast)
    assert full.checkpoint.to_dict() == resumed.checkpoint.to_dict()


def test_real_builder_overlap_and_both_cooldowns_are_preserved_byte_exactly():
    index = pd.bdate_range("2001-01-01", periods=130, name="date")
    frame = _market_frame(index)
    decline = 200.0 - np.arange(len(index), dtype=float) * 0.25
    frame["spy_adj_close"] = decline
    frame["qqq_adj_close"] = decline * 1.1
    for position in (126, 127, 128):
        frame.iloc[position, frame.columns.get_loc("aapl_open")] = 100.0
        frame.iloc[position, frame.columns.get_loc("aapl_close")] = 120.0
        frame.iloc[position, frame.columns.get_loc("aapl_adj_close")] = 80.0

    fixed = build_fixed_opportunity_frame(frame)
    direct = replay_module.build_fixed_expert_signals(frame)
    assert fixed.loc[:, replay_module.FIXED_CAUSAL_SIGNAL_COLUMNS].equals(
        direct.loc[:, replay_module.FIXED_CAUSAL_SIGNAL_COLUMNS]
    )
    rows = fixed.iloc[126:129]
    assert rows["contextual_raw_signal"].tolist() == [True, True, True]
    assert rows["weak_trend_raw_signal"].tolist() == [True, True, True]
    assert rows["contextual_virtual_signal"].tolist() == [True, False, True]
    assert rows["weak_trend_virtual_signal"].tolist() == [True, False, True]
    assert rows["unfiltered_union_candidate_signal"].tolist() == [
        True,
        False,
        True,
    ]
    assert rows["unfiltered_union_signal"].tolist() == [True, False, True]
    assert rows["fixed_contextual_only_target_exposure"].tolist() == [
        0.0,
        1.0,
        0.0,
    ]
    assert rows["fixed_weak_trend_only_target_exposure"].tolist() == [
        0.0,
        1.0,
        0.0,
    ]


def test_prefix_proof_is_exact_and_detects_feature_tamper(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=12, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[2, 4, 10],
        weak_positions=[7],
        index=index,
    )
    frame = _market_frame(index)
    left = replay_from_empty(frame)
    right = replay_from_empty(frame.copy())
    proof = prove_replay_prefix(left, right)
    proof.require()
    assert proof.checkpoint_envelope_equal
    assert proof.runtime_equal
    assert proof.checkpoint_digest_equal

    right.opportunity_frame.iloc[0, 0] += 1.0
    assert not prove_replay_prefix(left, right).passed


def test_proof_exposes_runtime_and_complete_envelope_inequality(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=30, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[21, 23, 25, 27],
        weak_positions=[],
        index=index,
    )
    frame = _market_frame(index, adjusted_opens=100.0 + 5.0 * np.sin(np.arange(30)))
    online = replay_from_empty(frame)
    global_only = replay_from_empty(frame, ablation_mode=GLOBAL_ONLY_MODE)
    proof = prove_replay_prefix(online, global_only)
    assert not proof.passed
    assert not proof.runtime_equal
    assert not proof.checkpoint_envelope_equal
    assert not proof.checkpoint_digest_equal


def test_proofs_validate_checkpoints_and_require_the_complete_envelope(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=8, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[1, 6],
        weak_positions=[],
        index=index,
    )
    result = replay_from_empty(_market_frame(index))
    malformed_checkpoint = replace(
        result.checkpoint,
        digest_sha256="0" * 64,
    )
    malformed_result = replace(result, checkpoint=malformed_checkpoint)
    with pytest.raises(ValueError, match="envelope digest mismatch"):
        prove_replay_prefix(result, malformed_result)

    payload = result.checkpoint.to_dict()
    payload["forecast_prefix_sha256"] = "0" * 64
    forged = ReplayCheckpoint.from_dict(_rehash_checkpoint_payload(payload))
    forged_result = replace(result, checkpoint=forged)
    proof = prove_replay_prefix(result, forged_result)
    assert not proof.passed
    assert not proof.prefix_hashes_equal
    assert not proof.checkpoint_envelope_equal
    assert not proof.checkpoint_digest_equal


def test_resume_with_no_suffix_maturities_uses_canonical_empty_event_equality(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=12, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[0],
        weak_positions=[],
        index=index,
    )
    frame = _market_frame(index)
    full = replay_from_empty(frame)
    prefix = replay_from_empty(frame.iloc[:6])
    resumed = continue_from_checkpoint(
        frame.iloc[6:],
        prefix.checkpoint,
        historical_prefix=frame.iloc[:6],
    )
    assert full.matured_lessons.shape[0] == 1
    assert resumed.matured_lessons.empty
    proof = verify_resume_equivalence(full, resumed)
    proof.require()
    assert proof.matured_events_equal
    assert proof.checkpoint_envelope_equal


def test_confirmation_forks_share_checkpoint_and_arm_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    prefix_index = pd.bdate_range(end="2018-12-31", periods=30, name="date")
    later_index = pd.bdate_range("2019-01-01", periods=8, name="date")
    full_index = prefix_index.append(later_index)
    _patch_signals(
        monkeypatch,
        contextual_positions=[2, 5, 8, 12, 16, 20, 24, 29, 32, 34, 36],
        weak_positions=[10, 18, 27],
        index=full_index,
    )
    opens = np.full(len(full_index), 100.0)
    opens[3::3] = 120.0
    prefix_frame = _market_frame(
        prefix_index, adjusted_opens=opens[: len(prefix_index)]
    )
    later_frame = _market_frame(
        later_index, adjusted_opens=opens[len(prefix_index) :]
    )
    full_frame = pd.concat([prefix_frame, later_frame])
    prefix = replay_from_empty(prefix_frame)
    original_verify = replay_module._verify_complete_historical_prefix
    verification_calls: list[str] = []

    def counted_verify(*args, **kwargs):
        verification_calls.append("verified")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        replay_module, "_verify_complete_historical_prefix", counted_verify
    )
    forks = fork_confirmation_arms(
        later_frame,
        prefix.checkpoint,
        historical_prefix=prefix_frame,
    )

    assert verification_calls == ["verified"]
    assert tuple(forks.arms) == CONFIRMATION_ARM_ORDER
    assert forks.checkpoint_sha256 == prefix.checkpoint.digest_sha256
    online = forks.arms[ONLINE_FULL_ARM]
    frozen = forks.arms[FROZEN_2018_ARM]
    global_only = forks.arms[GLOBAL_ONLY_ARM]
    lifetime = forks.arms[LIFETIME_ONLY_ARM]
    assert len(prefix.checkpoint.runtime_segments) == 1
    assert online.checkpoint.runtime_segments == prefix.checkpoint.runtime_segments
    for arm in (frozen, global_only, lifetime):
        assert len(arm.checkpoint.runtime_segments) == 2
        transition = arm.checkpoint.runtime_segments[-1]
        assert transition["first_session_date"] == later_index[0].date().isoformat()
        assert transition["source_offset"] == len(prefix_frame)
        assert transition["prior_segment_terminal_date"] == prefix.checkpoint.checkpoint_date
        assert transition["prior_segment_terminal_count"] == len(prefix_frame)
        assert transition["parent_checkpoint_digest"] == prefix.checkpoint.digest_sha256
    assert all(
        arm.opportunity_frame.equals(online.opportunity_frame)
        for arm in forks.arms.values()
    )
    assert all(arm.forecast.index.equals(later_index) for arm in forks.arms.values())
    assert online.checkpoint.model_payload["runtime"] == {
        "learning_mode": CAUSAL_ONLINE_MODE,
        "frozen_cutoff": None,
        "ablation_mode": FULL_MODE,
    }
    assert frozen.checkpoint.model_payload["runtime"] == {
        "learning_mode": FROZEN_CUTOFF_MODE,
        "frozen_cutoff": "2018-12-31",
        "ablation_mode": FULL_MODE,
    }
    assert global_only.checkpoint.model_payload["runtime"]["ablation_mode"] == GLOBAL_ONLY_MODE
    assert lifetime.checkpoint.model_payload["runtime"]["ablation_mode"] == LIFETIME_ONLY_MODE
    assert online.checkpoint.model_payload["state"] == global_only.checkpoint.model_payload["state"]
    assert online.checkpoint.model_payload["state"] == lifetime.checkpoint.model_payload["state"]
    assert online.forecast.iloc[1]["matured_admitted"]
    assert not frozen.forecast.iloc[1]["matured_admitted"]
    assert (
        online.checkpoint.model_payload["state"]["admitted_lesson_count"]
        > frozen.checkpoint.model_payload["state"]["admitted_lesson_count"]
    )
    assert all(
        global_only.forecast[f"rho__{scale}"].dropna().eq(0.0).all()
        for scale in SCALE_NAMES
    )
    assert lifetime.forecast["scale_active__lifetime"].all()
    assert not lifetime.forecast["scale_active__half_life_8"].any()
    assert not lifetime.forecast["scale_active__half_life_32"].any()

    full_online = replay_from_empty(full_frame)
    verify_resume_equivalence(full_online, online).require()
    regenerated_frozen = replay_module._verify_complete_historical_prefix(
        full_frame, frozen.checkpoint
    ).regenerated
    assert regenerated_frozen.checkpoint.to_dict() == frozen.checkpoint.to_dict()

    next_index = pd.bdate_range("2019-01-11", periods=3, name="date")
    next_frame = _market_frame(next_index)
    complete_frame = pd.concat([full_frame, next_frame])
    for arm in forks.arms.values():
        resumed_again = continue_from_checkpoint(
            next_frame,
            arm.checkpoint,
            historical_prefix=full_frame,
        )
        assert resumed_again.checkpoint.runtime_segments == arm.checkpoint.runtime_segments
        regenerated = replay_module._verify_complete_historical_prefix(
            complete_frame, resumed_again.checkpoint
        ).regenerated
        assert regenerated.checkpoint.to_dict() == resumed_again.checkpoint.to_dict()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("extra", "missing or unexpected"),
        ("pending", "pending state|digest"),
        ("cooldown", "cooldown|digest"),
        ("model_count", "model/session counts|model-state digest|inconsistent"),
        ("schema_bool", "unsupported replay checkpoint schema_version"),
    ],
)
def test_checkpoint_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
):
    index = pd.bdate_range("2001-01-01", periods=6, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[4],
        weak_positions=[],
        index=index,
    )
    checkpoint = replay_from_empty(_market_frame(index)).checkpoint.to_dict()
    if mutation == "extra":
        checkpoint["unexpected"] = 1
    elif mutation == "pending":
        checkpoint["pending_lessons"] = []
    elif mutation == "cooldown":
        observed = checkpoint["signal_cooldown_tail"][-1][
            "contextual_virtual_signal"
        ]
        checkpoint["signal_cooldown_tail"][-1][
            "contextual_virtual_signal"
        ] = not observed
    elif mutation == "schema_bool":
        checkpoint["schema_version"] = True
    else:
        checkpoint["model_payload"]["state"]["processed_session_count"] += 1
    with pytest.raises(ValueError, match=match):
        ReplayCheckpoint.from_dict(checkpoint)


def test_market_and_continuation_inputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=5, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[],
        weak_positions=[],
        index=index,
    )
    frame = _market_frame(index)
    with pytest.raises(ValueError, match="strictly chronological"):
        canonical_market_frame(frame.iloc[::-1])
    duplicate = pd.concat([frame.iloc[:2], frame.iloc[1:]])
    with pytest.raises(ValueError, match="duplicate"):
        canonical_market_frame(duplicate)
    with pytest.raises(ValueError, match="exactly the ordered"):
        canonical_market_frame(frame.drop(columns="qqq_adj_close"))
    nonfinite = frame.copy()
    nonfinite.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        canonical_market_frame(nonfinite)
    inconsistent = frame.copy()
    inconsistent.insert(3, "aapl_adj_open", 99.0)
    with pytest.raises(ValueError, match="bit-exactly"):
        canonical_market_frame(inconsistent)

    prefix = replay_from_empty(frame.iloc[:3])
    with pytest.raises(ValueError, match="later sessions only"):
        continue_from_checkpoint(
            frame.iloc[2:],
            prefix.checkpoint,
            historical_prefix=frame.iloc[:3],
        )


def test_continuation_requires_exact_complete_historical_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=12, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[1, 3, 8, 10],
        weak_positions=[5],
        index=index,
    )
    frame = _market_frame(index)
    prefix_frame = frame.iloc[:8].copy()
    later = frame.iloc[8:].copy()
    checkpoint = replay_from_empty(prefix_frame).checkpoint

    with pytest.raises(ValueError, match="complete historical_prefix"):
        build_fixed_opportunity_frame(later, checkpoint=checkpoint)

    underlying_builder = replay_module.build_fixed_expert_signals
    prefix_replay_calls: list[int] = []

    def counted_builder(value):
        prefix_replay_calls.append(len(value))
        return underlying_builder(value)

    monkeypatch.setattr(
        replay_module, "build_fixed_expert_signals", counted_builder
    )
    modified = prefix_frame.copy()
    modified.iloc[0, modified.columns.get_loc("aapl_open")] += 1.0
    with pytest.raises(ValueError, match="runtime segment|exact checkpoint envelope"):
        continue_from_checkpoint(
            object(),  # type: ignore[arg-type]
            checkpoint,
            historical_prefix=modified,
        )
    assert prefix_replay_calls == [len(modified)]

    with pytest.raises(ValueError, match="runtime segment|exact checkpoint envelope"):
        continue_from_checkpoint(
            later,
            checkpoint,
            historical_prefix=prefix_frame.iloc[1:],
        )

    extra = pd.concat([prefix_frame, later.iloc[:1]])
    with pytest.raises(ValueError, match="exact checkpoint envelope"):
        continue_from_checkpoint(
            later.iloc[1:],
            checkpoint,
            historical_prefix=extra,
        )


@pytest.mark.parametrize("mutation", ["extra", "reordered", "bool", "string", "complex"])
def test_canonical_market_schema_rejects_ambiguous_cells_and_columns(
    mutation: str,
):
    index = pd.bdate_range("2001-01-01", periods=3, name="date")
    frame = _market_frame(index)
    if mutation == "extra":
        frame["extra"] = 1.0
    elif mutation == "reordered":
        frame = frame.loc[:, list(reversed(frame.columns))]
    elif mutation == "bool":
        frame = frame.astype(object)
        frame.iloc[0, 0] = True
    elif mutation == "string":
        frame = frame.astype(object)
        frame.iloc[0, 0] = "100.0"
    else:
        frame = frame.astype(object)
        frame.iloc[0, 0] = 100.0 + 0.0j
    with pytest.raises(ValueError, match="ordered|real non-boolean"):
        canonical_market_frame(frame)

    canonical = canonical_market_frame(_market_frame(index))
    assert tuple(canonical.columns) == replay_module.CANONICAL_MARKET_COLUMNS
    assert canonical_market_frame(canonical).equals(canonical)
    one_ulp = canonical.copy()
    one_ulp.iloc[0, one_ulp.columns.get_loc("aapl_adj_open")] = np.nextafter(
        one_ulp.iloc[0]["aapl_adj_open"], np.inf
    )
    with pytest.raises(ValueError, match="bit-exactly"):
        canonical_market_frame(one_ulp)


def test_empty_learning_and_ablation_controls_do_not_fall_back_to_defaults():
    index = pd.bdate_range("2001-01-01", periods=3, name="date")
    frame = _market_frame(index)
    with pytest.raises(ValueError, match="unsupported learning_mode"):
        replay_from_empty(frame, learning_mode="")
    with pytest.raises(ValueError, match="unsupported ablation_mode"):
        replay_from_empty(frame, ablation_mode="")


@pytest.mark.parametrize(
    "corruption",
    ["model_state", "count", "hash", "tail", "runtime"],
)
def test_self_consistently_rehashed_checkpoint_corruption_fails_prefix_replay_before_suffix_access(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
):
    index = pd.bdate_range("2001-01-01", periods=140, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=list(range(22, 120, 4)),
        weak_positions=list(range(24, 120, 7)),
        index=index,
    )
    frame = _market_frame(index, adjusted_opens=100.0 + 8.0 * np.sin(np.arange(140)))
    payload = replay_from_empty(frame).checkpoint.to_dict()
    if corruption == "model_state":
        payload["model_payload"]["state"]["matured_lesson_count"] += 1
    elif corruption == "count":
        payload["source_session_count"] += 1
        payload["model_payload"]["state"]["processed_session_count"] += 1
    elif corruption == "hash":
        payload["forecast_prefix_sha256"] = "0" * 64
    elif corruption == "tail":
        payload["market_feature_tail"][0]["spy_adj_close"] += 1.0
    else:
        payload["model_payload"]["runtime"]["ablation_mode"] = GLOBAL_ONLY_MODE
        payload["runtime_segments"][-1]["runtime"][
            "ablation_mode"
        ] = GLOBAL_ONLY_MODE
    forged = ReplayCheckpoint.from_dict(_rehash_checkpoint_payload(payload))
    forged.validate()

    # ``object()`` is a poison suffix: touching it would produce a DataFrame
    # type error. Every forged checkpoint must fail from prefix evidence first.
    with pytest.raises(ValueError, match="historical prefix|checkpoint envelope"):
        continue_from_checkpoint(
            object(),  # type: ignore[arg-type]
            forged,
            historical_prefix=frame,
        )


@pytest.mark.parametrize(
    "tamper",
    ["missing", "extra", "reordered", "date", "offset", "runtime", "parent_digest"],
)
def test_runtime_segment_lineage_tampering_fails_before_suffix_access(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
):
    index = pd.bdate_range("2001-01-01", periods=24, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[2, 5, 8, 12, 15, 18, 21],
        weak_positions=[10, 17],
        index=index,
    )
    frame = _market_frame(index, adjusted_opens=100.0 + 6.0 * np.sin(np.arange(24)))
    prefix_frame = frame.iloc[:12]
    prefix = replay_from_empty(prefix_frame)
    transitioned = continue_from_checkpoint(
        frame.iloc[12:],
        prefix.checkpoint,
        historical_prefix=prefix_frame,
        ablation_mode=GLOBAL_ONLY_MODE,
    )
    assert len(transitioned.checkpoint.runtime_segments) == 2
    replay_module._verify_complete_historical_prefix(
        frame, transitioned.checkpoint
    )
    payload = transitioned.checkpoint.to_dict()
    segments = payload["runtime_segments"]
    if tamper == "missing":
        segments.pop()
    elif tamper == "extra":
        segments.append(
            {
                "first_session_date": index[-1].date().isoformat(),
                "source_offset": len(index) - 1,
                "runtime": {
                    "learning_mode": CAUSAL_ONLINE_MODE,
                    "frozen_cutoff": None,
                    "ablation_mode": FULL_MODE,
                },
                "prior_segment_terminal_date": index[-2].date().isoformat(),
                "prior_segment_terminal_count": len(index) - 1,
                "parent_checkpoint_digest": "1" * 64,
            }
        )
        payload["model_payload"]["runtime"]["ablation_mode"] = FULL_MODE
    elif tamper == "reordered":
        segments.reverse()
    elif tamper == "date":
        segments[1]["first_session_date"] = index[13].date().isoformat()
    elif tamper == "offset":
        segments[1]["source_offset"] = 13
        segments[1]["prior_segment_terminal_count"] = 13
        segments[1]["first_session_date"] = index[13].date().isoformat()
        segments[1]["prior_segment_terminal_date"] = index[12].date().isoformat()
    elif tamper == "runtime":
        segments[1]["runtime"]["ablation_mode"] = LIFETIME_ONLY_MODE
        payload["model_payload"]["runtime"][
            "ablation_mode"
        ] = LIFETIME_ONLY_MODE
    else:
        segments[1]["parent_checkpoint_digest"] = "0" * 64
    _rehash_checkpoint_payload(payload)

    with pytest.raises(ValueError):
        forged = ReplayCheckpoint.from_dict(payload)
        continue_from_checkpoint(
            object(),  # type: ignore[arg-type]
            forged,
            historical_prefix=frame,
        )


def test_row_diagnostics_are_complete_and_exact(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=25, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[1, 3, 20, 22],
        weak_positions=[6, 8, 20],
        index=index,
    )
    result = replay_from_empty(_market_frame(index))
    forecast = result.forecast
    assert (forecast["Q"] == forecast["cash_score"]).all()
    assert set(forecast["action"]) <= {"LONG", "CASH"}
    assert set(forecast["learner_target_exposure"]) <= {0.0, 1.0}
    assert (
        forecast.loc[~forecast["canonical_union_opportunity"], "action"]
        == "LONG"
    ).all()
    for expert in EXPERT_NAMES:
        assert f"advice__{expert}" in forecast
        assert f"aggregate_weight__{expert}" in forecast
        assert f"matured_advice__{expert}" in forecast
        assert f"matured_reward__{expert}" in forecast
    for scale in SCALE_NAMES:
        assert f"rho__{scale}" in forecast
        for expert in EXPERT_NAMES:
            assert f"weight__{scale}__{expert}" in forecast
    assert forecast.iloc[-1]["processed_session_count"] == len(index)
    assert result.checkpoint.opportunity_counts["unfiltered_union_signal"] == int(
        result.opportunity_frame["unfiltered_union_signal"].sum()
    )


def test_replay_imports_no_acquisition_news_network_or_llm_modules():
    path = Path(replay_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    project_imports: set[str] = set()
    imported_text: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_text.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_text.append(module)
            if node.level:
                project_imports.add(module)
    assert project_imports == {
        "chronological_exhaustion_expert",
        "contextual_expert_aggregation",
    }
    lowered = "\n".join(imported_text).lower()
    for prohibited in (
        "yfinance",
        "requests",
        "urllib",
        "news",
        "ollama",
        "openai",
        "experiment",
        "runner",
    ):
        assert prohibited not in lowered

    code = """
import sys
import agent_benchmark.contextual_expert_aggregation_replay
forbidden = {'yfinance', 'requests', 'urllib3', 'newsapi', 'ollama', 'openai'}
loaded = sorted(forbidden.intersection({name.split('.')[0] for name in sys.modules}))
allowed_project = {
    'agent_benchmark.contextual_expert_aggregation_replay',
    'agent_benchmark.contextual_expert_aggregation',
    'agent_benchmark.chronological_exhaustion_expert',
}
unexpected_project = sorted(
    name for name in sys.modules
    if name.startswith('agent_benchmark.') and name not in allowed_project
)
assert not loaded, loaded
assert not unexpected_project, unexpected_project
print('clean')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=path.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "clean"


def test_global_and_lifetime_explicit_empty_replays_are_valid(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=4, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[0, 2],
        weak_positions=[],
        index=index,
    )
    global_result = replay_from_empty(
        _market_frame(index),
        learning_mode=CAUSAL_ONLINE_MODE,
        ablation_mode=GLOBAL_ONLY_MODE,
    )
    lifetime_result = replay_from_empty(
        _market_frame(index),
        learning_mode=CAUSAL_ONLINE_MODE,
        ablation_mode=LIFETIME_ONLY_MODE,
    )
    assert global_result.checkpoint.model_payload["runtime"]["ablation_mode"] == GLOBAL_ONLY_MODE
    assert lifetime_result.checkpoint.model_payload["runtime"]["ablation_mode"] == LIFETIME_ONLY_MODE
    assert global_result.checkpoint.model_payload["state"] == lifetime_result.checkpoint.model_payload["state"]


def test_checkpoint_round_trip_is_byte_deterministic(
    monkeypatch: pytest.MonkeyPatch,
):
    index = pd.bdate_range("2001-01-01", periods=10, name="date")
    _patch_signals(
        monkeypatch,
        contextual_positions=[2, 8, 9],
        weak_positions=[5],
        index=index,
    )
    checkpoint = replay_from_empty(_market_frame(index)).checkpoint
    restored = ReplayCheckpoint.from_dict(copy.deepcopy(checkpoint.to_dict()))
    assert restored.to_dict() == checkpoint.to_dict()
    assert restored.digest_sha256 == checkpoint.digest_sha256
