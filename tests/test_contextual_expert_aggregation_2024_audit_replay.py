from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_2024_audit_replay as audit_replay
from agent_benchmark import contextual_expert_aggregation_replay as replay


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    step = np.arange(len(index), dtype=float)
    adjusted_close = 100.0 * np.exp(0.0004 * step + 0.01 * np.sin(step / 11.0))
    return pd.DataFrame(
        {
            "aapl_open": adjusted_close * 1.001,
            "aapl_close": adjusted_close,
            "aapl_adj_close": adjusted_close,
            "spy_adj_close": 200.0 * np.exp(0.00025 * step),
            "qqq_adj_close": 150.0 * np.exp(0.0003 * step),
        },
        index=pd.DatetimeIndex(index, name="date"),
    )


def _prefix_and_suffix() -> tuple[pd.DataFrame, pd.DataFrame, replay.ReplayResult]:
    prefix_index = pd.bdate_range("2022-01-03", "2023-12-29")
    suffix_index = pd.bdate_range("2024-01-02", periods=25)
    combined = _frame(prefix_index.append(suffix_index))
    prefix = combined.loc[prefix_index].copy()
    suffix = combined.loc[suffix_index].copy()
    parent = replay.replay_from_empty(prefix)
    return prefix, suffix, parent


def test_dedicated_fork_creates_only_frozen_lead_and_online_shadow() -> None:
    prefix, suffix, parent = _prefix_and_suffix()
    result = audit_replay.fork_2024_scenarios(
        suffix,
        parent.checkpoint,
        historical_prefix=prefix,
    )

    assert tuple(result.scenarios) == audit_replay.SCENARIO_ORDER
    assert result.parent_checkpoint_sha256 == parent.checkpoint.digest_sha256
    lead = result.scenarios[audit_replay.FROZEN_2023_LEAD]
    shadow = result.scenarios[audit_replay.ONLINE_2024_SHADOW]
    assert lead.opportunity_frame.equals(shadow.opportunity_frame)
    assert lead.diagnostics.admitted_events == 0
    assert lead.checkpoint.model_payload["runtime"] == {
        "learning_mode": replay.FROZEN_CUTOFF_MODE,
        "frozen_cutoff": audit_replay.FROZEN_CUTOFF,
        "ablation_mode": replay.FULL_MODE,
    }
    assert shadow.checkpoint.model_payload["runtime"] == {
        "learning_mode": replay.CAUSAL_ONLINE_MODE,
        "frozen_cutoff": None,
        "ablation_mode": replay.FULL_MODE,
    }


def test_fork_rejects_a_prefix_that_does_not_match_the_parent() -> None:
    prefix, suffix, parent = _prefix_and_suffix()
    changed = prefix.copy()
    changed.iloc[0, changed.columns.get_loc("aapl_close")] *= 1.01

    with pytest.raises(ValueError, match="market hash"):
        audit_replay.fork_2024_scenarios(
            suffix,
            parent.checkpoint,
            historical_prefix=changed,
        )


def test_policy_module_has_no_reporting_dependency_or_expected_result() -> None:
    source = __import__("inspect").getsource(audit_replay)
    assert "audit_artifacts" not in source
    assert "audit_evaluation" not in source
    assert "fork_confirmation_arms(" not in source
    assert "expected_2024" not in source.lower()
