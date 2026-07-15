"""Dedicated two-scenario continuation for the frozen 2024 audit.

This module is intentionally policy-only.  It has no artifact, evaluation, or
reporting imports and contains no expected 2024 action or result.  The caller
must establish the durable attempt lock and bounded-input authorization before
passing a suffix frame here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from . import contextual_expert_aggregation_replay as _replay


FROZEN_2023_LEAD = "frozen_2023_lead"
ONLINE_2024_SHADOW = "online_2024_shadow"
SCENARIO_ORDER = (FROZEN_2023_LEAD, ONLINE_2024_SHADOW)
FROZEN_CUTOFF = "2023-12-29"


@dataclass(frozen=True)
class TwoScenarioFork:
    """The exact parent checkpoint and its two authorized continuations."""

    parent_checkpoint_sha256: str
    scenarios: Mapping[str, _replay.ReplayResult]

    def __post_init__(self) -> None:
        if tuple(self.scenarios) != SCENARIO_ORDER:
            raise ValueError("2024 audit scenarios are missing or reordered")


def _require_exact_parent_prefix(
    historical_prefix: pd.DataFrame,
    checkpoint: _replay.ReplayCheckpoint,
) -> pd.DataFrame:
    """Bind the canonical prefix to the checkpoint before suffix inspection."""

    checkpoint.validate()
    canonical = _replay.canonical_market_frame(historical_prefix)
    if (
        len(canonical) != checkpoint.source_session_count
        or canonical.index[0].date().isoformat() != checkpoint.source_start_date
        or canonical.index[-1].date().isoformat() != checkpoint.checkpoint_date
    ):
        raise ValueError("historical prefix boundary disagrees with checkpoint")
    observed_market_hash = _replay._extend_frame_chain(
        _replay._initial_chain(
            "canonical_market_v1", _replay.CANONICAL_MARKET_COLUMNS
        ),
        canonical,
        columns=_replay.CANONICAL_MARKET_COLUMNS,
    )
    if observed_market_hash != checkpoint.market_prefix_sha256:
        raise ValueError("historical prefix market hash disagrees with checkpoint")
    return canonical


def fork_2024_scenarios(
    later_frame: pd.DataFrame,
    checkpoint: _replay.ReplayCheckpoint | Mapping[str, Any],
    *,
    historical_prefix: pd.DataFrame,
) -> TwoScenarioFork:
    """Continue exactly one frozen lead and one causal online shadow.

    The complete prefix is hashed before ``later_frame`` is canonicalized.  The
    inherited four-arm helper is deliberately not called.
    """

    parsed = (
        checkpoint
        if isinstance(checkpoint, _replay.ReplayCheckpoint)
        else _replay.ReplayCheckpoint.from_dict(checkpoint)
    )
    parsed.validate()
    if parsed.checkpoint_date != FROZEN_CUTOFF:
        raise ValueError("2024 audit must fork the exact through-2023 checkpoint")
    if parsed.model_payload["runtime"] != {
        "learning_mode": _replay.CAUSAL_ONLINE_MODE,
        "frozen_cutoff": None,
        "ablation_mode": _replay.FULL_MODE,
    }:
        raise ValueError("2024 audit parent must be causal-online full mode")

    canonical_prefix = _require_exact_parent_prefix(historical_prefix, parsed)
    # This is the first operation in this module that inspects the suffix.
    canonical_later = _replay.canonical_market_frame(later_frame)
    if canonical_later.index[0] <= pd.Timestamp(parsed.checkpoint_date):
        raise ValueError("2024 continuation overlaps its parent checkpoint")
    features = _replay._build_continuation_opportunity_frame(
        canonical_later,
        parsed,
        canonical_prefix,
    )

    lead = _replay._run_canonical_replay(
        canonical_later,
        features,
        previous=parsed,
        learning_mode=_replay.FROZEN_CUTOFF_MODE,
        frozen_cutoff=FROZEN_CUTOFF,
        ablation_mode=_replay.FULL_MODE,
    )
    shadow = _replay._run_canonical_replay(
        canonical_later,
        features,
        previous=parsed,
        learning_mode=_replay.CAUSAL_ONLINE_MODE,
        frozen_cutoff=None,
        ablation_mode=_replay.FULL_MODE,
    )
    if not lead.opportunity_frame.equals(shadow.opportunity_frame):
        raise ValueError("learning changed action-independent fixed features")
    if lead.diagnostics.admitted_events != 0:
        raise ValueError("frozen 2023 lead admitted a post-cutoff lesson")
    parent_admitted = parsed.model_payload["state"]["admitted_lesson_count"]
    lead_admitted = lead.checkpoint.model_payload["state"][
        "admitted_lesson_count"
    ]
    if lead_admitted != parent_admitted:
        raise ValueError("frozen 2023 lead changed its admitted lesson count")

    return TwoScenarioFork(
        parent_checkpoint_sha256=parsed.digest_sha256,
        scenarios={
            FROZEN_2023_LEAD: lead,
            ONLINE_2024_SHADOW: shadow,
        },
    )


__all__ = [
    "FROZEN_2023_LEAD",
    "ONLINE_2024_SHADOW",
    "SCENARIO_ORDER",
    "FROZEN_CUTOFF",
    "TwoScenarioFork",
    "fork_2024_scenarios",
]
