"""Sealed post-hoc long-run audit for the frozen binary-regime selector.

This runner intentionally accepts one exact *rejected* validation parent.  It
does not refit the selector.  A through-2023 prefix and checkpoint are replayed
before any 2024+ value row may be returned to Python.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import binary_regime_union_selector_experiment as _selector_runner
from .binary_regime_union_selector import (
    CAUSAL_ONLINE_MODE,
    FROZEN_CUTOFF_MODE,
    NEGATIVE_MEAN_THRESHOLD,
    POSITIVE_MEAN_THRESHOLD,
    REGIME_NAMES,
    build_binary_regime_union_selector_forecast,
)
from .deterministic_aapl import EvaluationPeriod
from .unleveraged_aapl import canonical_context_frame


CONTRACT_VERSION = "aapl-binary-regime-longrun-audit-v1"
PARENT_CONTRACT_VERSION = "aapl-binary-regime-union-selector-v1"
EVIDENCE_CLASSIFICATION = "post_hoc_repeated_historical_non_confirmatory_audit"
SENTIMENT_DISCLOSURE = (
    "SPY and QQQ price-based market-regime proxies only; no news, "
    "article text, or LLM sentiment"
)
ADAPTIVE_CONCLUSIONS = {
    "unexercised": (
        "Useful continual learning has not been demonstrated; retain the "
        "online arm only as a shadow challenger."
    ),
    "exercised_insufficient_evidence": (
        "Useful continual learning has not been demonstrated; retain the "
        "online arm only as a shadow challenger."
    ),
    "exercised_positive": (
        "The historical incremental observation is positive but is not "
        "reliable prospective evidence."
    ),
    "exercised_negative": (
        "The exercised online adaptation did not add positive historical "
        "value and is not reliable prospective evidence."
    ),
    "exercised_flat": (
        "The exercised online adaptation did not add positive historical "
        "value and is not reliable prospective evidence."
    ),
}
REQUIRED_INTEGRITY_TRUE_FIELDS = (
    "account_union_derived_from_raw_after_pre_inception_removal",
    "unresolved_tail_masked_after_cooldown",
    "selector_cash_subset_of_union",
    "pure_union_filter_identity",
    "frozen_binary_regime_rule_exact",
    "selector_state_replay_exact",
    "shadow_count_equality",
    "shadow_maturity_and_label_formula_exact",
    "causal_lesson_admission_exact",
    "same_action_stream_all_costs",
    "all_policy_ledgers_unleveraged",
    "continuous_account_episode_identity",
    "reporting_episode_and_veto_edge_identity",
)
PARENT_MANIFEST_SHA256 = (
    "sha256:0354355ac460042f96663d1a45cf5e9a8cf4fe873ddf5cc87afefd9c4b5d81dc"
)
PARENT_MANIFEST_PATH = Path(
    "e/binary_regime_union_selector_v1/"
    "binary-regime-union-selector-validation-v1/stage_manifest.json"
)
PARENT_RUN_ID = "binary-regime-union-selector-validation-v1"
AUDIT_RUN_ID = "binary-regime-longrun-audit-v1"
AUDIT_OUTPUT_PATH = Path("e/binary_regime_longrun_audit_v1")
ATTEMPT_LOCK_FILENAME = "AUDIT_ATTEMPT_LOCK.json"
ATTEMPT_LOCK_SCHEMA_VERSION = "binary-regime-longrun-audit-attempt-v1"
ATTEMPT_LOCK_CONTENT_KEYS = frozenset(
    {
        "schema_version",
        "contract_version",
        "run_id",
        "one_run_no_retry",
        "persistent_after_success_or_failure",
        "created_after_through_2023_checkpoint_replay",
        "created_before_raw_blob_or_2024_value_read",
        "parent_manifest_sha256",
        "git_commit",
        "through_2023_bounded_result_sha256",
        "pre_2024_gate_status_sha256",
    }
)
EXPECTED_VALIDATION_FAILURES = (
    "base_5bps_minimum_two_positive_incremental_years",
    "stress_10bps_minimum_two_positive_incremental_years",
    "stress_10bps_veto_benefit_not_concentrated",
)

ACCOUNT_START = pd.Timestamp("2005-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
AUDIT_START = pd.Timestamp("2024-01-01")
AUDIT_END = pd.Timestamp("2026-07-09")
REQUIRED_VALIDATION_LAST_SESSION = pd.Timestamp("2023-12-29")
FINAL_ROWS = 6_875
FINAL_FIRST_SESSION = pd.Timestamp("1999-03-10")
FINAL_DATE_SEQUENCE_SHA256 = (
    "b88df14b4ec60534ace68645ee19c8a0b7d03d0c2c1829a3ad48f8a0a24c9299"
)
FINAL_BOUNDED_RESULT_SHA256 = (
    "sha256:c01447f975d4a90e49c315f23177f357966363b1ec4790632fa54c0dee250b21"
)
RUN_TIME_LIMIT_SECONDS = 3_600.0
IDENTITY_TOLERANCE = 1e-10
ZERO_TOLERANCE = 1e-12
MATERIAL_EDGE = 0.001
MIN_POSITIVE_PERIODS = 12
MIN_COMPLETE_EPISODES = 150
MIN_ADAPTIVE_EPISODES = 10
MIN_ADAPTIVE_ENTRY_YEARS = 2
COST_SCENARIOS = _selector_runner.COST_SCENARIOS

CONTRACT_PATH = Path("docs/aapl_binary_regime_longrun_audit_v1.md")
IMPLEMENTATION_PATHS = (
    Path("agent_benchmark/binary_regime_longrun_audit.py"),
    Path("agent_benchmark/binary_regime_union_selector.py"),
    Path("agent_benchmark/binary_regime_union_selector_experiment.py"),
    Path("agent_benchmark/chronological_exhaustion_expert.py"),
    Path("agent_benchmark/chronological_exhaustion_experiment.py"),
    Path("agent_benchmark/deterministic_aapl.py"),
    Path("agent_benchmark/unleveraged_aapl.py"),
    Path("docs/aapl_binary_regime_union_selector_v1.md"),
)
PARENT_DEPENDENCY_PATHS = frozenset(
    {
        "agent_benchmark/binary_regime_union_selector.py",
        "agent_benchmark/binary_regime_union_selector_experiment.py",
        "agent_benchmark/chronological_exhaustion_expert.py",
        "agent_benchmark/chronological_exhaustion_experiment.py",
        "agent_benchmark/deterministic_aapl.py",
        "agent_benchmark/unleveraged_aapl.py",
        "docs/aapl_binary_regime_union_selector_v1.md",
    }
)

BinaryRegimeLongrunAuditError = (
    _selector_runner.BinaryRegimeUnionSelectorExperimentError
)
_canonical_json_bytes = _selector_runner._canonical_json_bytes
_pretty_json_bytes = _selector_runner._pretty_json_bytes
_sha256 = _selector_runner._sha256
_frame_csv_bytes = _selector_runner._frame_csv_bytes
_manifest = _selector_runner._manifest
_git_bytes = _selector_runner._git_bytes
_git_text = _selector_runner._git_text
load_bounded_prices = _selector_runner.load_bounded_prices


SEAL_TEMP_PREFIX = ".lr-"
SEAL_TEMP_GLOB = f"{SEAL_TEMP_PREFIX}*.sealing"
SEAL_TEMP_NONCE_CHARS = 12
# Leave room below legacy Windows MAX_PATH for APIs that still append an
# internal suffix.  The audit checks this before consuming its one-time lock.
WINDOWS_SAFE_SEAL_PATH_CHARS = 240


class _Deadline:
    """Audit-local strict runtime limit; exactly 3,600 seconds is a failure."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = float(clock())

    def elapsed(self) -> float:
        return float(self._clock()) - self._started

    def check(self, location: str) -> None:
        if self.elapsed() >= RUN_TIME_LIMIT_SECONDS:
            raise BinaryRegimeLongrunAuditError(
                f"Audit reached the strict {RUN_TIME_LIMIT_SECONDS:.0f}s limit at {location}"
            )


def _gate_report(gates: Mapping[str, bool], **extra: Any) -> dict[str, Any]:
    normalized = {str(name): bool(value) for name, value in gates.items()}
    failures = [name for name, passed in normalized.items() if not passed]
    return {
        "passed": not failures,
        "gates": normalized,
        "failures": failures,
        **extra,
    }


def _annual_periods(last_year: int) -> tuple[EvaluationPeriod, ...]:
    if last_year < 2005 or last_year > 2025:
        raise BinaryRegimeLongrunAuditError("Invalid final full calendar year")
    periods = tuple(
        EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        for year in range(2005, last_year + 1)
    )
    if last_year == 2025:
        periods += (
            EvaluationPeriod("2026_ytd", "2026-01-01", "2026-07-09"),
        )
    return periods


def _seal_artifact_names() -> frozenset[str]:
    """Return every filename that can exist in the promoted audit bundle."""

    return frozenset(_required_audit_payload_names()) | {
        ".gitattributes",
        "report.json",
        "stage_manifest.json",
        "checksums.json",
    }


def _require_seal_path_feasibility(
    *, output_dir: Path, path_limit_chars: int | None = None
) -> None:
    """Fail before the attempt lock if Windows cannot safely seal the bundle."""

    limit = path_limit_chars
    if limit is None:
        if os.name != "nt":
            return
        limit = WINDOWS_SAFE_SEAL_PATH_CHARS
    if limit <= 0:
        raise BinaryRegimeLongrunAuditError("Invalid seal path-length budget")

    destination = output_dir.resolve()
    run_dir = destination / AUDIT_RUN_ID
    representative_temporary = destination / (
        f"{SEAL_TEMP_PREFIX}{'0' * SEAL_TEMP_NONCE_CHARS}.sealing"
    )
    paths = [
        base / name
        for base in (run_dir, representative_temporary)
        for name in _seal_artifact_names()
    ]
    longest = max(paths, key=lambda path: len(str(path)))
    longest_length = len(str(longest))
    if longest_length >= limit:
        raise BinaryRegimeLongrunAuditError(
            "Frozen audit output path is too deep for safe atomic sealing "
            f"before the one-time attempt lock ({longest_length} >= {limit} chars): "
            f"{longest}"
        )


def _require_frozen_run_destination(
    *, repo_root: Path, output_dir: Path, run_id: str | None
) -> str:
    """Freeze the sole audit identity and reject retries before data access."""

    root = repo_root.resolve()
    expected_output = (root / AUDIT_OUTPUT_PATH).resolve()
    actual_output = output_dir.resolve()
    resolved_run_id = AUDIT_RUN_ID if run_id is None else run_id
    if actual_output != expected_output or resolved_run_id != AUDIT_RUN_ID:
        raise BinaryRegimeLongrunAuditError(
            "Audit output path and run id are frozen; alternate or retry runs are forbidden"
        )
    run_dir = expected_output / AUDIT_RUN_ID
    attempt_lock = expected_output / ATTEMPT_LOCK_FILENAME
    stale_seals = list(expected_output.glob(f".{AUDIT_RUN_ID}.*.sealing"))
    stale_seals.extend(expected_output.glob(SEAL_TEMP_GLOB))
    if run_dir.exists() or attempt_lock.exists() or stale_seals:
        raise BinaryRegimeLongrunAuditError(
            "The one frozen audit run already exists, was already attempted, or has a stale sealing attempt"
        )
    _require_seal_path_feasibility(output_dir=expected_output)
    return resolved_run_id


def _create_persistent_attempt_lock(
    *,
    repo_root: Path,
    output_dir: Path,
    parent: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    prefix_provenance: Mapping[str, Any],
    pre_2024_gate_status: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Durably consume the sole audit attempt before any 2024+ value read."""

    root = repo_root.resolve()
    destination = output_dir.resolve()
    if destination != (root / AUDIT_OUTPUT_PATH).resolve():
        raise BinaryRegimeLongrunAuditError(
            "Persistent attempt lock destination is not the frozen audit root"
        )
    payload = {
        "schema_version": ATTEMPT_LOCK_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "run_id": AUDIT_RUN_ID,
        "one_run_no_retry": True,
        "persistent_after_success_or_failure": True,
        "created_after_through_2023_checkpoint_replay": True,
        "created_before_raw_blob_or_2024_value_read": True,
        "parent_manifest_sha256": parent.get("manifest_sha256"),
        "git_commit": git_identity.get("commit"),
        "through_2023_bounded_result_sha256": prefix_provenance.get(
            "bounded_result_sha256"
        ),
        "pre_2024_gate_status_sha256": _sha256(
            _canonical_json_bytes(pre_2024_gate_status)
        ),
    }
    if (
        payload["parent_manifest_sha256"] != PARENT_MANIFEST_SHA256
        or re.fullmatch(r"[0-9a-f]{40}", str(payload["git_commit"])) is None
        or payload["through_2023_bounded_result_sha256"]
        != parent.get("bounded_result_sha256")
    ):
        raise BinaryRegimeLongrunAuditError(
            "Persistent attempt lock inputs are not the authorized frozen identities"
        )
    lock_bytes = _pretty_json_bytes(payload)
    lock_path = destination / ATTEMPT_LOCK_FILENAME
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with lock_path.open("xb") as handle:
            handle.write(lock_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = lock_path.read_bytes()
    except FileExistsError as exc:
        raise BinaryRegimeLongrunAuditError(
            "The sole long-run audit attempt has already been consumed"
        ) from exc
    except OSError as exc:
        # Never remove a partially created lock: its existence must block any
        # later retry after an ambiguous durability failure.
        raise BinaryRegimeLongrunAuditError(
            "Persistent audit-attempt lock could not be durably created"
        ) from exc
    if persisted != lock_bytes:
        raise BinaryRegimeLongrunAuditError(
            "Persistent audit-attempt lock readback changed"
        )
    try:
        relative = lock_path.relative_to(root).as_posix()
    except ValueError as exc:  # pragma: no cover - destination was checked above
        raise BinaryRegimeLongrunAuditError(
            "Persistent audit-attempt lock escaped the repository"
        ) from exc
    identity = {
        "path": relative,
        "sha256": _sha256(lock_bytes),
        "content": payload,
        "left_in_place_for_commit": True,
    }
    return identity, lock_bytes


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _always_long_matches(metrics: Mapping[str, Any]) -> bool:
    for cost_name, _ in COST_SCENARIOS:
        cost_metrics = metrics.get(cost_name, {})
        result = cost_metrics.get("always_long", {})
        values = (
            _finite_number(result.get("total_active_log_edge")),
            _finite_number(result.get("continuous_account_active_log_edge")),
            _finite_number(
                result.get("comparison", {}).get(
                    "relative_wealth_vs_aapl_buy_hold"
                )
            ),
        )
        if any(value is None for value in values):
            return False
        selector_periods = cost_metrics.get("selector", {}).get("periods")
        always_periods = result.get("periods")
        if (
            not isinstance(selector_periods, dict)
            or not selector_periods
            or not isinstance(always_periods, dict)
            or set(always_periods) != set(selector_periods)
        ):
            return False
        for period in always_periods.values():
            if not isinstance(period, dict):
                return False
            active = _finite_number(period.get("active_log_edge"))
            strategy_return = _finite_number(period.get("strategy_return"))
            aapl_return = _finite_number(period.get("aapl_buy_hold_return"))
            boundary = _finite_number(
                period.get("ledger_boundary_active_log_edge")
            )
            if (
                active is None
                or strategy_return is None
                or aapl_return is None
                or boundary is None
                or abs(active) > ZERO_TOLERANCE
                or abs(strategy_return - aapl_return) > ZERO_TOLERANCE
                or abs(boundary) > ZERO_TOLERANCE
            ):
                return False
    try:
        return bool(_selector_runner._always_long_gate(metrics))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _all_policy_no_leverage_proofs(metrics: Mapping[str, Any]) -> bool:
    required = {
        "passed",
        "maximum_requested_target",
        "maximum_post_fill_exposure",
        "maximum_holding_exposure",
        "minimum_cash",
        "minimum_shares",
        "total_margin_interest",
        "shorting",
        "borrowing",
    }
    try:
        for cost_name, _ in COST_SCENARIOS:
            for policy in ("selector", "union", "always_long"):
                proof = metrics[cost_name][policy]["no_leverage_proof"]
                if not isinstance(proof, dict) or set(proof) != required:
                    return False
                maxima = tuple(
                    _finite_number(proof[name])
                    for name in (
                        "maximum_requested_target",
                        "maximum_post_fill_exposure",
                        "maximum_holding_exposure",
                    )
                )
                minimum_cash = _finite_number(proof["minimum_cash"])
                minimum_shares = _finite_number(proof["minimum_shares"])
                margin = _finite_number(proof["total_margin_interest"])
                if (
                    proof["passed"] is not True
                    or proof["shorting"] is not False
                    or proof["borrowing"] is not False
                    or any(value is None for value in maxima)
                    or any(
                        value < -ZERO_TOLERANCE
                        or value > 1.0 + ZERO_TOLERANCE
                        for value in maxima
                        if value is not None
                    )
                    or minimum_cash is None
                    or minimum_cash < -ZERO_TOLERANCE
                    or minimum_shares is None
                    or minimum_shares < -ZERO_TOLERANCE
                    or margin is None
                    or abs(margin) > ZERO_TOLERANCE
                ):
                    return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def apply_strict_recent_history_gates(
    metrics: Mapping[str, Any], integrity: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the six frozen recent-period tests plus fail-closed integrity."""

    gates: dict[str, bool] = {
        **_selector_runner._integrity_gates(integrity),
        "always_long_matches_same_ledger_aapl": _always_long_matches(metrics),
        "all_policy_no_leverage_proofs": _all_policy_no_leverage_proofs(metrics),
    }
    inputs: dict[str, Any] = {}
    for cost_name, _ in COST_SCENARIOS:
        selector = metrics.get(cost_name, {}).get("selector", {})
        for period_name in ("2024", "2025", "2026_ytd"):
            value = _finite_number(
                selector.get("periods", {})
                .get(period_name, {})
                .get("active_log_edge")
            )
            key = f"{cost_name}_{period_name}_active_log_edge_above_001"
            gates[key] = value is not None and value > MATERIAL_EDGE
            inputs[key] = value
    result = _gate_report(gates, gate_inputs=inputs)
    result["strict_recent_history_pass"] = result["passed"]
    result["evidence_classification"] = (
        "repeated_historical_target_non_confirmatory"
    )
    result["ytd_label"] = "2026 YTD through 2026-07-09"
    return result


def apply_post_hoc_long_run_gates(
    metrics: Mapping[str, Any],
    episodes: Mapping[str, Mapping[str, pd.DataFrame]],
    integrity: Mapping[str, Any],
    *,
    expected_period_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply the exact descriptive long-run criterion at both costs."""

    gates: dict[str, bool] = {
        **_selector_runner._integrity_gates(integrity),
        "always_long_matches_same_ledger_aapl": _always_long_matches(metrics),
        "all_policy_no_leverage_proofs": _all_policy_no_leverage_proofs(metrics),
    }
    required_period_names = tuple(
        expected_period_names
        or (*tuple(str(year) for year in range(2005, 2026)), "2026_ytd")
    )
    diagnostics: dict[str, Any] = {}
    for cost_name, _ in COST_SCENARIOS:
        cost_metrics = metrics.get(cost_name, {})
        selector = cost_metrics.get("selector", {})
        union = cost_metrics.get("union", {})
        selector_vs_union = cost_metrics.get("selector_vs_union", {})
        period_items = selector.get("periods", {})
        names = list(required_period_names)
        exact_period_inventory = set(period_items) == set(names)
        period_active_values = [
            _finite_number(period_items.get(name, {}).get("active_log_edge"))
            for name in names
        ]
        period_aapl_values = [
            _finite_number(period_items.get(name, {}).get("aapl_buy_hold_return"))
            for name in names
        ]
        periods_finite = exact_period_inventory and all(
            value is not None for value in (*period_active_values, *period_aapl_values)
        )
        annual_edges = np.asarray(
            [float(value) for value in period_active_values if value is not None],
            dtype=float,
        )
        negative_aapl_names = [
            name
            for name, value in zip(names, period_aapl_values, strict=True)
            if value is not None and value < -ZERO_TOLERANCE
        ]
        negative_aapl_edge = (
            float(
                np.sum(
                    [
                        float(period_items[name]["active_log_edge"])
                        for name in negative_aapl_names
                    ]
                )
            )
            if periods_finite
            else None
        )
        total_edge = _finite_number(selector.get("continuous_account_active_log_edge"))
        union_incremental = _finite_number(
            selector_vs_union.get("continuous_account_total_active_log_edge")
        )
        episode_frame = episodes.get(cost_name, {}).get("selector")
        if not isinstance(episode_frame, pd.DataFrame) or "net_active_log_edge" not in episode_frame:
            episode_edges = np.asarray([], dtype=float)
        else:
            episode_edges = pd.to_numeric(
                episode_frame["net_active_log_edge"], errors="coerce"
            ).to_numpy(dtype=float)
        finite_episodes = bool(
            len(episode_edges) and np.isfinite(episode_edges).all()
        )
        positive_episode_edges = episode_edges[episode_edges > ZERO_TOLERANCE]
        win_rate = (
            float(np.mean(episode_edges > ZERO_TOLERANCE))
            if finite_episodes
            else None
        )
        mean_edge = float(np.mean(episode_edges)) if finite_episodes else None
        median_edge = float(np.median(episode_edges)) if finite_episodes else None
        five_largest = (
            np.sort(episode_edges)[-5:] if len(episode_edges) >= 5 else episode_edges
        )
        after_five = (
            float(total_edge - np.sum(five_largest))
            if total_edge is not None and finite_episodes
            else None
        )
        maximum_positive_share = (
            float(np.max(positive_episode_edges) / np.sum(positive_episode_edges))
            if len(positive_episode_edges)
            else None
        )
        positive_period_count = int(
            np.count_nonzero(annual_edges > ZERO_TOLERANCE)
        )
        after_best_period = (
            float(total_edge - np.max(annual_edges))
            if total_edge is not None and len(annual_edges)
            else None
        )
        comparison = selector.get("comparison", {})
        strategy_drawdown = _finite_number(
            comparison.get("strategy", {}).get("max_drawdown")
        )
        aapl_drawdown = _finite_number(
            comparison.get("aapl_buy_hold", {}).get("max_drawdown")
        )
        recorded_count = _finite_number(selector.get("cash_episode_count"))
        count_matches = bool(
            finite_episodes
            and recorded_count is not None
            and recorded_count.is_integer()
            and int(recorded_count) == len(episode_edges)
        )
        prefix = f"{cost_name}_"
        gates.update(
            {
                prefix + "total_active_log_edge_above_001": (
                    total_edge is not None and total_edge > MATERIAL_EDGE
                ),
                prefix + "minimum_12_strictly_positive_periods": (
                    periods_finite
                    and positive_period_count >= MIN_POSITIVE_PERIODS
                ),
                prefix + "positive_after_subtracting_largest_period": (
                    after_best_period is not None
                    and after_best_period > ZERO_TOLERANCE
                ),
                prefix + "positive_negative_aapl_period_aggregate": (
                    periods_finite
                    and bool(negative_aapl_names)
                    and negative_aapl_edge is not None
                    and negative_aapl_edge > ZERO_TOLERANCE
                ),
                prefix + "minimum_150_complete_cash_episodes": (
                    finite_episodes
                    and count_matches
                    and len(episode_edges) >= MIN_COMPLETE_EPISODES
                ),
                prefix + "beneficial_episode_rate_at_least_50pct": (
                    win_rate is not None and win_rate >= 0.5
                ),
                prefix + "positive_mean_episode_edge": (
                    mean_edge is not None and mean_edge > ZERO_TOLERANCE
                ),
                prefix + "positive_median_episode_edge": (
                    median_edge is not None and median_edge > ZERO_TOLERANCE
                ),
                prefix + "positive_after_subtracting_five_largest_episodes": (
                    after_five is not None and after_five > ZERO_TOLERANCE
                ),
                prefix + "no_episode_above_25pct_positive_edge": (
                    maximum_positive_share is not None
                    and maximum_positive_share <= 0.25 + ZERO_TOLERANCE
                ),
                prefix + "max_drawdown_no_worse_than_aapl": (
                    strategy_drawdown is not None
                    and aapl_drawdown is not None
                    and strategy_drawdown >= aapl_drawdown - ZERO_TOLERANCE
                ),
                prefix + "selector_minus_fixed_union_strictly_positive": (
                    union_incremental is not None
                    and union_incremental > ZERO_TOLERANCE
                ),
            }
        )
        diagnostics[cost_name] = {
            "reporting_period_count": len(names),
            "exact_reporting_period_inventory": exact_period_inventory,
            "all_period_gate_inputs_finite": periods_finite,
            "positive_period_count": positive_period_count,
            "total_active_log_edge": total_edge,
            "largest_period_active_log_edge": (
                float(np.max(annual_edges)) if len(annual_edges) else None
            ),
            "active_log_edge_after_largest_period": after_best_period,
            "negative_aapl_periods": negative_aapl_names,
            "negative_aapl_period_aggregate_active_log_edge": negative_aapl_edge,
            "complete_cash_episode_count": int(len(episode_edges)),
            "beneficial_episode_rate": win_rate,
            "mean_episode_edge": mean_edge,
            "median_episode_edge": median_edge,
            "active_log_edge_after_five_largest_episodes": after_five,
            "maximum_positive_episode_share": maximum_positive_share,
            "strategy_max_drawdown": strategy_drawdown,
            "aapl_max_drawdown": aapl_drawdown,
            "selector_minus_fixed_union_active_log_edge": union_incremental,
            "union_continuous_active_log_edge": _finite_number(
                union.get("continuous_account_active_log_edge")
            ),
        }
    result = _gate_report(gates, gate_inputs=diagnostics)
    result["post_hoc_long_run_robustness_pass"] = result["passed"]
    result["post_hoc_non_confirmatory"] = True
    return result


def _tracked_index_object_id(root: Path, relative: str) -> str:
    literal_pathspec = f":(top,literal){relative}"
    value = _git_text(root, "ls-files", "--stage", "--", literal_pathspec)
    match = re.fullmatch(r"[0-7]{6} ([0-9a-f]{40}) 0\t(.+)", value)
    if match is None or match.group(2) != relative:
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact must have one ordinary stage-zero index entry"
        )
    return match.group(1)


def _clean_git_identity(repo_root: Path, price_artifact: Path) -> dict[str, Any]:
    """Verify pushed code state without allowing Git to inspect final input."""

    root = repo_root.resolve()
    try:
        price_relative = price_artifact.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact must be inside the Git repository"
        ) from exc
    if any(character in price_relative for character in "\r\n\t"):
        raise BinaryRegimeLongrunAuditError("Final price artifact path is unsafe")
    excluded_price_pathspec = f":(top,exclude,literal){price_relative}"
    try:
        actual = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
        status = _git_text(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            excluded_price_pathspec,
        )
        branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
        commit = _git_text(root, "rev-parse", "HEAD")
        upstream = _git_text(
            root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        upstream_commit = _git_text(root, "rev-parse", "@{upstream}")
        price_head_object_id = _git_text(
            root, "rev-parse", f"HEAD:{price_relative}"
        )
        price_index_object_id = _tracked_index_object_id(root, price_relative)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise BinaryRegimeLongrunAuditError(
            "Audit requires a valid Git repository with a pushed upstream"
        ) from exc
    if actual != root:
        raise BinaryRegimeLongrunAuditError("repo_root must be the actual Git root")
    if status:
        raise BinaryRegimeLongrunAuditError(
            "Audit requires a completely clean worktree"
        )
    if not branch or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise BinaryRegimeLongrunAuditError(
            "Audit requires an attached valid Git commit"
        )
    if (
        not upstream
        or re.fullmatch(r"[0-9a-f]{40}", upstream_commit) is None
        or upstream_commit != commit
    ):
        raise BinaryRegimeLongrunAuditError(
            "Audit requires HEAD to equal its pushed upstream commit"
        )
    if (
        re.fullmatch(r"[0-9a-f]{40}", price_head_object_id) is None
        or price_index_object_id != price_head_object_id
    ):
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact HEAD and index object identities differ before the attempt lock"
        )
    hashes: dict[str, str] = {}
    for relative in (*IMPLEMENTATION_PATHS, CONTRACT_PATH):
        name = relative.as_posix()
        try:
            _git_bytes(root, "ls-files", "--error-unmatch", "--", name)
            committed = _git_bytes(root, "show", f"HEAD:{name}")
        except subprocess.CalledProcessError as exc:
            raise BinaryRegimeLongrunAuditError(
                f"Frozen audit dependency is not tracked: {name}"
            ) from exc
        if not (root / relative).is_file():
            raise BinaryRegimeLongrunAuditError(
                f"Frozen audit dependency is missing locally: {name}"
            )
        hashes[name] = _sha256(committed)
    return {
        "branch": branch,
        "commit": commit,
        "upstream": upstream,
        "upstream_commit": upstream_commit,
        "head_equals_upstream": True,
        "dirty": False,
        "pre_lock_worktree_clean_excluding_final_input": True,
        "final_input_path": price_relative,
        "pre_lock_final_input_head_index_equal": True,
        "final_input_head_object_id": price_head_object_id,
        "final_input_index_object_id": price_index_object_id,
        "final_input_verified_clean_after_attempt_lock": False,
        "tracked_dependency_sha256": hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "duckdb": __import__("duckdb").__version__,
        },
    }


def _tracked_input_identity(repo_root: Path, path: Path) -> dict[str, Any]:
    """After the attempt lock, bind HEAD, index, and local final-input bytes."""

    root = repo_root.resolve()
    source = path.resolve()
    try:
        relative = source.relative_to(root).as_posix()
        literal_pathspec = f":(top,literal){relative}"
        head_object_id = _git_text(root, "rev-parse", f"HEAD:{relative}")
        index_object_id = _tracked_index_object_id(root, relative)
        committed = _git_bytes(root, "show", f"HEAD:{relative}")
        indexed = _git_bytes(root, "show", f":{relative}")
        path_status = _git_text(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            literal_pathspec,
        )
        local = source.read_bytes()
    except (OSError, ValueError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact must be an exact clean tracked file at HEAD and in the index"
        ) from exc
    if (
        path_status
        or head_object_id != index_object_id
        or committed != indexed
        or indexed != local
    ):
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact HEAD, index, local bytes, or path status differ"
        )
    digest = _sha256(local)
    return {
        "path": relative,
        "sha256": digest,
        "head_blob_sha256": _sha256(committed),
        "index_blob_sha256": _sha256(indexed),
        "local_file_sha256": digest,
        "head_object_id": head_object_id,
        "index_object_id": index_object_id,
        "head_index_local_equal": True,
        "path_status_clean": True,
        "verified_after_attempt_lock": True,
    }


def _committed_local_bytes(
    repo_root: Path, path: Path, *, description: str
) -> bytes:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
        _git_bytes(root, "ls-files", "--error-unmatch", "--", relative)
        committed = _git_bytes(root, "show", f"HEAD:{relative}")
        local = resolved.read_bytes()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise BinaryRegimeLongrunAuditError(
            f"{description} must be an exact tracked file at HEAD"
        ) from exc
    if local != committed:
        raise BinaryRegimeLongrunAuditError(
            f"{description} differs from its committed blob"
        )
    return local


def _object_from_bytes(value: bytes, *, description: str) -> dict[str, Any]:
    try:
        result = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryRegimeLongrunAuditError(f"{description} is unreadable") from exc
    if not isinstance(result, dict):
        raise BinaryRegimeLongrunAuditError(
            f"{description} must be a JSON object"
        )
    return result


def _require_valid_self_hash(
    manifest: Mapping[str, Any], *, exact_hash: str | None = None
) -> None:
    payload = dict(manifest)
    recorded = payload.pop("manifest_sha256", None)
    calculated = _sha256(_canonical_json_bytes(payload))
    if recorded != calculated or (exact_hash is not None and recorded != exact_hash):
        raise BinaryRegimeLongrunAuditError("Parent manifest self-hash is invalid")


def _validated_rejected_validation_parent(
    *, repo_root: Path, path: Path
) -> dict[str, Any]:
    """Verify the one exact rejected validation bundle and its passing parent."""

    root = repo_root.resolve()
    manifest_path = path.resolve()
    try:
        relative = manifest_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise BinaryRegimeLongrunAuditError(
            "Rejected validation parent must be inside the repository"
        ) from exc
    if relative != PARENT_MANIFEST_PATH.as_posix():
        raise BinaryRegimeLongrunAuditError(
            "Audit accepts only the frozen rejected validation parent path"
        )
    manifest_bytes = _committed_local_bytes(
        root, manifest_path, description="Rejected validation manifest"
    )
    manifest = _object_from_bytes(
        manifest_bytes, description="Rejected validation manifest"
    )
    _require_valid_self_hash(manifest, exact_hash=PARENT_MANIFEST_SHA256)
    if (
        manifest.get("contract_version") != PARENT_CONTRACT_VERSION
        or manifest.get("stage") != "validation"
        or manifest.get("stage_pass") is not False
        or manifest.get("run_id") != PARENT_RUN_ID
    ):
        raise BinaryRegimeLongrunAuditError(
            "Validation parent is not the exact frozen rejected stage"
        )

    hashes = manifest.get("payload_sha256")
    required = _selector_runner._required_parent_payloads("validation")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(required)
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None
            for name, expected in hashes.items()
        )
    ):
        raise BinaryRegimeLongrunAuditError(
            "Rejected validation payload inventory is not exact"
        )
    verified: dict[str, bytes] = {}
    for filename, expected in hashes.items():
        value = _committed_local_bytes(
            root,
            manifest_path.parent / filename,
            description=f"Rejected validation payload {filename}",
        )
        if _sha256(value) != expected:
            raise BinaryRegimeLongrunAuditError(
                f"Rejected validation payload checksum changed: {filename}"
            )
        verified[filename] = value

    gate = _object_from_bytes(
        verified["validation_gate_report.json"],
        description="Rejected validation gate report",
    )
    report = _object_from_bytes(
        verified["report.json"], description="Rejected validation report"
    )
    gate_values = gate.get("gates")
    if not isinstance(gate_values, dict) or any(
        not isinstance(value, bool) for value in gate_values.values()
    ):
        raise BinaryRegimeLongrunAuditError(
            "Rejected validation gate values are malformed"
        )
    false_gates = tuple(sorted(name for name, value in gate_values.items() if not value))
    expected_failures = tuple(sorted(EXPECTED_VALIDATION_FAILURES))
    if (
        gate.get("passed") is not False
        or tuple(sorted(gate.get("failures", ()))) != expected_failures
        or false_gates != expected_failures
        or report.get("contract_version") != PARENT_CONTRACT_VERSION
        or report.get("stage") != "validation"
        or report.get("run_id") != PARENT_RUN_ID
        or report.get("gate_report") != gate
        or report.get("post_2023_outcomes_accessed") is not False
    ):
        raise BinaryRegimeLongrunAuditError(
            "Rejected manifest, report, and exact three gate failures disagree"
        )

    provenance = manifest.get("source_provenance", {})
    if (
        provenance.get("bounded_last_date") != "2023-12-29"
        or provenance.get("physical_snapshot_has_later_rows") is not False
        or provenance.get("rows_after_bound_returned") is not False
        or pd.Timestamp(provenance.get("bounded_last_date")) > VALIDATION_END
    ):
        raise BinaryRegimeLongrunAuditError(
            "Rejected validation parent contains or returned post-2023 rows"
        )

    embedded_bytes = verified["authorized_development_manifest.json"]
    embedded = _object_from_bytes(
        embedded_bytes, description="Embedded development parent"
    )
    _require_valid_self_hash(embedded)
    if (
        embedded.get("contract_version") != PARENT_CONTRACT_VERSION
        or embedded.get("stage") != "development"
        or embedded.get("stage_pass") is not True
        or embedded.get("manifest_sha256")
        != manifest.get("parent_manifest_sha256")
    ):
        raise BinaryRegimeLongrunAuditError(
            "Embedded development parent is not the passing parent"
        )
    development_path = (
        manifest_path.parent.parent
        / str(embedded.get("run_id"))
        / "stage_manifest.json"
    )
    actual_development_bytes = _committed_local_bytes(
        root, development_path, description="Passing development manifest"
    )
    if actual_development_bytes != embedded_bytes:
        raise BinaryRegimeLongrunAuditError(
            "Embedded development parent differs from its committed manifest"
        )
    # Reuse the passing-parent verifier only for the embedded passing stage.
    development = _selector_runner._validated_prior_manifest(
        repo_root=root, path=development_path, expected_stage="development"
    )
    if _canonical_json_bytes(development) != _canonical_json_bytes(embedded):
        raise BinaryRegimeLongrunAuditError(
            "Passing development parent continuity failed"
        )
    parent_git = manifest.get("git_identity", {})
    development_git = embedded.get("git_identity", {})
    for key in ("tracked_dependency_sha256", "runtime_versions"):
        if parent_git.get(key) != development_git.get(key):
            raise BinaryRegimeLongrunAuditError(
                "Implementation or runtime changed between development and validation"
            )
    return manifest


# Public spelling for focused verifier tests and downstream audit tooling.
validate_rejected_validation_parent = _validated_rejected_validation_parent


def _require_dependency_continuity(
    current_git_identity: Mapping[str, Any], parent: Mapping[str, Any]
) -> None:
    current = current_git_identity.get("tracked_dependency_sha256")
    parent_git = parent.get("git_identity", {})
    parent_hashes = parent_git.get("tracked_dependency_sha256")
    if (
        not isinstance(current, dict)
        or not isinstance(parent_hashes, dict)
        or set(parent_hashes) != set(PARENT_DEPENDENCY_PATHS)
        or any(current.get(name) != expected for name, expected in parent_hashes.items())
        or current_git_identity.get("runtime_versions")
        != parent_git.get("runtime_versions")
    ):
        raise BinaryRegimeLongrunAuditError(
            "Frozen selector dependency or runtime changed after validation"
        )
    required_current = {
        *(path.as_posix() for path in IMPLEMENTATION_PATHS),
        CONTRACT_PATH.as_posix(),
    }
    if set(current) != required_current:
        raise BinaryRegimeLongrunAuditError(
            "Current audit identity did not bind every frozen dependency"
        )


def _require_final_input_identity(
    frame: pd.DataFrame, provenance: Mapping[str, Any]
) -> None:
    """Require the exact frozen six-column through-2026 physical result."""

    required = (
        "aapl_open",
        "aapl_close",
        "aapl_adj_close",
        "spy_adj_close",
        "qqq_adj_close",
    )
    try:
        raw_values = frame.loc[:, required].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact does not have the frozen value schema"
        ) from exc
    if not np.isfinite(raw_values).all() or bool(np.any(raw_values <= 0.0)):
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact contains nonfinite or nonpositive values"
        )
    data = canonical_context_frame(frame)
    coverage = provenance.get("session_coverage", {})
    if (
        len(data) != FINAL_ROWS
        or data.index.min() != FINAL_FIRST_SESSION
        or data.index.max() != AUDIT_END
        or not data.index.is_unique
        or not data.index.is_monotonic_increasing
        or provenance.get("bounded_rows") != FINAL_ROWS
        or provenance.get("bounded_first_date")
        != FINAL_FIRST_SESSION.date().isoformat()
        or provenance.get("bounded_last_date") != AUDIT_END.date().isoformat()
        or provenance.get("bounded_result_sha256")
        != FINAL_BOUNDED_RESULT_SHA256
        or provenance.get("expected_bounded_result_sha256")
        != FINAL_BOUNDED_RESULT_SHA256
        or coverage.get("date_sequence_sha256")
        != FINAL_DATE_SEQUENCE_SHA256
        or provenance.get("physical_snapshot_has_later_rows") is not False
        or provenance.get("rows_after_bound_returned") is not False
    ):
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact does not have the exact frozen identity"
        )
    values = data.loc[:, required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or bool(np.any(values <= 0.0)):
        raise BinaryRegimeLongrunAuditError(
            "Final price artifact contains nonfinite or nonpositive values"
        )


def _require_comparator_identity(
    online_metrics: Mapping[str, Any],
    frozen_metrics: Mapping[str, Any],
    online_integrity: Mapping[str, Any],
    frozen_integrity: Mapping[str, Any],
) -> dict[str, bool]:
    gates = {
        "online_always_long_matches_aapl": _always_long_matches(online_metrics),
        "frozen_always_long_matches_aapl": _always_long_matches(frozen_metrics),
        "online_all_policy_no_leverage_proofs": _all_policy_no_leverage_proofs(
            online_metrics
        ),
        "frozen_2023_all_policy_no_leverage_proofs": _all_policy_no_leverage_proofs(
            frozen_metrics
        ),
    }
    for name, passed in _selector_runner._integrity_gates(
        online_integrity
    ).items():
        gates[f"online_{name}"] = passed
    for name, passed in _selector_runner._integrity_gates(
        frozen_integrity
    ).items():
        gates[f"frozen_2023_{name}"] = passed
    for cost_name, _ in COST_SCENARIOS:
        for policy in ("union", "always_long"):
            gates[f"{cost_name}_{policy}_identical_between_online_and_frozen"] = (
                _canonical_json_bytes(online_metrics[cost_name][policy])
                == _canonical_json_bytes(frozen_metrics[cost_name][policy])
            )
    if not all(gates.values()):
        raise BinaryRegimeLongrunAuditError(
            "Online/frozen comparator identity or ledger integrity failed"
        )
    return gates


def _threshold_crossing_rows(
    online_forecast: pd.DataFrame,
) -> pd.DataFrame:
    value = online_forecast.copy()
    if not isinstance(value.index, pd.DatetimeIndex):
        raise BinaryRegimeLongrunAuditError("Forecast index must be dates")
    lesson_added = _selector_runner._bool_column(value, "shadow_lesson_added_now")
    matured_risk_on = _selector_runner._bool_column(
        value, "shadow_matured_signal_risk_on"
    )
    rows: list[dict[str, Any]] = []
    for regime in REGIME_NAMES:
        cash_name = _selector_runner._state_column(regime, "cash_selected")
        cash = _selector_runner._bool_column(value, cash_name)
        for position in range(1, len(value)):
            date = value.index[position]
            before = bool(cash.iloc[position - 1])
            after = bool(cash.iloc[position])
            if date <= VALIDATION_END or before == after:
                continue
            signal_is_risk_on = bool(matured_risk_on.iloc[position])
            expected_regime = "risk_on" if signal_is_risk_on else "not_risk_on"
            mean = float(value.iloc[position][f"{regime}_mean"])
            ready = bool(value.iloc[position][f"{regime}_ready"])
            if (
                not bool(lesson_added.iloc[position])
                or expected_regime != regime
                or not ready
                or not math.isfinite(mean)
                or (after and not mean > POSITIVE_MEAN_THRESHOLD)
                or ((not after) and not mean < NEGATIVE_MEAN_THRESHOLD)
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Post-2023 regime latch transition is not a causal threshold crossing"
                )
            before_mean = _finite_number(value.iloc[position - 1][f"{regime}_mean"])
            rows.append(
                {
                    "crossing_date": date.date().isoformat(),
                    "regime": regime,
                    "transition": "LONG_TO_CASH" if after else "CASH_TO_LONG",
                    "pre_n_raw": int(value.iloc[position - 1][f"{regime}_n_raw"]),
                    "pre_n_eff": float(value.iloc[position - 1][f"{regime}_n_eff"]),
                    "pre_mean": before_mean,
                    "pre_cash_selected": before,
                    "post_n_raw": int(value.iloc[position][f"{regime}_n_raw"]),
                    "post_n_eff": float(value.iloc[position][f"{regime}_n_eff"]),
                    "post_mean": mean,
                    "post_cash_selected": after,
                    "matured_signal_close": pd.Timestamp(
                        value.iloc[position]["shadow_signal_close"]
                    ).date().isoformat(),
                    "matured_label_10bps": float(
                        value.iloc[position]["shadow_label_10bps"]
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "crossing_date",
            "regime",
            "transition",
            "pre_n_raw",
            "pre_n_eff",
            "pre_mean",
            "pre_cash_selected",
            "post_n_raw",
            "post_n_eff",
            "post_mean",
            "post_cash_selected",
            "matured_signal_close",
            "matured_label_10bps",
        ),
    )


def _divergence_run_rows(mask: pd.Series) -> pd.DataFrame:
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))
    rows: list[dict[str, Any]] = []
    if not len(positions):
        return pd.DataFrame(
            columns=("run_id", "first_decision_date", "last_decision_date", "sessions")
        )
    start = previous = int(positions[0])
    run_id = 1
    for raw_position in positions[1:]:
        position = int(raw_position)
        if position != previous + 1:
            rows.append(
                {
                    "run_id": run_id,
                    "first_decision_date": mask.index[start].date().isoformat(),
                    "last_decision_date": mask.index[previous].date().isoformat(),
                    "sessions": previous - start + 1,
                }
            )
            run_id += 1
            start = position
        previous = position
    rows.append(
        {
            "run_id": run_id,
            "first_decision_date": mask.index[start].date().isoformat(),
            "last_decision_date": mask.index[previous].date().isoformat(),
            "sessions": previous - start + 1,
        }
    )
    return pd.DataFrame(rows)


def _adaptive_value_analysis(
    frame: pd.DataFrame,
    online_forecast: pd.DataFrame,
    frozen_forecast: pd.DataFrame,
    online_metrics: Mapping[str, Any],
    frozen_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Reconcile online-minus-frozen to exact XOR one-session episodes."""

    data = canonical_context_frame(frame)
    online_targets, _ = _selector_runner._stage_targets(
        data, online_forecast, administrative_start=ACCOUNT_START
    )
    frozen_targets, _ = _selector_runner._stage_targets(
        data, frozen_forecast, administrative_start=ACCOUNT_START
    )
    if not online_targets["union"].equals(frozen_targets["union"]):
        raise BinaryRegimeLongrunAuditError(
            "Online and frozen arms do not share the exact fixed union"
        )
    online = online_targets["selector"]
    frozen = frozen_targets["selector"]
    xor = (online == 0.0) ^ (frozen == 0.0)
    if bool(xor.loc[xor.index <= VALIDATION_END].any()):
        raise BinaryRegimeLongrunAuditError(
            "Online and frozen selector actions diverged before 2024"
        )
    positions = np.flatnonzero(xor.to_numpy(dtype=bool))
    if any(position + 2 >= len(data) for position in positions):
        raise BinaryRegimeLongrunAuditError(
            "An online/frozen divergence does not form a complete episode"
        )
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    dates = data.index
    difference_frames: dict[str, pd.DataFrame] = {}
    period_names = tuple(str(year) for year in range(2005, 2026)) + ("2026_ytd",)
    reconciliations: dict[str, Any] = {}
    expected_columns = (
        "decision_date",
        "entry_date",
        "exit_date",
        "entry_period",
        "orientation",
        "online_target",
        "frozen_target",
        "raw_cash_edge",
        "net_cash_edge",
        "online_minus_frozen_active_log_edge",
    )
    for cost_name, cost_bps in COST_SCENARIOS:
        friction = math.log(
            (1.0 - float(cost_bps) / 10_000.0)
            / (1.0 + float(cost_bps) / 10_000.0)
        )
        rows: list[dict[str, Any]] = []
        for raw_position in positions:
            position = int(raw_position)
            online_cash = float(online.iloc[position]) == 0.0
            frozen_cash = float(frozen.iloc[position]) == 0.0
            if online_cash == frozen_cash:
                raise BinaryRegimeLongrunAuditError(
                    "Divergence row does not satisfy exact XOR identity"
                )
            raw_cash_edge = math.log(opens[position + 1] / opens[position + 2])
            net_cash_edge = raw_cash_edge + friction
            signed = net_cash_edge if online_cash else -net_cash_edge
            entry_date = dates[position + 1]
            entry_period = (
                "2026_ytd" if entry_date.year == 2026 else str(entry_date.year)
            )
            rows.append(
                {
                    "decision_date": dates[position].date().isoformat(),
                    "entry_date": entry_date.date().isoformat(),
                    "exit_date": dates[position + 2].date().isoformat(),
                    "entry_period": entry_period,
                    "orientation": (
                        "online_cash_frozen_long"
                        if online_cash
                        else "online_long_frozen_cash"
                    ),
                    "online_target": float(online.iloc[position]),
                    "frozen_target": float(frozen.iloc[position]),
                    "raw_cash_edge": raw_cash_edge,
                    "net_cash_edge": net_cash_edge,
                    "online_minus_frozen_active_log_edge": signed,
                }
            )
        difference = pd.DataFrame(rows, columns=expected_columns)
        difference_frames[cost_name] = difference
        values = (
            difference["online_minus_frozen_active_log_edge"].to_numpy(dtype=float)
            if len(difference)
            else np.asarray([], dtype=float)
        )
        expected_full = float(
            online_metrics[cost_name]["selector"][
                "continuous_account_active_log_edge"
            ]
            - frozen_metrics[cost_name]["selector"][
                "continuous_account_active_log_edge"
            ]
        )
        observed_full = float(np.sum(values))
        full_error = expected_full - observed_full
        if not all(
            math.isfinite(value)
            for value in (expected_full, observed_full, full_error)
        ) or abs(full_error) > IDENTITY_TOLERANCE:
            raise BinaryRegimeLongrunAuditError(
                f"{cost_name} online-minus-frozen full edge is nonfinite or does not reconcile"
            )
        period_errors: dict[str, float] = {}
        period_edges: dict[str, float] = {}
        for period_name in period_names:
            expected_period = float(
                online_metrics[cost_name]["selector"]["periods"][period_name][
                    "active_log_edge"
                ]
                - frozen_metrics[cost_name]["selector"]["periods"][period_name][
                    "active_log_edge"
                ]
            )
            observed_period = float(
                difference.loc[
                    difference["entry_period"] == period_name,
                    "online_minus_frozen_active_log_edge",
                ].sum()
            )
            error = expected_period - observed_period
            if not all(
                math.isfinite(value)
                for value in (expected_period, observed_period, error)
            ) or abs(error) > IDENTITY_TOLERANCE:
                raise BinaryRegimeLongrunAuditError(
                    f"{cost_name} {period_name} adaptive edge is nonfinite or does not reconcile"
                )
            period_edges[period_name] = observed_period
            period_errors[period_name] = error
        reconciliations[cost_name] = {
            "complete_xor_episode_count": int(len(difference)),
            "full_account_online_minus_frozen_active_log_edge": expected_full,
            "xor_episode_incremental_edge": observed_full,
            "full_account_identity_error": full_error,
            "period_incremental_edges": period_edges,
            "period_identity_errors": period_errors,
            "identity_tolerance": IDENTITY_TOLERANCE,
        }

    crossings = _threshold_crossing_rows(online_forecast)
    runs = _divergence_run_rows(xor)
    base_rows = difference_frames[COST_SCENARIOS[0][0]]
    entry_years = sorted(
        {pd.Timestamp(value).year for value in base_rows.get("entry_date", [])}
    )
    complete_count = int(len(base_rows))
    crossing_count = int(len(crossings))
    action_difference_count = int(xor.sum())
    incremental_10bps = float(
        reconciliations["stress_10bps"][
            "xor_episode_incremental_edge"
        ]
    )
    sufficient = (
        complete_count >= MIN_ADAPTIVE_EPISODES
        and len(entry_years) >= MIN_ADAPTIVE_ENTRY_YEARS
    )
    if crossing_count == 0 or action_difference_count == 0:
        status = "unexercised"
    elif not sufficient:
        status = "exercised_insufficient_evidence"
    elif incremental_10bps > ZERO_TOLERANCE:
        status = "exercised_positive"
    elif incremental_10bps < -ZERO_TOLERANCE:
        status = "exercised_negative"
    else:
        status = "exercised_flat"
    useful_demonstrated = status == "exercised_positive"
    conclusion = ADAPTIVE_CONCLUSIONS[status]
    diagnostic = {
        "adaptive_status": status,
        "state_threshold_crossing_count": crossing_count,
        "state_threshold_crossings": crossings.to_dict(orient="records"),
        "action_difference_count": action_difference_count,
        "divergence_run_count": int(len(runs)),
        "divergence_runs": runs.to_dict(orient="records"),
        "complete_xor_differing_episode_count": complete_count,
        "distinct_entry_years": entry_years,
        "distinct_entry_year_count": len(entry_years),
        "sufficient_exposure": sufficient,
        "stress_10bps_incremental_active_log_edge": incremental_10bps,
        "reconciliations": reconciliations,
        "xor_orientation_values": [
            "online_cash_frozen_long",
            "online_long_frozen_cash",
        ],
        "online_learning_historical_value_demonstrated": useful_demonstrated,
        "reliable_prospective_evidence": False,
        "conclusion": conclusion,
    }
    return diagnostic, difference_frames, crossings, runs


def _adaptive_value_diagnostic(
    frame: pd.DataFrame,
    online_forecast: pd.DataFrame,
    frozen_forecast: pd.DataFrame,
    online_metrics: Mapping[str, Any],
    frozen_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return _adaptive_value_analysis(
        frame,
        online_forecast,
        frozen_forecast,
        online_metrics,
        frozen_metrics,
    )[0]


def _pre_2024_disclosure(gate_report: Mapping[str, Any]) -> dict[str, Any]:
    inputs = gate_report.get("gate_inputs", {})
    gate_values = gate_report.get("gates")
    failures = gate_report.get("failures")
    expected_failures = {
        "base_5bps_minimum_150_complete_cash_episodes",
        "stress_10bps_minimum_150_complete_cash_episodes",
    }
    if (
        not isinstance(inputs, dict)
        or gate_report.get("passed") is not False
        or not isinstance(gate_values, dict)
        or any(not isinstance(value, bool) for value in gate_values.values())
        or not isinstance(failures, list)
        or any(not isinstance(name, str) for name in failures)
        or set(failures) != expected_failures
        or len(failures) != len(expected_failures)
        or {name for name, value in gate_values.items() if not value}
        != expected_failures
    ):
        raise BinaryRegimeLongrunAuditError(
            "Through-2023 post-hoc gate status did not fail only the episode-count gate"
        )
    summary: dict[str, Any] = {}
    references = {
        "base_5bps": {
            "positive_period_count": 16,
            "complete_cash_episode_count": 137,
            "beneficial_episode_rate_percent_rounded_1dp": 64.2,
            "total_active_log_edge_rounded_4dp": 1.2767,
        },
        "stress_10bps": {
            "positive_period_count": 14,
            "complete_cash_episode_count": 137,
            "beneficial_episode_rate_percent_rounded_1dp": 62.0,
            "total_active_log_edge_rounded_4dp": 1.1397,
        },
    }
    for cost_name, expected in references.items():
        value = inputs.get(cost_name)
        if not isinstance(value, dict):
            raise BinaryRegimeLongrunAuditError(
                "Through-2023 gate inputs are incomplete"
            )
        observed = {
            "positive_period_count": value.get("positive_period_count"),
            "complete_cash_episode_count": value.get(
                "complete_cash_episode_count"
            ),
            "beneficial_episode_rate": value.get("beneficial_episode_rate"),
            "beneficial_episode_rate_percent_rounded_1dp": round(
                float(value["beneficial_episode_rate"]) * 100.0, 1
            ),
            "total_active_log_edge": value.get("total_active_log_edge"),
            "total_active_log_edge_rounded_4dp": round(
                float(value["total_active_log_edge"]), 4
            ),
        }
        for name, expected_value in expected.items():
            if observed[name] != expected_value:
                raise BinaryRegimeLongrunAuditError(
                    "Through-2023 known gate inputs changed from the frozen disclosure"
                )
        summary[cost_name] = observed
    return {
        "known_before_new_2024_2026_audit": True,
        "source_period": "2005 through 2023",
        "post_hoc_thresholds_selected_after_this_history_was_known": True,
        "gate_report": dict(gate_report),
        "summary": summary,
        "plain_language_disclosure": (
            "Before opening the new audit rows, almost every proposed robustness "
            "condition already passed; the 137 episodes were below the post-hoc "
            "minimum of 150 at both costs."
        ),
    }


def _policy_comparison(
    online_metrics: Mapping[str, Any], frozen_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        cost_name: {
            "online_selector": online_metrics[cost_name]["selector"],
            "frozen_2023_selector": frozen_metrics[cost_name]["selector"],
            "fixed_union": online_metrics[cost_name]["union"],
            "always_long_aapl_control": online_metrics[cost_name]["always_long"],
            "online_selector_vs_fixed_union": online_metrics[cost_name][
                "selector_vs_union"
            ],
            "frozen_selector_vs_fixed_union": frozen_metrics[cost_name][
                "selector_vs_union"
            ],
        }
        for cost_name, _ in COST_SCENARIOS
    }


def _runtime_report(deadline: _Deadline) -> dict[str, Any]:
    elapsed = deadline.elapsed()
    return {
        "seconds_before_seal": elapsed,
        "limit_seconds": RUN_TIME_LIMIT_SECONDS,
        "within_limit": elapsed < RUN_TIME_LIMIT_SECONDS,
        "network_access": False,
        "news_calls": 0,
        "llm_calls": 0,
        "api_calls": 0,
        "external_cost_usd": 0.0,
    }


def _require_consistent_report_statuses(report: Mapping[str, Any]) -> None:
    post_hoc = report.get("post_hoc_long_run_robustness_pass")
    strict = report.get("strict_recent_history_pass")
    fixed = report.get("fixed_policy_candidate_for_prospective_paper")
    adaptive = report.get("adaptive_value", {})
    adaptive_status = adaptive.get("adaptive_status") if isinstance(adaptive, dict) else None
    demonstrated = report.get("online_learning_historical_value_demonstrated")
    online = report.get("online_learning_candidate_for_paper")
    role = report.get("online_arm_role_for_prospective_paper")
    allowed_adaptive = {
        "unexercised",
        "exercised_insufficient_evidence",
        "exercised_positive",
        "exercised_negative",
        "exercised_flat",
    }
    expected_fixed = post_hoc is True and strict is True
    expected_demonstrated = adaptive_status == "exercised_positive"
    expected_online = expected_fixed and expected_demonstrated
    expected_role = "learning_candidate" if expected_online else "shadow_challenger"
    if (
        report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != "post_hoc_long_run_audit"
        or report.get("run_id") != AUDIT_RUN_ID
        or report.get("evidence_classification") != EVIDENCE_CLASSIFICATION
        or report.get("parent_validation_rejection_remains_final") is not True
        or report.get("parent_validation_failures")
        != list(EXPECTED_VALIDATION_FAILURES)
        or report.get("physical_data_end") != "2026-07-09"
        or report.get("ytd_label") != "2026 YTD through 2026-07-09"
        or report.get("continuous_account_start") != "2005-01-01"
        or report.get("continuous_account_reset_count_after_inception") != 0
        or report.get("sentiment_inputs") != SENTIMENT_DISCLOSURE
        or report.get("historical_results_authorize_real_capital") is not False
        or not isinstance(report.get("online_metrics_2005_through_2026_07_09"), dict)
        or not isinstance(report.get("frozen_2023_metrics_2005_through_2026_07_09"), dict)
        or not isinstance(report.get("policy_comparison"), dict)
        or not isinstance(report.get("online_integrity"), dict)
        or not isinstance(report.get("frozen_2023_integrity"), dict)
        or not isinstance(report.get("comparator_integrity"), dict)
        or not isinstance(post_hoc, bool)
        or not isinstance(strict, bool)
        or adaptive_status not in allowed_adaptive
        or fixed is not expected_fixed
        or demonstrated is not expected_demonstrated
        or online is not expected_online
        or role != expected_role
    ):
        raise BinaryRegimeLongrunAuditError(
            "Report candidate and adaptive statuses are inconsistent"
        )
    _require_gate_report_consistency(
        report.get("strict_recent_history_gate_report"),
        named_pass_field="strict_recent_history_pass",
        expected_pass=strict,
    )
    _require_gate_report_consistency(
        report.get("post_hoc_long_run_gate_report"),
        named_pass_field="post_hoc_long_run_robustness_pass",
        expected_pass=post_hoc,
    )
    strict_report = report["strict_recent_history_gate_report"]
    long_report = report["post_hoc_long_run_gate_report"]
    if (
        strict_report.get("evidence_classification")
        != "repeated_historical_target_non_confirmatory"
        or strict_report.get("ytd_label") != "2026 YTD through 2026-07-09"
        or long_report.get("post_hoc_non_confirmatory") is not True
    ):
        raise BinaryRegimeLongrunAuditError(
            "Embedded gate evidence classification is inconsistent"
        )
    _require_adaptive_report_consistency(
        adaptive, expected_demonstrated=demonstrated
    )
    _require_runtime_report(report.get("runtime"))
    pre_2024 = report.get("pre_2024_gate_status")
    if not isinstance(pre_2024, dict):
        raise BinaryRegimeLongrunAuditError("Pre-2024 disclosure is missing")
    expected_pre_2024 = _pre_2024_disclosure(pre_2024.get("gate_report", {}))
    if _canonical_json_bytes(pre_2024) != _canonical_json_bytes(expected_pre_2024):
        raise BinaryRegimeLongrunAuditError(
            "Pre-2024 disclosure contradicts its frozen gate inputs"
        )


def _required_audit_payload_names() -> frozenset[str]:
    names = {
        ATTEMPT_LOCK_FILENAME,
        "authorized_rejected_validation_manifest.json",
        "input_provenance.json",
        "prefix_preflight_provenance.json",
        "pre_2024_gate_status.json",
        "pre_2024_metrics.json",
        "audit_prices_through_2026_07_09.csv",
        "audit_online_forecast_2024_through_2026_07_09.csv",
        "audit_frozen_2023_forecast_2024_through_2026_07_09.csv",
        "audit_online_metrics.json",
        "audit_frozen_2023_metrics.json",
        "audit_policy_comparison.json",
        "strict_recent_history_gate_report.json",
        "post_hoc_long_run_gate_report.json",
        "adaptive_value.json",
        "adaptive_threshold_crossings.csv",
        "adaptive_divergence_runs.csv",
        "online_checkpoint_through_2026_07_09.json",
    }
    for prefix in (
        "pre_2024_online",
        "longrun_online",
        "longrun_frozen_2023",
    ):
        for cost_name, _ in COST_SCENARIOS:
            names.update(
                {
                    f"{prefix}_{cost_name}_ledgers.csv",
                    f"{prefix}_{cost_name}_selector_episodes.csv",
                    f"{prefix}_{cost_name}_union_episodes.csv",
                    f"{prefix}_{cost_name}_veto_benefits.csv",
                }
            )
    for cost_name, _ in COST_SCENARIOS:
        names.add(f"adaptive_{cost_name}_xor_differing_episodes.csv")
    return frozenset(names)


def _require_gate_report_consistency(
    value: Any, *, named_pass_field: str, expected_pass: bool
) -> None:
    if not isinstance(value, dict):
        raise BinaryRegimeLongrunAuditError("Embedded gate report is missing")
    gates = value.get("gates")
    failures = value.get("failures")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(not isinstance(name, str) or not isinstance(passed, bool) for name, passed in gates.items())
        or not isinstance(failures, list)
        or any(not isinstance(name, str) for name in failures)
    ):
        raise BinaryRegimeLongrunAuditError("Embedded gate report is malformed")
    expected_failures = [name for name, passed in gates.items() if not passed]
    calculated_pass = not expected_failures
    if (
        failures != expected_failures
        or value.get("passed") is not calculated_pass
        or value.get(named_pass_field) is not calculated_pass
        or calculated_pass is not expected_pass
    ):
        raise BinaryRegimeLongrunAuditError(
            "Embedded gate report contradicts its top-level status"
        )


def _require_adaptive_report_consistency(
    adaptive: Any, *, expected_demonstrated: bool
) -> None:
    if not isinstance(adaptive, dict):
        raise BinaryRegimeLongrunAuditError("Adaptive diagnostic is missing")
    status = adaptive.get("adaptive_status")
    crossings = adaptive.get("state_threshold_crossing_count")
    crossing_rows = adaptive.get("state_threshold_crossings")
    action_differences = adaptive.get("action_difference_count")
    divergence_count = adaptive.get("divergence_run_count")
    divergence_rows = adaptive.get("divergence_runs")
    complete = adaptive.get("complete_xor_differing_episode_count")
    years = adaptive.get("distinct_entry_years")
    year_count = adaptive.get("distinct_entry_year_count")
    sufficient = adaptive.get("sufficient_exposure")
    edge = _finite_number(adaptive.get("stress_10bps_incremental_active_log_edge"))
    if (
        not isinstance(crossings, int)
        or crossings < 0
        or not isinstance(crossing_rows, list)
        or len(crossing_rows) != crossings
        or not isinstance(action_differences, int)
        or action_differences < 0
        or not isinstance(divergence_count, int)
        or divergence_count < 0
        or not isinstance(divergence_rows, list)
        or len(divergence_rows) != divergence_count
        or not isinstance(complete, int)
        or complete < 0
        or not isinstance(years, list)
        or any(not isinstance(year, int) for year in years)
        or years != sorted(set(years))
        or year_count != len(years)
        or not isinstance(sufficient, bool)
        or sufficient
        is not (
            complete >= MIN_ADAPTIVE_EPISODES
            and len(years) >= MIN_ADAPTIVE_ENTRY_YEARS
        )
        or edge is None
        or adaptive.get("online_learning_historical_value_demonstrated")
        is not expected_demonstrated
        or adaptive.get("reliable_prospective_evidence") is not False
        or set(adaptive.get("reconciliations", {}))
        != {cost_name for cost_name, _ in COST_SCENARIOS}
        or adaptive.get("xor_orientation_values")
        != ["online_cash_frozen_long", "online_long_frozen_cash"]
    ):
        raise BinaryRegimeLongrunAuditError(
            "Adaptive diagnostic exposure or reconciliation fields are inconsistent"
        )
    if crossings == 0 or action_differences == 0:
        expected_status = "unexercised"
    elif not sufficient:
        expected_status = "exercised_insufficient_evidence"
    elif edge > ZERO_TOLERANCE:
        expected_status = "exercised_positive"
    elif edge < -ZERO_TOLERANCE:
        expected_status = "exercised_negative"
    else:
        expected_status = "exercised_flat"
    if (
        status != expected_status
        or expected_demonstrated is not (status == "exercised_positive")
        or adaptive.get("conclusion") != ADAPTIVE_CONCLUSIONS.get(status)
    ):
        raise BinaryRegimeLongrunAuditError(
            "Adaptive status was relabeled inconsistently with its evidence"
        )


def _require_runtime_report(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "seconds_before_seal",
        "limit_seconds",
        "within_limit",
        "network_access",
        "news_calls",
        "llm_calls",
        "api_calls",
        "external_cost_usd",
    }:
        raise BinaryRegimeLongrunAuditError("Runtime and cost report is incomplete")
    elapsed = _finite_number(value.get("seconds_before_seal"))
    cost = _finite_number(value.get("external_cost_usd"))
    if (
        elapsed is None
        or elapsed < 0.0
        or elapsed >= RUN_TIME_LIMIT_SECONDS
        or value.get("limit_seconds") != RUN_TIME_LIMIT_SECONDS
        or value.get("within_limit") is not True
        or value.get("network_access") is not False
        or value.get("news_calls") != 0
        or value.get("llm_calls") != 0
        or value.get("api_calls") != 0
        or cost != 0.0
    ):
        raise BinaryRegimeLongrunAuditError(
            "Runtime, network, API, or external-cost safety is inconsistent"
        )


def _require_json_payload_matches(
    payloads: Mapping[str, bytes], filename: str, expected: Any
) -> None:
    if payloads.get(filename) != _pretty_json_bytes(expected):
        raise BinaryRegimeLongrunAuditError(
            f"Sealed JSON payload contradicts the report: {filename}"
        )


def _csv_payload_frame(
    payloads: Mapping[str, bytes], filename: str
) -> pd.DataFrame:
    try:
        return pd.read_csv(
            io.BytesIO(payloads[filename]), float_precision="round_trip"
        )
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise BinaryRegimeLongrunAuditError(
            f"Sealed CSV payload is unreadable: {filename}"
        ) from exc


def _normalized_frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        normalized: dict[str, Any] = {}
        for name, raw in row.items():
            if pd.isna(raw):
                value: Any = None
            elif isinstance(raw, np.generic):
                value = raw.item()
            else:
                value = raw
            normalized[str(name)] = value
        records.append(normalized)
    return records


def _selector_episode_payloads(
    payloads: Mapping[str, bytes], *, prefix: str
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        cost_name: {
            "selector": _csv_payload_frame(
                payloads, f"{prefix}_{cost_name}_selector_episodes.csv"
            )
        }
        for cost_name, _ in COST_SCENARIOS
    }


def _require_recomputed_gate_reports(
    report: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> None:
    metrics = report["online_metrics_2005_through_2026_07_09"]
    integrity = report["online_integrity"]
    strict = apply_strict_recent_history_gates(metrics, integrity)
    long_run = apply_post_hoc_long_run_gates(
        metrics,
        _selector_episode_payloads(payloads, prefix="longrun_online"),
        integrity,
    )
    if (
        _canonical_json_bytes(strict)
        != _canonical_json_bytes(report["strict_recent_history_gate_report"])
        or _canonical_json_bytes(long_run)
        != _canonical_json_bytes(report["post_hoc_long_run_gate_report"])
    ):
        raise BinaryRegimeLongrunAuditError(
            "Sealed gate reports do not recompute from their bound metrics and episodes"
        )


def _require_adaptive_payload_consistency(
    report: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> None:
    adaptive = report["adaptive_value"]
    online_metrics = report["online_metrics_2005_through_2026_07_09"]
    frozen_metrics = report["frozen_2023_metrics_2005_through_2026_07_09"]
    complete = adaptive["complete_xor_differing_episode_count"]
    expected_years = adaptive["distinct_entry_years"]
    period_names = tuple(str(year) for year in range(2005, 2026)) + (
        "2026_ytd",
    )
    expected_columns = {
        "decision_date",
        "entry_date",
        "exit_date",
        "entry_period",
        "orientation",
        "online_target",
        "frozen_target",
        "raw_cash_edge",
        "net_cash_edge",
        "online_minus_frozen_active_log_edge",
    }
    allowed_orientations = {
        "online_cash_frozen_long",
        "online_long_frozen_cash",
    }
    observed_edges: dict[str, float] = {}
    shared_episode_identity: list[dict[str, Any]] | None = None
    xor_decision_dates: list[pd.Timestamp] = []
    reconciliations = adaptive["reconciliations"]
    for cost_name, cost_bps in COST_SCENARIOS:
        frame = _csv_payload_frame(
            payloads,
            f"adaptive_{cost_name}_xor_differing_episodes.csv",
        )
        if set(frame.columns) != expected_columns or len(frame) != complete:
            raise BinaryRegimeLongrunAuditError(
                "Adaptive XOR episode CSV schema or count contradicts its diagnostic"
            )
        numeric_columns = (
            "online_target",
            "frozen_target",
            "raw_cash_edge",
            "net_cash_edge",
            "online_minus_frozen_active_log_edge",
        )
        numeric = {
            name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            for name in numeric_columns
        }
        online_targets = numeric["online_target"]
        frozen_targets = numeric["frozen_target"]
        raw_edges = numeric["raw_cash_edge"]
        net_edges = numeric["net_cash_edge"]
        signed = numeric["online_minus_frozen_active_log_edge"]
        decision_dates = pd.to_datetime(frame["decision_date"], errors="coerce")
        dates = pd.to_datetime(frame["entry_date"], errors="coerce")
        exit_dates = pd.to_datetime(frame["exit_date"], errors="coerce")
        periods = frame["entry_period"].astype(str)
        orientation_values = frame["orientation"].astype(str)
        orientations = set(orientation_values)
        expected_orientations = np.where(
            online_targets == 0.0,
            "online_cash_frozen_long",
            "online_long_frozen_cash",
        )
        expected_signed = np.where(online_targets == 0.0, net_edges, -net_edges)
        friction = math.log(
            (1.0 - float(cost_bps) / 10_000.0)
            / (1.0 + float(cost_bps) / 10_000.0)
        )
        if (
            any(not np.isfinite(values).all() for values in numeric.values())
            or bool(
                np.any(~np.isin(online_targets, (0.0, 1.0)))
                or np.any(~np.isin(frozen_targets, (0.0, 1.0)))
                or np.any(online_targets == frozen_targets)
            )
            or not np.array_equal(
                orientation_values.to_numpy(dtype=str), expected_orientations
            )
            or bool(
                np.any(np.abs(expected_signed - signed) > IDENTITY_TOLERANCE)
                or np.any(
                    np.abs((raw_edges + friction) - net_edges)
                    > IDENTITY_TOLERANCE
                )
            )
            or bool(decision_dates.isna().any())
            or bool(dates.isna().any())
            or bool(exit_dates.isna().any())
            or bool(np.any(decision_dates >= dates))
            or bool(np.any(dates >= exit_dates))
            or not orientations.issubset(allowed_orientations)
            or not set(periods).issubset(set(period_names))
        ):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive XOR episode CSV contains invalid evidence"
            )
        identity_columns = (
            "decision_date",
            "entry_date",
            "exit_date",
            "entry_period",
            "orientation",
            "online_target",
            "frozen_target",
            "raw_cash_edge",
        )
        identity = _normalized_frame_records(frame.loc[:, identity_columns])
        if shared_episode_identity is None:
            shared_episode_identity = identity
            xor_decision_dates = sorted(pd.Timestamp(value) for value in decision_dates)
        elif _canonical_json_bytes(identity) != _canonical_json_bytes(
            shared_episode_identity
        ):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive cost scenarios do not share exact XOR episode identity"
            )
        derived_periods = pd.Series(
            [
                "2026_ytd" if value.year == 2026 else str(value.year)
                for value in dates
            ],
            dtype=str,
        )
        if not periods.reset_index(drop=True).equals(derived_periods):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive XOR entry periods do not match their entry dates"
            )
        years = sorted({int(value.year) for value in dates})
        if years != expected_years:
            raise BinaryRegimeLongrunAuditError(
                "Adaptive XOR entry years contradict the diagnostic"
            )
        observed_full = float(np.sum(signed))
        observed_edges[cost_name] = observed_full
        expected_full = _finite_number(
            online_metrics[cost_name]["selector"].get(
                "continuous_account_active_log_edge"
            )
        )
        frozen_full = _finite_number(
            frozen_metrics[cost_name]["selector"].get(
                "continuous_account_active_log_edge"
            )
        )
        reconciliation = reconciliations.get(cost_name)
        if (
            expected_full is None
            or frozen_full is None
            or not isinstance(reconciliation, dict)
            or set(reconciliation)
            != {
                "complete_xor_episode_count",
                "full_account_online_minus_frozen_active_log_edge",
                "xor_episode_incremental_edge",
                "full_account_identity_error",
                "period_incremental_edges",
                "period_identity_errors",
                "identity_tolerance",
            }
        ):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive reconciliation inventory is incomplete"
            )
        expected_difference = expected_full - frozen_full
        recorded_full = _finite_number(
            reconciliation["full_account_online_minus_frozen_active_log_edge"]
        )
        recorded_xor = _finite_number(
            reconciliation["xor_episode_incremental_edge"]
        )
        recorded_error = _finite_number(
            reconciliation["full_account_identity_error"]
        )
        tolerance = _finite_number(reconciliation["identity_tolerance"])
        calculated_error = expected_difference - observed_full
        if (
            reconciliation["complete_xor_episode_count"] != complete
            or recorded_full is None
            or recorded_xor is None
            or recorded_error is None
            or tolerance != IDENTITY_TOLERANCE
            or abs(recorded_full - expected_difference) > IDENTITY_TOLERANCE
            or abs(recorded_xor - observed_full) > IDENTITY_TOLERANCE
            or abs(recorded_error - calculated_error) > IDENTITY_TOLERANCE
            or abs(recorded_error) > IDENTITY_TOLERANCE
        ):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive full-account reconciliation contradicts bound evidence"
            )
        recorded_periods = reconciliation["period_incremental_edges"]
        recorded_period_errors = reconciliation["period_identity_errors"]
        if (
            not isinstance(recorded_periods, dict)
            or set(recorded_periods) != set(period_names)
            or not isinstance(recorded_period_errors, dict)
            or set(recorded_period_errors) != set(period_names)
        ):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive period reconciliation inventory is incomplete"
            )
        for period_name in period_names:
            online_period = _finite_number(
                online_metrics[cost_name]["selector"]["periods"][period_name].get(
                    "active_log_edge"
                )
            )
            frozen_period = _finite_number(
                frozen_metrics[cost_name]["selector"]["periods"][period_name].get(
                    "active_log_edge"
                )
            )
            recorded_period = _finite_number(recorded_periods[period_name])
            recorded_period_error = _finite_number(
                recorded_period_errors[period_name]
            )
            observed_period = float(np.sum(signed[periods == period_name]))
            if (
                online_period is None
                or frozen_period is None
                or recorded_period is None
                or recorded_period_error is None
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Adaptive period reconciliation contains nonfinite values"
                )
            expected_period = online_period - frozen_period
            calculated_period_error = expected_period - observed_period
            if (
                abs(recorded_period - observed_period) > IDENTITY_TOLERANCE
                or abs(recorded_period_error - calculated_period_error)
                > IDENTITY_TOLERANCE
                or abs(recorded_period_error) > IDENTITY_TOLERANCE
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Adaptive period reconciliation contradicts bound evidence"
                )

    stress_edge = _finite_number(
        adaptive["stress_10bps_incremental_active_log_edge"]
    )
    if (
        adaptive["action_difference_count"] != complete
        or stress_edge is None
        or abs(stress_edge - observed_edges["stress_10bps"])
        > IDENTITY_TOLERANCE
    ):
        raise BinaryRegimeLongrunAuditError(
            "Adaptive top-level edge or action count contradicts XOR evidence"
        )
    crossings = _csv_payload_frame(payloads, "adaptive_threshold_crossings.csv")
    crossing_columns = {
        "crossing_date",
        "regime",
        "transition",
        "pre_n_raw",
        "pre_n_eff",
        "pre_mean",
        "pre_cash_selected",
        "post_n_raw",
        "post_n_eff",
        "post_mean",
        "post_cash_selected",
        "matured_signal_close",
        "matured_label_10bps",
    }
    crossing_records = _normalized_frame_records(crossings)
    reported_crossings = [dict(row) for row in adaptive["state_threshold_crossings"]]
    for records in (crossing_records, reported_crossings):
        for row in records:
            for name in ("pre_n_raw", "post_n_raw"):
                row[name] = int(row[name])
            for name in (
                "pre_n_eff",
                "pre_mean",
                "post_n_eff",
                "post_mean",
                "matured_label_10bps",
            ):
                if row[name] is not None:
                    row[name] = float(row[name])
            for name in ("pre_cash_selected", "post_cash_selected"):
                row[name] = bool(row[name])
    if (
        set(crossings.columns) != crossing_columns
        or _canonical_json_bytes(crossing_records)
        != _canonical_json_bytes(reported_crossings)
    ):
        raise BinaryRegimeLongrunAuditError(
            "Adaptive threshold-crossing CSV contradicts its diagnostic"
        )
    crossing_dates: list[pd.Timestamp] = []
    for row in crossing_records:
        crossing_date = pd.to_datetime(row["crossing_date"], errors="coerce")
        matured_date = pd.to_datetime(
            row["matured_signal_close"], errors="coerce"
        )
        pre_n_raw = _finite_number(row["pre_n_raw"])
        post_n_raw = _finite_number(row["post_n_raw"])
        post_n_eff = _finite_number(row["post_n_eff"])
        post_mean = _finite_number(row["post_mean"])
        label = _finite_number(row["matured_label_10bps"])
        transition = row["transition"]
        pre_cash = row["pre_cash_selected"]
        post_cash = row["post_cash_selected"]
        transition_valid = (
            transition == "LONG_TO_CASH"
            and pre_cash is False
            and post_cash is True
            and post_mean is not None
            and post_mean > POSITIVE_MEAN_THRESHOLD
        ) or (
            transition == "CASH_TO_LONG"
            and pre_cash is True
            and post_cash is False
            and post_mean is not None
            and post_mean < NEGATIVE_MEAN_THRESHOLD
        )
        if (
            pd.isna(crossing_date)
            or pd.isna(matured_date)
            or crossing_date <= VALIDATION_END
            or crossing_date > AUDIT_END
            or matured_date >= crossing_date
            or row["regime"] not in REGIME_NAMES
            or pre_n_raw is None
            or post_n_raw is None
            or post_n_raw <= pre_n_raw
            or post_n_eff is None
            or post_n_eff <= 0.0
            or label is None
            or not transition_valid
        ):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive threshold-crossing row is not causal evidence"
            )
        crossing_dates.append(pd.Timestamp(crossing_date))
    if (
        adaptive["action_difference_count"] > 0
        and crossing_dates
        and xor_decision_dates
        and min(crossing_dates) > max(xor_decision_dates)
    ):
        raise BinaryRegimeLongrunAuditError(
            "Adaptive threshold crossings occur only after all XOR decisions"
        )

    divergence_runs = _csv_payload_frame(payloads, "adaptive_divergence_runs.csv")
    divergence_columns = {
        "run_id",
        "first_decision_date",
        "last_decision_date",
        "sessions",
    }
    divergence_records = _normalized_frame_records(divergence_runs)
    reported_divergence = [dict(row) for row in adaptive["divergence_runs"]]
    for records in (divergence_records, reported_divergence):
        for row in records:
            row["run_id"] = int(row["run_id"])
            row["sessions"] = int(row["sessions"])
    if (
        len(crossings) != adaptive["state_threshold_crossing_count"]
        or len(divergence_runs) != adaptive["divergence_run_count"]
        or set(divergence_runs.columns) != divergence_columns
        or _canonical_json_bytes(divergence_records)
        != _canonical_json_bytes(reported_divergence)
    ):
        raise BinaryRegimeLongrunAuditError(
            "Adaptive crossing or divergence CSV count contradicts its diagnostic"
        )
    covered_decisions: list[pd.Timestamp] = []
    session_total = 0
    for expected_run_id, row in enumerate(divergence_records, start=1):
        first = pd.to_datetime(row["first_decision_date"], errors="coerce")
        last = pd.to_datetime(row["last_decision_date"], errors="coerce")
        sessions = _finite_number(row["sessions"])
        contained = [
            value
            for value in xor_decision_dates
            if not pd.isna(first)
            and not pd.isna(last)
            and first <= value <= last
        ]
        if (
            row["run_id"] != expected_run_id
            or pd.isna(first)
            or pd.isna(last)
            or first > last
            or sessions is None
            or not sessions.is_integer()
            or int(sessions) <= 0
            or len(contained) != int(sessions)
            or not contained
            or contained[0] != first
            or contained[-1] != last
        ):
            raise BinaryRegimeLongrunAuditError(
                "Adaptive divergence run does not cover its XOR decisions"
            )
        session_total += int(sessions)
        covered_decisions.extend(contained)
    if (
        session_total != adaptive["action_difference_count"]
        or covered_decisions != xor_decision_dates
    ):
        raise BinaryRegimeLongrunAuditError(
            "Adaptive divergence sessions do not reconcile to action differences"
        )


def _require_online_artifact_consistency(
    report: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> None:
    metrics = report["online_metrics_2005_through_2026_07_09"]
    ledger_columns = {
        "policy",
        "ledger_role",
        "decision_date",
        "fill_date",
        "adjusted_open",
        "equity_before_fill",
        "equity",
        "cash",
        "shares",
        "holding_exposure_for_return",
        "target_exposure",
        "new_exposure_after_fill",
        "signed_share_delta",
        "reference_price",
        "fill_price",
        "turnover",
        "fees",
        "slippage",
        "margin_interest",
        "trade_executed",
        "daily_return",
        "monetary_pnl",
        "drawdown",
    }
    numeric_ledger_columns = tuple(
        name
        for name in ledger_columns
        if name
        not in {
            "policy",
            "ledger_role",
            "decision_date",
            "fill_date",
            "trade_executed",
        }
    )
    period_names = tuple(str(year) for year in range(2005, 2026)) + (
        "2026_ytd",
    )
    periods = _annual_periods(2025)
    reference_ledger_dates: list[str] | None = None
    for cost_name, cost_bps in COST_SCENARIOS:
        ledger = _csv_payload_frame(
            payloads, f"longrun_online_{cost_name}_ledgers.csv"
        )
        if set(ledger.columns) != ledger_columns or ledger.empty:
            raise BinaryRegimeLongrunAuditError(
                "Long-run online ledger schema is incomplete"
            )
        numeric = ledger.loc[:, numeric_ledger_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise BinaryRegimeLongrunAuditError(
                "Long-run online ledger contains nonfinite values"
            )
        policy_roles = {
            "selector": "strategy",
            "union": "strategy",
            "always_long": "strategy",
            "aapl_buy_hold": "benchmark",
        }
        if set(ledger["policy"].astype(str)) != set(policy_roles):
            raise BinaryRegimeLongrunAuditError(
                "Long-run online ledger policy inventory is not exact"
            )
        benchmark = ledger.loc[
            ledger["policy"].astype(str) == "aapl_buy_hold"
        ].reset_index(drop=True)
        benchmark_dates = benchmark["fill_date"].astype(str).tolist()
        parsed_benchmark_dates = pd.to_datetime(
            benchmark["fill_date"], errors="coerce"
        )
        if (
            not benchmark_dates
            or bool(parsed_benchmark_dates.isna().any())
            or not parsed_benchmark_dates.is_monotonic_increasing
            or bool(parsed_benchmark_dates.duplicated().any())
            or benchmark_dates[0] != "2005-01-03"
            or benchmark_dates[-1] != "2026-07-09"
        ):
            raise BinaryRegimeLongrunAuditError(
                "Long-run online benchmark ledger dates are incomplete"
            )
        if reference_ledger_dates is None:
            reference_ledger_dates = benchmark_dates
        elif benchmark_dates != reference_ledger_dates:
            raise BinaryRegimeLongrunAuditError(
                "Long-run online ledger date sequences differ across costs"
            )
        policy_frames: dict[str, pd.DataFrame] = {}
        for policy, role in policy_roles.items():
            rows = ledger.loc[ledger["policy"].astype(str) == policy].reset_index(
                drop=True
            )
            if (
                rows.empty
                or set(rows["ledger_role"].astype(str)) != {role}
                or rows["fill_date"].astype(str).tolist() != benchmark_dates
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Long-run online strategy/benchmark ledgers are misaligned"
                )
            policy_frames[policy] = rows
            targets = pd.to_numeric(rows["target_exposure"]).to_numpy(dtype=float)
            post_fill = pd.to_numeric(
                rows["new_exposure_after_fill"]
            ).to_numpy(dtype=float)
            holding = pd.to_numeric(
                rows["holding_exposure_for_return"]
            ).to_numpy(dtype=float)
            cash = pd.to_numeric(rows["cash"]).to_numpy(dtype=float)
            shares = pd.to_numeric(rows["shares"]).to_numpy(dtype=float)
            margin = pd.to_numeric(rows["margin_interest"]).to_numpy(dtype=float)
            equity = pd.to_numeric(rows["equity"]).to_numpy(dtype=float)
            price = pd.to_numeric(rows["adjusted_open"]).to_numpy(dtype=float)
            calculated_equity = cash + shares * price
            expected_returns = np.diff(
                np.r_[_selector_runner.INITIAL_CASH, equity]
            ) / np.r_[_selector_runner.INITIAL_CASH, equity[:-1]]
            recorded_returns = pd.to_numeric(rows["daily_return"]).to_numpy(
                dtype=float
            )
            expected_pnl = np.diff(
                np.r_[_selector_runner.INITIAL_CASH, equity]
            )
            recorded_pnl = pd.to_numeric(rows["monetary_pnl"]).to_numpy(
                dtype=float
            )
            expected_drawdown = equity / np.maximum.accumulate(
                np.r_[_selector_runner.INITIAL_CASH, equity]
            )[1:] - 1.0
            recorded_drawdown = pd.to_numeric(rows["drawdown"]).to_numpy(
                dtype=float
            )
            if (
                np.any(targets < -IDENTITY_TOLERANCE)
                or np.any(targets > 1.0 + IDENTITY_TOLERANCE)
                or np.any(post_fill < -IDENTITY_TOLERANCE)
                or np.any(post_fill > 1.0 + IDENTITY_TOLERANCE)
                or np.any(holding < -IDENTITY_TOLERANCE)
                or np.any(holding > 1.0 + IDENTITY_TOLERANCE)
                or np.any(cash < -IDENTITY_TOLERANCE)
                or np.any(shares < -IDENTITY_TOLERANCE)
                or np.any(np.abs(margin) > IDENTITY_TOLERANCE)
                or np.any(equity <= 0.0)
                or np.any(
                    np.abs(calculated_equity - equity) > IDENTITY_TOLERANCE
                )
                or np.any(
                    np.abs(expected_returns - recorded_returns)
                    > IDENTITY_TOLERANCE
                )
                or np.any(
                    np.abs(expected_pnl - recorded_pnl) > IDENTITY_TOLERANCE
                )
                or np.any(
                    np.abs(expected_drawdown - recorded_drawdown)
                    > IDENTITY_TOLERANCE
                )
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Long-run online ledger violates unleveraged accounting identities"
                )
            if policy == "aapl_buy_hold":
                continue
            result = metrics[cost_name][policy]
            strategy_ledger = rows.drop(columns=["policy", "ledger_role"])
            benchmark_ledger = benchmark.drop(
                columns=["policy", "ledger_role"]
            )
            try:
                recalculated_proof = _selector_runner.assert_unleveraged_ledger(
                    strategy_ledger
                )
                recalculated_comparison = _selector_runner.compare_ledgers(
                    strategy_ledger,
                    benchmark_ledger,
                    initial_cash=_selector_runner.INITIAL_CASH,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise BinaryRegimeLongrunAuditError(
                    "Long-run policy ledger cannot reproduce its bound metrics"
                ) from exc
            if _canonical_json_bytes(recalculated_proof) != _canonical_json_bytes(
                result.get("no_leverage_proof")
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Long-run ledger no-leverage proof contradicts bound metrics"
                )
            strategy_return = float(
                recalculated_comparison["strategy"]["total_return"]
            )
            benchmark_return = float(
                recalculated_comparison["aapl_buy_hold"]["total_return"]
            )
            comparison = result.get("comparison", {})
            bound_strategy = _finite_number(
                comparison.get("strategy", {}).get("total_return")
            )
            bound_benchmark = _finite_number(
                comparison.get("aapl_buy_hold", {}).get("total_return")
            )
            bound_edge = _finite_number(
                result.get("continuous_account_active_log_edge")
            )
            bound_strategy_drawdown = _finite_number(
                comparison.get("strategy", {}).get("max_drawdown")
            )
            bound_benchmark_drawdown = _finite_number(
                comparison.get("aapl_buy_hold", {}).get("max_drawdown")
            )
            calculated_edge = math.log1p(strategy_return) - math.log1p(
                benchmark_return
            )
            if (
                bound_strategy is None
                or bound_benchmark is None
                or bound_edge is None
                or bound_strategy_drawdown is None
                or bound_benchmark_drawdown is None
                or abs(bound_strategy - strategy_return) > IDENTITY_TOLERANCE
                or abs(bound_benchmark - benchmark_return) > IDENTITY_TOLERANCE
                or abs(bound_edge - calculated_edge) > IDENTITY_TOLERANCE
                or abs(
                    bound_strategy_drawdown
                    - recalculated_comparison["strategy"]["max_drawdown"]
                )
                > IDENTITY_TOLERANCE
                or abs(
                    bound_benchmark_drawdown
                    - recalculated_comparison["aapl_buy_hold"]["max_drawdown"]
                )
                > IDENTITY_TOLERANCE
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Long-run ledger terminal wealth contradicts bound metrics"
                )
            for period in periods:
                try:
                    period_strategy_return = _selector_runner._period_return(
                        strategy_ledger,
                        period,
                        initial_cash=_selector_runner.INITIAL_CASH,
                    )
                    period_benchmark_return = _selector_runner._period_return(
                        benchmark_ledger,
                        period,
                        initial_cash=_selector_runner.INITIAL_CASH,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise BinaryRegimeLongrunAuditError(
                        "Long-run ledger cannot reproduce annual period returns"
                    ) from exc
                period_edge = math.log1p(period_strategy_return) - math.log1p(
                    period_benchmark_return
                )
                bound_period = result.get("periods", {}).get(period.name, {})
                bound_period_strategy = _finite_number(
                    bound_period.get("strategy_return")
                )
                bound_period_benchmark = _finite_number(
                    bound_period.get("aapl_buy_hold_return")
                )
                bound_period_edge = _finite_number(
                    bound_period.get("ledger_boundary_active_log_edge")
                )
                if (
                    bound_period_strategy is None
                    or bound_period_benchmark is None
                    or bound_period_edge is None
                    or abs(bound_period_strategy - period_strategy_return)
                    > IDENTITY_TOLERANCE
                    or abs(bound_period_benchmark - period_benchmark_return)
                    > IDENTITY_TOLERANCE
                    or abs(bound_period_edge - period_edge)
                    > IDENTITY_TOLERANCE
                ):
                    raise BinaryRegimeLongrunAuditError(
                        "Long-run ledger period returns contradict bound metrics"
                    )

        always_long_path = policy_frames["always_long"].drop(
            columns=["policy", "ledger_role"]
        )
        benchmark_path = policy_frames["aapl_buy_hold"].drop(
            columns=["policy", "ledger_role"]
        )
        if not always_long_path.equals(benchmark_path):
            raise BinaryRegimeLongrunAuditError(
                "Long-run always-long and benchmark ledger paths differ"
            )

        friction = math.log(
            (1.0 - cost_bps / 10_000.0) / (1.0 + cost_bps / 10_000.0)
        )
        for episode_policy in ("selector", "union"):
            policy_episodes = _csv_payload_frame(
                payloads,
                f"longrun_online_{cost_name}_{episode_policy}_episodes.csv",
            )
            if tuple(policy_episodes.columns) != tuple(
                _selector_runner._EPISODE_COLUMNS
            ):
                raise BinaryRegimeLongrunAuditError(
                    f"Long-run {episode_policy} episode schema is not exact"
                )
            entry_dates = pd.to_datetime(
                policy_episodes["entry_date"], errors="coerce"
            )
            decision_dates = pd.to_datetime(
                policy_episodes["decision_date"], errors="coerce"
            )
            exit_dates = pd.to_datetime(
                policy_episodes["exit_date"], errors="coerce"
            )
            raw_edges = pd.to_numeric(
                policy_episodes["raw_active_log_edge"], errors="coerce"
            ).to_numpy(dtype=float)
            net_edges = pd.to_numeric(
                policy_episodes["net_active_log_edge"], errors="coerce"
            ).to_numpy(dtype=float)
            episode_metrics = metrics[cost_name][episode_policy]
            recognized_periods = [
                "2026_ytd" if value.year == 2026 else str(value.year)
                for value in entry_dates
            ]
            policy_open_by_date = pd.Series(
                pd.to_numeric(
                    policy_frames[episode_policy]["adjusted_open"]
                ).to_numpy(dtype=float),
                index=pd.to_datetime(
                    policy_frames[episode_policy]["fill_date"]
                ),
            )
            try:
                entry_opens = policy_open_by_date.loc[
                    entry_dates.to_list()
                ].to_numpy(dtype=float)
                exit_opens = policy_open_by_date.loc[
                    exit_dates.to_list()
                ].to_numpy(dtype=float)
            except KeyError as exc:
                raise BinaryRegimeLongrunAuditError(
                    f"Long-run {episode_policy} episode dates lack ledger opens"
                ) from exc
            ledger_raw_edges = np.log(entry_opens / exit_opens)
            if (
                bool(entry_dates.isna().any())
                or bool(decision_dates.isna().any())
                or bool(exit_dates.isna().any())
                or not entry_dates.is_monotonic_increasing
                or bool(entry_dates.duplicated().any())
                or bool(np.any(decision_dates >= entry_dates))
                or bool(np.any(entry_dates >= exit_dates))
                or any(name not in period_names for name in recognized_periods)
                or not np.isfinite(raw_edges).all()
                or not np.isfinite(net_edges).all()
                or not np.isfinite(ledger_raw_edges).all()
                or bool(np.any(entry_opens <= 0.0))
                or bool(np.any(exit_opens <= 0.0))
                or bool(
                    np.any(
                        np.abs((raw_edges + friction) - net_edges)
                        > IDENTITY_TOLERANCE
                    )
                )
                or bool(
                    np.any(
                        np.abs(raw_edges - ledger_raw_edges)
                        > IDENTITY_TOLERANCE
                    )
                )
                or not np.array_equal(
                    policy_episodes["win"].astype(bool).to_numpy(),
                    net_edges > 0.0,
                )
                or episode_metrics.get("cash_episode_count")
                != len(policy_episodes)
            ):
                raise BinaryRegimeLongrunAuditError(
                    f"Long-run {episode_policy} episodes violate causal, ledger-open, or cost identities"
                )
            episode_total = float(np.sum(net_edges))
            for name in (
                "total_active_log_edge",
                "attributed_episode_active_log_edge",
                "continuous_account_active_log_edge",
                "continuous_account_attributed_episode_active_log_edge",
            ):
                value = _finite_number(episode_metrics.get(name))
                if value is None or abs(value - episode_total) > IDENTITY_TOLERANCE:
                    raise BinaryRegimeLongrunAuditError(
                        f"Long-run {episode_policy} episode total contradicts bound metrics"
                    )
            reported_episode_identity = _finite_number(
                episode_metrics.get(
                    "continuous_account_episode_ledger_identity_error"
                )
            )
            derived_episode_identity = float(
                episode_metrics["continuous_account_active_log_edge"]
                - episode_total
            )
            if (
                reported_episode_identity is None
                or abs(reported_episode_identity - derived_episode_identity)
                > IDENTITY_TOLERANCE
                or abs(derived_episode_identity) > IDENTITY_TOLERANCE
            ):
                raise BinaryRegimeLongrunAuditError(
                    f"Long-run {episode_policy} episode-ledger identity diagnostic contradicts artifacts"
                )
            for period_name in period_names:
                observed = float(
                    np.sum(
                        net_edges[
                            np.asarray(
                                [
                                    name == period_name
                                    for name in recognized_periods
                                ],
                                dtype=bool,
                            )
                        ]
                    )
                )
                bound = _finite_number(
                    episode_metrics["periods"][period_name].get(
                        "active_log_edge"
                    )
                )
                if bound is None or abs(bound - observed) > IDENTITY_TOLERANCE:
                    raise BinaryRegimeLongrunAuditError(
                        f"Long-run {episode_policy} episode period attribution contradicts metrics"
                    )

        union_metrics = metrics[cost_name]["union"]

        benefits = _csv_payload_frame(
            payloads, f"longrun_online_{cost_name}_veto_benefits.csv"
        )
        if set(benefits.columns) != set(_selector_runner._VETO_BENEFIT_COLUMNS):
            raise BinaryRegimeLongrunAuditError(
                "Long-run veto-benefit schema is not exact"
            )
        benefit_dates = pd.to_datetime(benefits["entry_date"], errors="coerce")
        benefit_decisions = pd.to_datetime(
            benefits["decision_date"], errors="coerce"
        )
        benefit_exits = pd.to_datetime(benefits["exit_date"], errors="coerce")
        raw_benefit = pd.to_numeric(
            benefits["raw_union_cash_edge"], errors="coerce"
        ).to_numpy(dtype=float)
        net_benefit = pd.to_numeric(
            benefits["net_union_cash_edge"], errors="coerce"
        ).to_numpy(dtype=float)
        veto = pd.to_numeric(benefits["veto_benefit"], errors="coerce").to_numpy(
            dtype=float
        )
        selector = metrics[cost_name]["selector"]
        incremental = metrics[cost_name]["selector_vs_union"]
        summary = incremental.get("veto_benefit", {})
        if (
            bool(benefit_dates.isna().any())
            or bool(benefit_decisions.isna().any())
            or bool(benefit_exits.isna().any())
            or bool(np.any(benefit_decisions >= benefit_dates))
            or bool(np.any(benefit_dates >= benefit_exits))
            or not np.isfinite(raw_benefit).all()
            or not np.isfinite(net_benefit).all()
            or not np.isfinite(veto).all()
            or bool(
                np.any(
                    np.abs((raw_benefit + friction) - net_benefit)
                    > IDENTITY_TOLERANCE
                )
                or np.any(np.abs(veto + net_benefit) > IDENTITY_TOLERANCE)
            )
            or not np.array_equal(
                benefits["beneficial_veto"].astype(bool).to_numpy(), veto > 0.0
            )
            or summary.get("veto_count") != len(benefits)
        ):
            raise BinaryRegimeLongrunAuditError(
                "Long-run veto-benefit rows violate causal or cost identities"
            )
        total_benefit = float(np.sum(veto))
        summary_total = _finite_number(summary.get("total_veto_benefit"))
        total_incremental = _finite_number(incremental.get("total_active_log_edge"))
        continuous_incremental = _finite_number(
            incremental.get("continuous_account_total_active_log_edge")
        )
        reporting_identity = _finite_number(
            incremental.get("veto_benefit_identity_error")
        )
        continuous_benefit = _finite_number(
            incremental.get("continuous_account_veto_benefit")
        )
        continuous_identity = _finite_number(
            incremental.get("continuous_account_veto_benefit_identity_error")
        )
        derived_reporting_identity = (
            float(total_incremental - total_benefit)
            if total_incremental is not None
            else math.nan
        )
        derived_continuous_identity = (
            float(continuous_incremental - total_benefit)
            if continuous_incremental is not None
            else math.nan
        )
        if (
            summary_total is None
            or abs(summary_total - total_benefit) > IDENTITY_TOLERANCE
            or total_incremental is None
            or abs(total_incremental - total_benefit) > IDENTITY_TOLERANCE
            or continuous_incremental is None
            or abs(continuous_incremental - total_benefit) > IDENTITY_TOLERANCE
            or reporting_identity is None
            or abs(reporting_identity - derived_reporting_identity)
            > IDENTITY_TOLERANCE
            or abs(derived_reporting_identity) > IDENTITY_TOLERANCE
            or continuous_benefit is None
            or abs(continuous_benefit - total_benefit) > IDENTITY_TOLERANCE
            or continuous_identity is None
            or abs(continuous_identity - derived_continuous_identity)
            > IDENTITY_TOLERANCE
            or abs(derived_continuous_identity) > IDENTITY_TOLERANCE
            or abs(
                selector["total_active_log_edge"]
                - union_metrics["total_active_log_edge"]
                - total_benefit
            )
            > IDENTITY_TOLERANCE
            or abs(
                selector["continuous_account_active_log_edge"]
                - union_metrics["continuous_account_active_log_edge"]
                - total_benefit
            )
            > IDENTITY_TOLERANCE
        ):
            raise BinaryRegimeLongrunAuditError(
                "Long-run veto total contradicts selector-minus-union metrics"
            )
        for period_name in period_names:
            observed = float(
                np.sum(
                    veto[
                        np.asarray(
                            [
                                ("2026_ytd" if value.year == 2026 else str(value.year))
                                == period_name
                                for value in benefit_dates
                            ],
                            dtype=bool,
                        )
                    ]
                )
            )
            summary_value = _finite_number(summary["periods"].get(period_name))
            incremental_value = _finite_number(
                incremental["periods"].get(period_name)
            )
            selector_difference = (
                selector["periods"][period_name]["active_log_edge"]
                - union_metrics["periods"][period_name]["active_log_edge"]
            )
            if (
                summary_value is None
                or incremental_value is None
                or abs(summary_value - observed) > IDENTITY_TOLERANCE
                or abs(incremental_value - observed) > IDENTITY_TOLERANCE
                or abs(selector_difference - observed) > IDENTITY_TOLERANCE
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Long-run veto period attribution contradicts metrics"
                )


def _require_attempt_lock_evidence(
    *,
    output_dir: Path,
    report: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    source_provenance: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    identity = report.get("attempt_lock")
    if not isinstance(identity, dict) or set(identity) != {
        "path",
        "sha256",
        "content",
        "left_in_place_for_commit",
    }:
        raise BinaryRegimeLongrunAuditError(
            "Sealed audit report lacks exact persistent attempt-lock evidence"
        )
    content = identity.get("content")
    if (
        not isinstance(content, dict)
        or set(content) != set(ATTEMPT_LOCK_CONTENT_KEYS)
        or content.get("schema_version") != ATTEMPT_LOCK_SCHEMA_VERSION
        or content.get("contract_version") != CONTRACT_VERSION
        or content.get("run_id") != AUDIT_RUN_ID
        or content.get("one_run_no_retry") is not True
        or content.get("persistent_after_success_or_failure") is not True
        or content.get("created_after_through_2023_checkpoint_replay") is not True
        or content.get("created_before_raw_blob_or_2024_value_read") is not True
        or content.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256
        or re.fullmatch(r"[0-9a-f]{40}", str(content.get("git_commit"))) is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(content.get("through_2023_bounded_result_sha256")),
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(content.get("pre_2024_gate_status_sha256")),
        )
        is None
        or identity.get("left_in_place_for_commit") is not True
        or identity.get("path")
        != (AUDIT_OUTPUT_PATH / ATTEMPT_LOCK_FILENAME).as_posix()
    ):
        raise BinaryRegimeLongrunAuditError(
            "Persistent attempt-lock contract is malformed"
        )
    expected_bytes = _pretty_json_bytes(content)
    expected_hash = _sha256(expected_bytes)
    supplied = payloads.get(ATTEMPT_LOCK_FILENAME)
    try:
        prefix_provenance = json.loads(
            payloads["prefix_preflight_provenance.json"].decode("utf-8")
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryRegimeLongrunAuditError(
            "Prefix-preflight provenance payload is unreadable"
        ) from exc
    lock_path = output_dir.resolve() / ATTEMPT_LOCK_FILENAME
    try:
        persisted = lock_path.read_bytes()
    except OSError as exc:
        raise BinaryRegimeLongrunAuditError(
            "Persistent attempt lock is missing at seal time"
        ) from exc
    if (
        identity.get("sha256") != expected_hash
        or supplied != expected_bytes
        or persisted != expected_bytes
        or content.get("git_commit") != git_identity.get("commit")
        or content.get("parent_manifest_sha256")
        != parent_manifest.get("manifest_sha256")
        or content.get("through_2023_bounded_result_sha256")
        != parent_manifest.get("bounded_result_sha256")
        or not isinstance(prefix_provenance, dict)
        or content.get("through_2023_bounded_result_sha256")
        != prefix_provenance.get("bounded_result_sha256")
        or content.get("pre_2024_gate_status_sha256")
        != _sha256(_canonical_json_bytes(report.get("pre_2024_gate_status")))
        or source_provenance.get("authorization_preflight", {}).get(
            "persistent_attempt_lock_sha256"
        )
        != expected_hash
    ):
        raise BinaryRegimeLongrunAuditError(
            "Persistent attempt lock, report, and sealed payload do not match"
        )
    return dict(identity)


def _require_seal_contract(
    *,
    output_dir: Path,
    run_id: str,
    report: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    source_provenance: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if run_id != AUDIT_RUN_ID or report.get("run_id") != run_id:
        raise BinaryRegimeLongrunAuditError(
            "Sealed audit run id is not the one frozen identity"
        )
    required_payloads = _required_audit_payload_names()
    if set(payloads) != set(required_payloads) or any(
        not isinstance(value, bytes) or not value
        for value in payloads.values()
    ):
        raise BinaryRegimeLongrunAuditError(
            "Sealed audit payload inventory is incomplete, extra, or empty"
        )
    _require_consistent_report_statuses(report)
    tracked_input = source_provenance.get("tracked_input")
    authorization = source_provenance.get("authorization_preflight")
    if (
        parent_manifest.get("manifest_sha256") != PARENT_MANIFEST_SHA256
        or source_provenance.get("bounded_result_sha256")
        != FINAL_BOUNDED_RESULT_SHA256
        or source_provenance.get("expected_bounded_result_sha256")
        != FINAL_BOUNDED_RESULT_SHA256
        or source_provenance.get("bounded_rows") != FINAL_ROWS
        or source_provenance.get("bounded_first_date") != "1999-03-10"
        or source_provenance.get("bounded_last_date") != "2026-07-09"
        or source_provenance.get("physical_snapshot_has_later_rows") is not False
        or source_provenance.get("rows_after_bound_returned") is not False
        or not isinstance(tracked_input, dict)
        or tracked_input.get("path") != git_identity.get("final_input_path")
        or tracked_input.get("head_index_local_equal") is not True
        or tracked_input.get("path_status_clean") is not True
        or tracked_input.get("verified_after_attempt_lock") is not True
        or tracked_input.get("head_object_id")
        != git_identity.get("final_input_head_object_id")
        or tracked_input.get("index_object_id")
        != git_identity.get("final_input_index_object_id")
        or tracked_input.get("sha256") != tracked_input.get("head_blob_sha256")
        or tracked_input.get("sha256") != tracked_input.get("index_blob_sha256")
        or tracked_input.get("sha256") != tracked_input.get("local_file_sha256")
        or git_identity.get("pre_lock_worktree_clean_excluding_final_input")
        is not True
        or git_identity.get("pre_lock_final_input_head_index_equal") is not True
        or git_identity.get("final_input_verified_clean_after_attempt_lock")
        is not True
        or not isinstance(authorization, dict)
        or authorization.get("final_input_verified_clean_after_attempt_lock")
        is not True
        or re.fullmatch(r"[0-9a-f]{40}", str(git_identity.get("commit"))) is None
    ):
        raise BinaryRegimeLongrunAuditError(
            "Seal inputs do not match the exact parent, source, or Git identity"
        )
    online_metrics = report["online_metrics_2005_through_2026_07_09"]
    frozen_metrics = report["frozen_2023_metrics_2005_through_2026_07_09"]
    recomputed_comparator = _require_comparator_identity(
        online_metrics,
        frozen_metrics,
        report["online_integrity"],
        report["frozen_2023_integrity"],
    )
    if (
        not _always_long_matches(online_metrics)
        or not _always_long_matches(frozen_metrics)
        or _canonical_json_bytes(report["policy_comparison"])
        != _canonical_json_bytes(_policy_comparison(online_metrics, frozen_metrics))
        or not all(
            _selector_runner._integrity_gates(report["online_integrity"]).values()
        )
        or not all(
            _selector_runner._integrity_gates(
                report["frozen_2023_integrity"]
            ).values()
        )
        or not _all_policy_no_leverage_proofs(online_metrics)
        or not _all_policy_no_leverage_proofs(frozen_metrics)
        or not report["comparator_integrity"]
        or any(value is not True for value in report["comparator_integrity"].values())
        or _canonical_json_bytes(report["comparator_integrity"])
        != _canonical_json_bytes(recomputed_comparator)
    ):
        raise BinaryRegimeLongrunAuditError(
            "Sealed policy metrics, controls, or integrity fields are inconsistent"
        )
    cross_bound_payloads = {
        "authorized_rejected_validation_manifest.json": parent_manifest,
        "input_provenance.json": source_provenance,
        "pre_2024_gate_status.json": report["pre_2024_gate_status"],
        "audit_online_metrics.json": online_metrics,
        "audit_frozen_2023_metrics.json": frozen_metrics,
        "audit_policy_comparison.json": report["policy_comparison"],
        "strict_recent_history_gate_report.json": report[
            "strict_recent_history_gate_report"
        ],
        "post_hoc_long_run_gate_report.json": report[
            "post_hoc_long_run_gate_report"
        ],
        "adaptive_value.json": report["adaptive_value"],
    }
    for filename, expected in cross_bound_payloads.items():
        _require_json_payload_matches(payloads, filename, expected)
    _require_recomputed_gate_reports(report, payloads)
    _require_adaptive_payload_consistency(report, payloads)
    _require_online_artifact_consistency(report, payloads)
    attempt_lock = _require_attempt_lock_evidence(
        output_dir=output_dir,
        report=report,
        payloads=payloads,
        source_provenance=source_provenance,
        git_identity=git_identity,
        parent_manifest=parent_manifest,
    )
    try:
        checkpoint = json.loads(
            payloads["online_checkpoint_through_2026_07_09.json"].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryRegimeLongrunAuditError(
            "Final online continuation checkpoint is unreadable"
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("learning_mode") != CAUSAL_ONLINE_MODE
        or checkpoint.get("checkpoint_cutoff") != "2026-07-09"
        or checkpoint.get("last_observed_session") != "2026-07-09"
        or not isinstance(checkpoint.get("serialized_regime_states"), dict)
        or not isinstance(checkpoint.get("serialized_pending_shadow_lessons"), dict)
        or not isinstance(checkpoint.get("account_trailing_cooldown_context"), list)
    ):
        raise BinaryRegimeLongrunAuditError(
            "Final online continuation checkpoint is incomplete"
        )
    return attempt_lock


def _seal_audit_bundle(
    run_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    before_promote: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Seal via one short private directory and one atomic promotion.

    Files are written directly inside the unpromoted private directory.  A
    second per-file temporary name would add no atomicity to the final bundle
    and can exceed Windows MAX_PATH in an otherwise valid workspace.
    """

    destination = run_dir.resolve()
    if destination.exists():
        raise BinaryRegimeLongrunAuditError(
            f"Artifact directory already exists: {destination}"
        )
    _require_seal_path_feasibility(output_dir=destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f"{SEAL_TEMP_PREFIX}{uuid.uuid4().hex[:SEAL_TEMP_NONCE_CHARS]}.sealing"
    )
    try:
        temporary.mkdir(exist_ok=False)
    except OSError as exc:
        raise BinaryRegimeLongrunAuditError(
            f"Private audit sealing directory could not be created: {temporary}"
        ) from exc

    promoted = False
    try:
        normalized: dict[str, bytes] = {}
        for name, data in payloads.items():
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name in {".", "..", "checksums.json"}
                or not isinstance(data, bytes)
            ):
                raise BinaryRegimeLongrunAuditError(
                    "Audit artifact names must be flat, safe, unique strings and payloads must be bytes"
                )
            normalized[name] = data

        checksums = {
            name: _sha256(data) for name, data in sorted(normalized.items())
        }
        for name, data in sorted(normalized.items()):
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        checksum_bytes = _pretty_json_bytes(checksums)
        checksum_path = temporary / "checksums.json"
        with checksum_path.open("xb") as handle:
            handle.write(checksum_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        for name, expected in checksums.items():
            if _sha256((temporary / name).read_bytes()) != expected:
                raise BinaryRegimeLongrunAuditError(
                    f"Audit artifact checksum mismatch: {name}"
                )
        if checksum_path.read_bytes() != checksum_bytes:
            raise BinaryRegimeLongrunAuditError(
                "Audit checksum inventory readback mismatch"
            )
        if destination.exists():
            raise BinaryRegimeLongrunAuditError(
                f"Artifact directory already exists: {destination}"
            )
        if before_promote is not None:
            before_promote()
        temporary.replace(destination)
        promoted = True
        return checksums
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)


def _stage_bundle(
    *,
    output_dir: Path,
    run_id: str,
    report: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    source_provenance: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    deadline: _Deadline,
) -> dict[str, Any]:
    attempt_lock = _require_seal_contract(
        output_dir=output_dir,
        run_id=run_id,
        report=report,
        payloads=payloads,
        source_provenance=source_provenance,
        git_identity=git_identity,
        parent_manifest=parent_manifest,
    )
    data_payloads = {".gitattributes": b"* -text\n", **dict(payloads)}
    data_payloads["report.json"] = _pretty_json_bytes(report)
    hashes = {name: _sha256(value) for name, value in sorted(data_payloads.items())}
    manifest_payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": "post_hoc_long_run_audit",
        "stage_pass": bool(
            report["fixed_policy_candidate_for_prospective_paper"]
        ),
        "run_id": run_id,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "parent_validation_rejection_remains_final": True,
        "parent_manifest_sha256": parent_manifest["manifest_sha256"],
        "expected_parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "attempt_lock": attempt_lock,
        "post_hoc_long_run_robustness_pass": bool(
            report["post_hoc_long_run_robustness_pass"]
        ),
        "strict_recent_history_pass": bool(report["strict_recent_history_pass"]),
        "fixed_policy_candidate_for_prospective_paper": bool(
            report["fixed_policy_candidate_for_prospective_paper"]
        ),
        "adaptive_status": report["adaptive_value"]["adaptive_status"],
        "online_learning_historical_value_demonstrated": bool(
            report["online_learning_historical_value_demonstrated"]
        ),
        "online_learning_candidate_for_paper": bool(
            report["online_learning_candidate_for_paper"]
        ),
        "source_path": source_provenance["source_path"],
        "bounded_result_sha256": source_provenance["bounded_result_sha256"],
        "source_provenance": dict(source_provenance),
        "git_identity": dict(git_identity),
        "payload_sha256": hashes,
        "execution": {
            "asset": "AAPL",
            "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
            "continuous_account_start": "2005-01-01",
            "account_resets_after_inception": 0,
            "maximum_target_exposure": 1.0,
            "shorting": False,
            "leverage": False,
            "borrowing": False,
            "negative_cash": False,
            "cash_interest": False,
            "margin_interest": False,
            "network_access": False,
            "news_calls": 0,
            "llm_calls": 0,
            "api_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "cost_scenarios_bps_per_changing_leg": [5.0, 10.0],
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "promotion_requires_strict_pre_rename_deadline": True,
            "ytd_label": "through 2026-07-09",
        },
    }
    manifest = _manifest(manifest_payload)
    data_payloads["stage_manifest.json"] = _pretty_json_bytes(manifest)
    run_dir = output_dir.resolve() / run_id
    checksums = _seal_audit_bundle(
        run_dir,
        data_payloads,
        before_promote=lambda: deadline.check("before long-run artifact promotion"),
    )
    return {
        "stage": "post_hoc_long_run_audit",
        "stage_pass": bool(manifest_payload["stage_pass"]),
        "run_id": run_id,
        "artifact_dir": str(run_dir),
        "stage_manifest": str(run_dir / "stage_manifest.json"),
        "manifest_sha256": manifest["manifest_sha256"],
        "checksums": checksums,
    }


def run_longrun_audit(
    *,
    repo_root: Path,
    price_artifact: Path,
    validation_manifest: Path,
    output_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the one authorized post-hoc audit and seal its complete evidence."""

    deadline = _Deadline(clock)
    # This rejected-parent verification deliberately precedes all final-input
    # access and even global worktree enumeration.
    parent = _validated_rejected_validation_parent(
        repo_root=repo_root, path=validation_manifest
    )
    resolved_run_id = _require_frozen_run_destination(
        repo_root=repo_root, output_dir=output_dir, run_id=run_id
    )
    git_identity = _clean_git_identity(repo_root, price_artifact)
    _require_dependency_continuity(git_identity, parent)
    deadline.check("rejected-parent authorization")

    # The loader may inspect the physical maximum date, but returns only the
    # through-2023 prefix at this point.
    parent_prefix, prefix_provenance = load_bounded_prices(
        price_artifact,
        end=VALIDATION_END,
        required_last_session=REQUIRED_VALIDATION_LAST_SESSION,
    )
    _selector_runner._require_source_continuity(
        parent_prefix,
        prefix_provenance,
        parent,
        parent_end=VALIDATION_END,
    )
    parent_prefix_forecast = build_binary_regime_union_selector_forecast(
        parent_prefix, learning_mode=CAUSAL_ONLINE_MODE
    )
    _selector_runner._require_checkpoint_continuity(
        parent_prefix,
        parent_prefix_forecast,
        cutoff=VALIDATION_END,
        parent_manifest_path=validation_manifest,
        checkpoint_filename="validation_checkpoint_through_2023.json",
    )
    deadline.check("through-2023 prefix and checkpoint replay")

    pre_periods = _annual_periods(2023)
    (
        pre_metrics,
        pre_ledgers,
        pre_episodes,
        pre_benefits,
        pre_integrity,
    ) = _selector_runner._evaluate_policy_set(
        parent_prefix,
        parent_prefix_forecast,
        periods=pre_periods,
        administrative_start=ACCOUNT_START,
    )
    pre_gate_report = apply_post_hoc_long_run_gates(
        pre_metrics,
        pre_episodes,
        pre_integrity,
        expected_period_names=tuple(period.name for period in pre_periods),
    )
    pre_2024_gate_status = _pre_2024_disclosure(pre_gate_report)
    deadline.check("pre-2024 gate mirror")

    # Permanently consume the one audit attempt immediately before the first
    # raw/full input read. The lock is deliberately never removed, including
    # on any later failure.
    attempt_lock, attempt_lock_bytes = _create_persistent_attempt_lock(
        repo_root=repo_root,
        output_dir=output_dir,
        parent=parent,
        git_identity=git_identity,
        prefix_provenance=prefix_provenance,
        pre_2024_gate_status=pre_2024_gate_status,
    )
    # Only after exact prefix/checkpoint replay and durable attempt locking may
    # code read the full tracked blob or return a DataFrame with 2024+ values.
    input_identity = _tracked_input_identity(repo_root, price_artifact)
    if (
        git_identity.get("final_input_path") != input_identity.get("path")
        or git_identity.get("final_input_head_object_id")
        != input_identity.get("head_object_id")
        or git_identity.get("final_input_index_object_id")
        != input_identity.get("index_object_id")
    ):
        raise BinaryRegimeLongrunAuditError(
            "Pre-lock price identity differs from the post-lock verified input"
        )
    git_identity["final_input_verified_clean_after_attempt_lock"] = True
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=AUDIT_END,
        required_last_session=AUDIT_END,
    )
    _selector_runner._require_exact_physical_stage_bound(
        provenance, stage="Long-run audit"
    )
    provenance["tracked_input"] = input_identity
    _selector_runner._require_source_continuity(
        frame, provenance, parent, parent_end=VALIDATION_END
    )
    _require_final_input_identity(frame, provenance)
    provenance["authorization_preflight"] = {
        "rejected_parent_verified_before_input_access": True,
        "through_2023_prefix_verified_before_2024_value_return": True,
        "through_2023_checkpoint_verified_before_2024_value_return": True,
        "persistent_attempt_lock_created_before_raw_or_2024_value_read": True,
        "persistent_attempt_lock_sha256": attempt_lock["sha256"],
        "final_input_verified_clean_after_attempt_lock": True,
        "parent_prefix_bounded_result_sha256": prefix_provenance[
            "bounded_result_sha256"
        ],
        "parent_checkpoint_manifest_sha256": parent["manifest_sha256"],
    }
    deadline.check("authorized final-row load")

    online = build_binary_regime_union_selector_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    frozen = build_binary_regime_union_selector_forecast(
        frame,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=VALIDATION_END,
    )
    periods = _annual_periods(2025)
    (
        online_metrics,
        online_ledgers,
        online_episodes,
        online_benefits,
        online_integrity,
    ) = _selector_runner._evaluate_policy_set(
        frame,
        online,
        periods=periods,
        administrative_start=ACCOUNT_START,
    )
    (
        frozen_metrics,
        frozen_ledgers,
        frozen_episodes,
        frozen_benefits,
        frozen_integrity,
    ) = _selector_runner._evaluate_policy_set(
        frame,
        frozen,
        periods=periods,
        administrative_start=ACCOUNT_START,
    )
    comparator_integrity = _require_comparator_identity(
        online_metrics,
        frozen_metrics,
        online_integrity,
        frozen_integrity,
    )
    strict = apply_strict_recent_history_gates(online_metrics, online_integrity)
    long_run = apply_post_hoc_long_run_gates(
        online_metrics, online_episodes, online_integrity
    )
    (
        adaptive,
        differing_episodes,
        threshold_crossings,
        divergence_runs,
    ) = _adaptive_value_analysis(
        frame, online, frozen, online_metrics, frozen_metrics
    )
    post_hoc_pass = bool(long_run["post_hoc_long_run_robustness_pass"])
    strict_pass = bool(strict["strict_recent_history_pass"])
    fixed_candidate = post_hoc_pass and strict_pass
    learning_demonstrated = adaptive["adaptive_status"] == "exercised_positive"
    online_candidate = fixed_candidate and learning_demonstrated
    deadline.check("long-run replay, reconciliation, and gates")

    checkpoint = _selector_runner._checkpoint_from_forecast(
        frame,
        online,
        cutoff=AUDIT_END,
        learning_mode=CAUSAL_ONLINE_MODE,
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "post_hoc_long_run_audit",
        "run_id": resolved_run_id,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "parent_validation_rejection_remains_final": True,
        "parent_validation_failures": list(EXPECTED_VALIDATION_FAILURES),
        "physical_data_end": "2026-07-09",
        "ytd_label": "2026 YTD through 2026-07-09",
        "continuous_account_start": "2005-01-01",
        "continuous_account_reset_count_after_inception": 0,
        "attempt_lock": attempt_lock,
        "sentiment_inputs": SENTIMENT_DISCLOSURE,
        "pre_2024_gate_status": pre_2024_gate_status,
        "online_metrics_2005_through_2026_07_09": online_metrics,
        "frozen_2023_metrics_2005_through_2026_07_09": frozen_metrics,
        "policy_comparison": _policy_comparison(online_metrics, frozen_metrics),
        "online_integrity": online_integrity,
        "frozen_2023_integrity": frozen_integrity,
        "comparator_integrity": comparator_integrity,
        "strict_recent_history_gate_report": strict,
        "post_hoc_long_run_gate_report": long_run,
        "strict_recent_history_pass": strict_pass,
        "post_hoc_long_run_robustness_pass": post_hoc_pass,
        "fixed_policy_candidate_for_prospective_paper": fixed_candidate,
        "adaptive_value": adaptive,
        "online_learning_historical_value_demonstrated": learning_demonstrated,
        "online_learning_candidate_for_paper": online_candidate,
        "online_arm_role_for_prospective_paper": (
            "learning_candidate" if online_candidate else "shadow_challenger"
        ),
        "historical_results_authorize_real_capital": False,
        "runtime": _runtime_report(deadline),
    }

    payloads: dict[str, bytes] = {
        ATTEMPT_LOCK_FILENAME: attempt_lock_bytes,
        "authorized_rejected_validation_manifest.json": _pretty_json_bytes(parent),
        "input_provenance.json": _pretty_json_bytes(provenance),
        "prefix_preflight_provenance.json": _pretty_json_bytes(prefix_provenance),
        "pre_2024_gate_status.json": _pretty_json_bytes(pre_2024_gate_status),
        "pre_2024_metrics.json": _pretty_json_bytes(pre_metrics),
        "audit_prices_through_2026_07_09.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "audit_online_forecast_2024_through_2026_07_09.csv": _frame_csv_bytes(
            _selector_runner._flatten_forecast(online.loc[AUDIT_START:AUDIT_END])
        ),
        "audit_frozen_2023_forecast_2024_through_2026_07_09.csv": _frame_csv_bytes(
            _selector_runner._flatten_forecast(frozen.loc[AUDIT_START:AUDIT_END])
        ),
        "audit_online_metrics.json": _pretty_json_bytes(online_metrics),
        "audit_frozen_2023_metrics.json": _pretty_json_bytes(frozen_metrics),
        "audit_policy_comparison.json": _pretty_json_bytes(
            _policy_comparison(online_metrics, frozen_metrics)
        ),
        "strict_recent_history_gate_report.json": _pretty_json_bytes(strict),
        "post_hoc_long_run_gate_report.json": _pretty_json_bytes(long_run),
        "adaptive_value.json": _pretty_json_bytes(adaptive),
        "adaptive_threshold_crossings.csv": _frame_csv_bytes(threshold_crossings),
        "adaptive_divergence_runs.csv": _frame_csv_bytes(divergence_runs),
        "online_checkpoint_through_2026_07_09.json": _pretty_json_bytes(checkpoint),
        **_selector_runner._ledger_payloads(
            pre_ledgers, prefix="pre_2024_online"
        ),
        **_selector_runner._episode_payloads(
            pre_episodes, prefix="pre_2024_online"
        ),
        **_selector_runner._benefit_payloads(
            pre_benefits, prefix="pre_2024_online"
        ),
        **_selector_runner._ledger_payloads(
            online_ledgers, prefix="longrun_online"
        ),
        **_selector_runner._episode_payloads(
            online_episodes, prefix="longrun_online"
        ),
        **_selector_runner._benefit_payloads(
            online_benefits, prefix="longrun_online"
        ),
        **_selector_runner._ledger_payloads(
            frozen_ledgers, prefix="longrun_frozen_2023"
        ),
        **_selector_runner._episode_payloads(
            frozen_episodes, prefix="longrun_frozen_2023"
        ),
        **_selector_runner._benefit_payloads(
            frozen_benefits, prefix="longrun_frozen_2023"
        ),
    }
    for cost_name, difference in differing_episodes.items():
        payloads[f"adaptive_{cost_name}_xor_differing_episodes.csv"] = (
            _frame_csv_bytes(difference)
        )
    deadline.check("before long-run seal")
    result = _stage_bundle(
        output_dir=output_dir,
        run_id=resolved_run_id,
        report=report,
        payloads=payloads,
        source_provenance=provenance,
        git_identity=git_identity,
        parent_manifest=parent,
        deadline=deadline,
    )
    return {
        **result,
        "strict_recent_history_pass": strict_pass,
        "post_hoc_long_run_robustness_pass": post_hoc_pass,
        "fixed_policy_candidate_for_prospective_paper": fixed_candidate,
        "adaptive_status": adaptive["adaptive_status"],
        "online_learning_historical_value_demonstrated": learning_demonstrated,
        "online_learning_candidate_for_paper": online_candidate,
        "runtime_seconds": deadline.elapsed(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit",))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--price-artifact", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest", type=Path, default=PARENT_MANIFEST_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=AUDIT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    price_artifact = (
        args.price_artifact
        if args.price_artifact.is_absolute()
        else root / args.price_artifact
    )
    validation_manifest = (
        args.validation_manifest
        if args.validation_manifest.is_absolute()
        else root / args.validation_manifest
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    result = run_longrun_audit(
        repo_root=root,
        price_artifact=price_artifact,
        validation_manifest=validation_manifest,
        output_dir=output_dir,
        run_id=AUDIT_RUN_ID,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BinaryRegimeLongrunAuditError",
    "apply_post_hoc_long_run_gates",
    "apply_strict_recent_history_gates",
    "load_bounded_prices",
    "main",
    "run_longrun_audit",
    "validate_rejected_validation_parent",
]
