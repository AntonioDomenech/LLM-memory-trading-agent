"""Create the first immutable prospective paper decision for the frozen selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .binary_regime_union_selector import (
    FROZEN_CUTOFF_MODE,
    build_binary_regime_union_selector_forecast,
)
from .chronological_exhaustion_expert import canonicalize_one_session_signals
from .unleveraged_aapl import canonical_context_frame, download_context_frame


CONTRACT_VERSION = "aapl-binary-regime-prospective-paper-v1"
BRANCH = "codex/aapl-binary-regime-prospective-paper"
FROZEN_CUTOFF = pd.Timestamp("2023-12-31")
ACCOUNT_START = pd.Timestamp("2005-01-01")
AUDITED_END = pd.Timestamp("2026-07-09")
FIRST_AS_OF = pd.Timestamp("2026-07-17")
FIRST_DECISION_DEADLINE_UTC = datetime(
    2026, 7, 20, 13, 30, tzinfo=timezone.utc
)
AUDITED_INPUT_RELATIVE = Path(
    "e/binary_regime_longrun_audit_v1/authorized_inputs/"
    "aapl_spy_qqq_through_2026_ytd.csv"
)
AUDITED_INPUT_BYTES = 720_140
AUDITED_INPUT_SHA256 = (
    "cf384e8218f97d4c6e7ca89860877ef5edb67e938af66886bdf81912bd35479b"
)
AUDITED_ROWS = 6_875
AUDITED_ACTION_STREAM_SHA256 = (
    "f77c68462ced8158bca6bf5a0aec95b4161bacd811075048e7918ba1de4d15ed"
)
EXPECTED_POST_AUDIT_SESSIONS = pd.DatetimeIndex(
    pd.to_datetime(
        [
            "2026-07-10",
            "2026-07-13",
            "2026-07-14",
            "2026-07-15",
            "2026-07-16",
            "2026-07-17",
        ]
    )
)
CHECKPOINT_RELATIVE = Path(
    "e/binary_regime_union_selector_v1/"
    "binary-regime-union-selector-validation-v1/"
    "validation_checkpoint_through_2023.json"
)
INITIAL_EQUITY = 1_000.0
CANONICAL_OUTPUT_RELATIVE = Path(
    "e/binary_regime_prospective_paper/decisions"
)
REQUIRED_COLUMNS = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "aapl_adj_open",
    "spy_adj_close",
    "qqq_adj_close",
)
RAW_PRICE_ABS_TOLERANCE = 1e-12
ADJUSTED_PRICE_REL_TOLERANCE = 1e-4
FROZEN_STATE_NUMERIC_DRIFT_TOLERANCE = 1e-4


class ProspectivePaperError(RuntimeError):
    """The prospective decision could not be created without ambiguity."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.reset_index(names="date").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def _git_authority(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ProspectivePaperError("Paper decision requires a clean worktree")
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise ProspectivePaperError("Paper decision is on the wrong branch")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    if head != upstream:
        raise ProspectivePaperError("Paper implementation is not pushed")
    paths = (
        "docs/aapl_binary_regime_prospective_paper.md",
        "agent_benchmark/binary_regime_prospective_paper.py",
        "tests/test_binary_regime_prospective_paper.py",
        "agent_benchmark/binary_regime_union_selector.py",
        "agent_benchmark/chronological_exhaustion_expert.py",
        "agent_benchmark/unleveraged_aapl.py",
        AUDITED_INPUT_RELATIVE.as_posix(),
        CHECKPOINT_RELATIVE.as_posix(),
    )
    identities: dict[str, Any] = {}
    for name in paths:
        path = root / name
        git("ls-files", "--error-unmatch", "--", name)
        identities[name] = {
            "git_blob": _git_blob_oid(path),
            "literal_sha256": _file_sha256(path),
        }
    return {
        "branch": branch,
        "commit": head,
        "upstream_commit": upstream,
        "clean_before_download": True,
        "files": identities,
    }


def _load_audited_input(root: Path) -> pd.DataFrame:
    path = root / AUDITED_INPUT_RELATIVE
    if (
        not path.is_file()
        or path.stat().st_size != AUDITED_INPUT_BYTES
        or _file_sha256(path) != AUDITED_INPUT_SHA256
    ):
        raise ProspectivePaperError("Preserved audited input identity changed")
    frame = canonical_context_frame(pd.read_csv(path))
    if (
        len(frame) != AUDITED_ROWS
        or frame.index.max() != AUDITED_END
        or frame.index.has_duplicates
    ):
        raise ProspectivePaperError("Preserved audited input coverage changed")
    return frame


def _account_action_stream(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = canonical_context_frame(frame)
    forecast = build_binary_regime_union_selector_forecast(
        data,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=FROZEN_CUTOFF,
    )
    raw = forecast["union_candidate_signal"].astype(bool).copy()
    raw.loc[raw.index < ACCOUNT_START] = False
    account_union = canonicalize_one_session_signals(raw)
    selector_cash = account_union & forecast["selector_cash_prediction"].astype(bool)
    stream = pd.DataFrame(
        {
            "union_candidate_signal": forecast[
                "union_candidate_signal"
            ].astype(bool),
            "account_union_cash_signal": account_union.astype(bool),
            "risk_on": forecast["risk_on"].astype(bool),
            "selector_cash_prediction": forecast[
                "selector_cash_prediction"
            ].astype(bool),
            "selector_cash_signal": selector_cash.astype(bool),
            "target_exposure": np.where(selector_cash, 0.0, 1.0),
        },
        index=data.index,
    )
    return forecast, stream


def _sealed_action_stream_sha256(stream: pd.DataFrame) -> str:
    """Reproduce the exact action fingerprint stored by the sealed audit."""

    action_frame = pd.DataFrame(
        {
            "decision_date": stream.index,
            "selector_target": stream["target_exposure"].to_numpy(dtype=float),
            "union_target": np.where(
                stream["account_union_cash_signal"].to_numpy(dtype=bool),
                0.0,
                1.0,
            ),
        }
    )
    payload = action_frame.to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    return _sha256(payload)


def _states_match_checkpoint(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    scalar_keys = (
        "schema_version",
        "regime_order",
        "regime_feature_order",
        "lesson_discount",
        "minimum_effective_lessons",
        "positive_mean_threshold",
        "negative_mean_threshold",
        "structural_default_cash",
    )
    if any(observed.get(key) != expected.get(key) for key in scalar_keys):
        return False
    for regime in ("risk_on", "not_risk_on"):
        left = observed["states"][regime]
        right = expected["states"][regime]
        for key in ("n_raw", "cash_selected"):
            if left[key] != right[key]:
                return False
        for key in (
            "n_eff",
            "weighted_label_sum",
            "weighted_squared_label_sum",
        ):
            if not math.isclose(
                float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=5e-12
            ):
                return False
    return True


def _frozen_state_compatibility(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the same frozen decisions while recording tiny label drift."""

    scalar_keys = (
        "schema_version",
        "regime_order",
        "regime_feature_order",
        "lesson_discount",
        "minimum_effective_lessons",
        "positive_mean_threshold",
        "negative_mean_threshold",
        "structural_default_cash",
    )
    if any(observed.get(key) != expected.get(key) for key in scalar_keys):
        raise ProspectivePaperError("Frozen selector contract changed")
    diagnostics: dict[str, Any] = {
        "numeric_drift_tolerance": FROZEN_STATE_NUMERIC_DRIFT_TOLERANCE,
        "regimes": {},
    }
    for regime in ("risk_on", "not_risk_on"):
        left = observed["states"][regime]
        right = expected["states"][regime]
        if left["n_raw"] != right["n_raw"]:
            raise ProspectivePaperError("Frozen selector lesson count changed")
        if left["cash_selected"] != right["cash_selected"]:
            raise ProspectivePaperError("Frozen selector LONG/CASH latch changed")
        differences = {
            key: abs(float(left[key]) - float(right[key]))
            for key in (
                "n_eff",
                "weighted_label_sum",
                "weighted_squared_label_sum",
            )
        }
        if max(differences.values()) > FROZEN_STATE_NUMERIC_DRIFT_TOLERANCE:
            raise ProspectivePaperError(
                "Frozen selector numeric state changed too much"
            )
        diagnostics["regimes"][regime] = {
            "n_raw_exact": True,
            "cash_selected_exact": True,
            "numeric_absolute_differences": differences,
        }
    diagnostics["decision_latches_exact"] = True
    return diagnostics


def _prefix_price_compatibility(
    *, fresh: pd.DataFrame, audited: pd.DataFrame
) -> dict[str, Any]:
    """Allow harmless vendor rounding, never a trading-scale price change."""

    raw_columns = ("aapl_open", "aapl_close")
    adjusted_columns = (
        "aapl_adj_close",
        "aapl_adj_open",
        "spy_adj_close",
        "qqq_adj_close",
    )
    diagnostics: dict[str, Any] = {
        "raw_absolute_tolerance": RAW_PRICE_ABS_TOLERANCE,
        "adjusted_relative_tolerance": ADJUSTED_PRICE_REL_TOLERANCE,
        "columns": {},
    }
    all_exact = True
    for column in REQUIRED_COLUMNS:
        old = audited[column].to_numpy(dtype=float)
        new = fresh[column].to_numpy(dtype=float)
        if not bool(np.isfinite(old).all() and np.isfinite(new).all()):
            raise ProspectivePaperError("Fresh Yahoo prefix contains non-finite prices")
        difference = np.abs(new - old)
        relative = difference / np.maximum(np.abs(old), 1e-300)
        exact = bool(np.array_equal(new, old))
        all_exact = all_exact and exact
        diagnostics["columns"][column] = {
            "exact": exact,
            "changed_values": int(np.count_nonzero(difference)),
            "maximum_absolute_difference": float(np.max(difference)),
            "maximum_relative_difference": float(np.max(relative)),
        }
    maximum_raw_absolute = max(
        diagnostics["columns"][column]["maximum_absolute_difference"]
        for column in raw_columns
    )
    maximum_adjusted_relative = max(
        diagnostics["columns"][column]["maximum_relative_difference"]
        for column in adjusted_columns
    )
    if maximum_raw_absolute > RAW_PRICE_ABS_TOLERANCE:
        raise ProspectivePaperError(
            "Fresh Yahoo raw AAPL history changed beyond machine rounding"
        )
    if maximum_adjusted_relative > ADJUSTED_PRICE_REL_TOLERANCE:
        raise ProspectivePaperError(
            "Fresh Yahoo adjusted history changed at trading scale"
        )
    diagnostics["all_values_exact"] = all_exact
    diagnostics["accepted_as_numerical_vendor_revision"] = not all_exact
    return diagnostics


def verify_fresh_history(
    *, root: Path, fresh: pd.DataFrame
) -> dict[str, Any]:
    audited = _load_audited_input(root)
    data = canonical_context_frame(fresh)
    if data.index.max() != FIRST_AS_OF or bool((data.index > FIRST_AS_OF).any()):
        raise ProspectivePaperError("Fresh snapshot does not end at the as-of close")
    observed_post_audit = data.index[data.index > AUDITED_END]
    if not observed_post_audit.equals(EXPECTED_POST_AUDIT_SESSIONS):
        raise ProspectivePaperError(
            "Fresh snapshot is missing or adds a post-audit market session"
        )
    prefix = data.loc[data.index <= AUDITED_END]
    if not prefix.index.equals(audited.index):
        raise ProspectivePaperError("Fresh Yahoo prefix session dates changed")
    price_compatibility = _prefix_price_compatibility(
        fresh=prefix,
        audited=audited,
    )

    audited_forecast, audited_stream = _account_action_stream(audited)
    fresh_forecast, fresh_stream = _account_action_stream(data)
    audited_action_hash = _sealed_action_stream_sha256(audited_stream)
    if audited_action_hash != AUDITED_ACTION_STREAM_SHA256:
        raise ProspectivePaperError(
            "Frozen policy no longer reproduces the sealed audit action stream"
        )
    fresh_prefix_stream = fresh_stream.loc[fresh_stream.index <= AUDITED_END]
    if not fresh_prefix_stream.equals(audited_stream):
        raise ProspectivePaperError("Frozen action stream changed through 2026-07-09")
    if (
        _sealed_action_stream_sha256(fresh_prefix_stream)
        != AUDITED_ACTION_STREAM_SHA256
    ):
        raise ProspectivePaperError(
            "Fresh history changed the sealed action fingerprint"
        )
    checkpoint = json.loads((root / CHECKPOINT_RELATIVE).read_text(encoding="utf-8"))
    expected_states = checkpoint["serialized_regime_states"]
    if not _states_match_checkpoint(
        audited_forecast.attrs["final_states"], expected_states
    ):
        raise ProspectivePaperError("Preserved frozen-through-2023 state changed")
    state_compatibility = _frozen_state_compatibility(
        fresh_forecast.attrs["final_states"], expected_states
    )
    return {
        "audited_rows": int(len(audited)),
        "fresh_rows": int(len(data)),
        "fresh_rows_after_audit": int(len(data) - len(audited)),
        "exact_prefix_prices": price_compatibility["all_values_exact"],
        "prefix_price_compatibility": price_compatibility,
        "exact_prefix_action_stream": True,
        "preserved_frozen_checkpoint_state_exact": True,
        "fresh_frozen_state_compatibility": state_compatibility,
        "audited_action_stream_sha256": AUDITED_ACTION_STREAM_SHA256,
        "audited_final_state": audited_forecast.attrs["final_states"],
    }


def build_decision(
    frame: pd.DataFrame, *, created_at: datetime
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = canonical_context_frame(frame)
    if data.index.max() != FIRST_AS_OF:
        raise ProspectivePaperError("Decision frame has the wrong as-of close")
    if created_at.tzinfo is None:
        raise ProspectivePaperError("Decision creation time must be timezone-aware")
    created_utc = created_at.astimezone(timezone.utc)
    if created_utc >= FIRST_DECISION_DEADLINE_UTC:
        raise ProspectivePaperError("The first prospective fill deadline has passed")
    forecast, stream = _account_action_stream(data)
    row = stream.iloc[-1]
    detail = forecast.iloc[-1]
    union = bool(row["account_union_cash_signal"])
    cash_prediction = bool(row["selector_cash_prediction"])
    cash = bool(row["selector_cash_signal"])
    if cash:
        action = "SELL_ALL_AAPL_TO_CASH_AT_NEXT_ACTUAL_OPEN"
        reason = "accepted union signal in frozen not-risk-on CASH regime"
    elif union:
        action = "HOLD_AAPL"
        reason = "accepted union signal vetoed by frozen risk-on LONG regime"
    else:
        action = "HOLD_AAPL"
        reason = "no accepted account-union cash signal"
    adjusted_close = float(data.iloc[-1]["aapl_adj_close"])
    starting_units = INITIAL_EQUITY / adjusted_close
    decision = {
        "schema_version": "binary-regime-prospective-paper-decision-v1",
        "contract_version": CONTRACT_VERSION,
        "evidence_classification": "prospective_paper_decision_before_outcome",
        "created_at_utc": created_utc.isoformat().replace("+00:00", "Z"),
        "as_of_close": FIRST_AS_OF.date().isoformat(),
        "frozen_learning_cutoff": FROZEN_CUTOFF.date().isoformat(),
        "policy": "frozen_2023_binary_regime_union_selector",
        "promotion_basis": (
            "current_goal_long_term_success_post_hoc_override_of_the_older_"
            "strict_recent_gate"
        ),
        "original_sealed_audit_fixed_policy_candidate": False,
        "original_sealed_audit_strict_recent_history_pass": False,
        "sealed_action_stream_sha256": AUDITED_ACTION_STREAM_SHA256,
        "account_union_cash_signal": union,
        "union_candidate_signal": bool(row["union_candidate_signal"]),
        "risk_on": bool(row["risk_on"]),
        "spy_return_20": float(detail["spy_return_20"]),
        "qqq_return_20": float(detail["qqq_return_20"]),
        "selector_cash_prediction": cash_prediction,
        "selector_cash_signal": cash,
        "target_exposure_next_open": 0.0 if cash else 1.0,
        "action": action,
        "reason": reason,
        "fill_role": "next_actual_AAPL_adjusted_open_after_as_of_close",
        "exit_role_if_cash": "following_actual_AAPL_adjusted_open",
        "fill_price": None,
        "exit_price": None,
        "outcome_known": False,
        "outcome": None,
        "cost_scenarios_bps_per_changing_leg": [5.0, 10.0],
        "broker_calls": 0,
        "real_money_actions": 0,
    }
    state = {
        "schema_version": "binary-regime-prospective-paper-state-v1",
        "created_at_utc": decision["created_at_utc"],
        "as_of_close": decision["as_of_close"],
        "initial_equity": INITIAL_EQUITY,
        "price_basis": "AAPL adjusted prices with snapshot-to-snapshot rebasing",
        "adjustment_rebase_formula": (
            "new_units = old_units * old_anchor_adjusted_close / "
            "new_snapshot_anchor_adjusted_close"
        ),
        "common_starting_cash": 0.0,
        "common_starting_adjusted_units": starting_units,
        "cost_ledgers": {
            f"{int(cost)}bps": {
                "cost_bps_per_changing_leg": cost,
                "strategy_cash": 0.0,
                "strategy_adjusted_units": starting_units,
                "benchmark_cash": 0.0,
                "benchmark_adjusted_units": starting_units,
                "strategy_equity": INITIAL_EQUITY,
                "benchmark_equity": INITIAL_EQUITY,
                "adjustment_anchor_date": decision["as_of_close"],
                "adjustment_anchor_adjusted_close": adjusted_close,
                "changing_legs": 0,
                "estimated_costs": 0.0,
            }
            for cost in (5.0, 10.0)
        },
        "strategy_target_for_next_open": decision["target_exposure_next_open"],
        "benchmark_target_for_next_open": 1.0,
        "no_pre_paper_return_scored": True,
        "pending_decision_as_of": decision["as_of_close"],
        "completed_cash_episodes": 0,
        "real_money": False,
    }
    return decision, state


def settle_decision_outcome(
    *,
    decision: Mapping[str, Any],
    state: Mapping[str, Any],
    observed: pd.DataFrame,
    cost_bps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the frozen next-open ledger without changing the saved decision."""

    if cost_bps not in (5.0, 10.0):
        raise ProspectivePaperError("Only the preregistered 5/10 bps costs are valid")
    data = canonical_context_frame(observed)
    as_of = pd.Timestamp(str(decision["as_of_close"]))
    if as_of not in data.index:
        raise ProspectivePaperError("Outcome snapshot lacks the saved as-of session")
    target = float(decision["target_exposure_next_open"])
    if target not in (0.0, 1.0):
        raise ProspectivePaperError("Saved target is not exactly AAPL or cash")
    future_sessions = data.index[data.index > as_of]
    required = 2 if target == 0.0 else 1
    if len(future_sessions) < required:
        raise ProspectivePaperError("The saved decision outcome has not matured")

    key = f"{int(cost_bps)}bps"
    old_ledger = dict(state["cost_ledgers"][key])
    anchor_date = pd.Timestamp(str(old_ledger["adjustment_anchor_date"]))
    if anchor_date not in data.index:
        raise ProspectivePaperError("Outcome snapshot lacks the adjustment anchor")
    old_anchor = float(old_ledger["adjustment_anchor_adjusted_close"])
    new_anchor = float(data.loc[anchor_date, "aapl_adj_close"])
    if old_anchor <= 0.0 or new_anchor <= 0.0:
        raise ProspectivePaperError("Adjustment anchor price is invalid")
    unit_rebase = old_anchor / new_anchor
    strategy_units = float(old_ledger["strategy_adjusted_units"]) * unit_rebase
    benchmark_units = float(old_ledger["benchmark_adjusted_units"]) * unit_rebase
    strategy_cash = float(old_ledger["strategy_cash"])
    benchmark_cash = float(old_ledger["benchmark_cash"])
    entry_date = pd.Timestamp(future_sessions[0])
    entry_open = float(data.loc[entry_date, "aapl_adj_open"])
    exit_date = entry_date
    exit_open = entry_open
    changing_legs = 0
    costs_paid = 0.0
    rate = cost_bps / 10_000.0

    if target == 0.0:
        if strategy_cash != 0.0 or strategy_units <= 0.0:
            raise ProspectivePaperError(
                "Cash episode must begin fully invested in AAPL"
            )
        gross_sale = strategy_units * entry_open
        sale_cost = gross_sale * rate
        strategy_cash = gross_sale - sale_cost
        strategy_units = 0.0
        changing_legs += 1
        costs_paid += sale_cost

        exit_date = pd.Timestamp(future_sessions[1])
        exit_open = float(data.loc[exit_date, "aapl_adj_open"])
        strategy_units = strategy_cash / (exit_open * (1.0 + rate))
        buy_cost = strategy_units * exit_open * rate
        strategy_cash = 0.0
        changing_legs += 1
        costs_paid += buy_cost

    strategy_equity = strategy_cash + strategy_units * exit_open
    benchmark_equity = benchmark_cash + benchmark_units * exit_open
    if (
        strategy_cash < -1e-10
        or benchmark_cash < -1e-10
        or strategy_units < -1e-12
        or benchmark_units < -1e-12
    ):
        raise ProspectivePaperError("Prospective ledger breached cash-only safety")

    outcome = {
        "schema_version": "binary-regime-prospective-paper-outcome-v1",
        "decision_as_of_close": str(decision["as_of_close"]),
        "cost_bps_per_changing_leg": cost_bps,
        "entry_open_date": entry_date.date().isoformat(),
        "entry_adjusted_open": entry_open,
        "exit_open_date": exit_date.date().isoformat(),
        "exit_adjusted_open": exit_open,
        "target_exposure": target,
        "changing_legs": changing_legs,
        "estimated_costs": costs_paid,
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "excess_return_percentage_points_from_inception": (
            (strategy_equity - benchmark_equity) / INITIAL_EQUITY * 100.0
        ),
        "relative_ending_wealth": strategy_equity / benchmark_equity,
        "outcome_known": True,
        "real_money": False,
    }
    new_ledger = {
        **old_ledger,
        "strategy_cash": strategy_cash,
        "strategy_adjusted_units": strategy_units,
        "benchmark_cash": benchmark_cash,
        "benchmark_adjusted_units": benchmark_units,
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "changing_legs": int(old_ledger["changing_legs"]) + changing_legs,
        "estimated_costs": float(old_ledger["estimated_costs"]) + costs_paid,
        "adjustment_anchor_date": exit_date.date().isoformat(),
        "adjustment_anchor_adjusted_close": float(
            data.loc[exit_date, "aapl_adj_close"]
        ),
    }
    new_state = dict(state)
    new_state["cost_ledgers"] = {
        name: dict(value) for name, value in state["cost_ledgers"].items()
    }
    new_state["cost_ledgers"][key] = new_ledger
    return outcome, new_state


def _publish(
    *,
    destination: Path,
    payloads: Mapping[str, bytes],
    deadline_utc: datetime | None = None,
) -> Path:
    final = destination / FIRST_AS_OF.date().isoformat()
    pending = destination / f".{FIRST_AS_OF.date().isoformat()}.pending"
    if final.exists() or pending.exists():
        raise ProspectivePaperError("First prospective decision path already exists")
    destination.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    try:
        for name, payload in payloads.items():
            (pending / name).write_bytes(payload)
        if deadline_utc is not None and datetime.now(timezone.utc) >= deadline_utc:
            raise ProspectivePaperError("Prospective publication missed its deadline")
        pending.rename(final)
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return final


def create_first_decision(*, root: Path) -> dict[str, Any]:
    repo = root.resolve()
    authority = _git_authority(repo)
    fresh = download_context_frame(
        "1999-03-10", FIRST_AS_OF.date().isoformat()
    )
    continuity = verify_fresh_history(root=repo, fresh=fresh)
    decision_time = datetime.now(timezone.utc)
    decision, state = build_decision(fresh, created_at=decision_time)
    snapshot = _frame_bytes(canonical_context_frame(fresh))
    decision["market_snapshot_sha256"] = _sha256(snapshot)
    decision["market_snapshot_rows"] = int(len(fresh))
    decision["continuity"] = continuity
    decision["implementation_authority"] = authority
    prepublication_time = datetime.now(timezone.utc)
    if prepublication_time >= FIRST_DECISION_DEADLINE_UTC:
        raise ProspectivePaperError(
            "The first prospective publication deadline passed"
        )
    decision["local_prepublication_check_utc"] = (
        prepublication_time.isoformat().replace("+00:00", "Z")
    )
    decision["external_publication_requirement"] = {
        "branch": BRANCH,
        "must_commit_and_push_packet_before_utc": (
            FIRST_DECISION_DEADLINE_UTC.isoformat().replace("+00:00", "Z")
        ),
    }
    decision_bytes = _json_bytes(decision)
    state["decision_sha256"] = _sha256(decision_bytes)
    state_bytes = _json_bytes(state)
    payloads = {
        "market_snapshot.csv": snapshot,
        "decision.json": decision_bytes,
        "initial_paper_state.json": state_bytes,
    }
    manifest = {
        "schema_version": "binary-regime-prospective-paper-manifest-v1",
        "contract_version": CONTRACT_VERSION,
        "as_of_close": decision["as_of_close"],
        "outcome_known": False,
        "files": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in sorted(payloads.items())
        },
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    destination = (repo / CANONICAL_OUTPUT_RELATIVE).resolve()
    final = _publish(
        destination=destination,
        payloads=payloads,
        deadline_utc=FIRST_DECISION_DEADLINE_UTC,
    )
    return {
        "decision_dir": str(final),
        "as_of_close": decision["as_of_close"],
        "created_at_utc": decision["created_at_utc"],
        "action": decision["action"],
        "target_exposure_next_open": decision["target_exposure_next_open"],
        "outcome_known": False,
        "market_snapshot_sha256": decision["market_snapshot_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = create_first_decision(
        root=args.repo_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
