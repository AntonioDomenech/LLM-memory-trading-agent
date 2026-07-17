"""Pure metrics and preregistered gates for the SEC/Gemma online overlay.

The module has no filesystem, network, model, clock, or acquisition authority.
It consumes already-sealed continuous ledgers and decision artifacts, validates
their internal identities, computes exact unrounded metrics, and applies the
literal development, confirmation, and final gates.

All persisted floating-point values use canonical ``float.hex`` text.  Report
windows never restart an account: ledger return factors assigned to destination
opens are sliced from the one continuous account, and the close variant appends
only the report window's final adjusted-close observation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hmac
import math
from statistics import median
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_BLOCKS,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    POSITIVE_EDGE_TOLERANCE,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_learner import (
    validate_online_overlay_feature_row,
    validate_online_overlay_lesson,
    validate_online_overlay_prediction_from_fit,
)
from agent_benchmark.sec_gemma_online_risk_overlay_ledger import (
    GENESIS_PREVIOUS_ROW_SHA256,
    INITIAL_CASH,
    LEDGER_ROW_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    OVERLAY_EPISODE_OBSERVATION_SCHEMA_VERSION,
)
from agent_benchmark.sec_gemma_online_risk_overlay_policy import (
    POLICY_ACTION_ROW_SCHEMA_VERSION,
    replay_nonoverlapping_overlay_policy,
)


METRICS_INPUT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-metrics-input-v1"
)
STAGE_METRICS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-stage-metrics-v1"
)
WINDOW_COMPARISON_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-window-comparison-v1"
)
ACTION_DIFFERENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-action-differences-v1"
)
XOR_ATTRIBUTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-xor-attribution-v1"
)
EPISODE_ATTRIBUTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-episode-attribution-v1"
)
COVERAGE_METRICS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-coverage-metrics-v1"
)
BRIER_METRICS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-brier-metrics-v1"
)
CALENDAR_DIAGNOSTICS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-calendar-diagnostics-v1"
)
GATE_REPORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-gate-report-v1"
)

STAGES: Final[tuple[str, ...]] = (
    "development",
    "confirmation",
    "final",
)
COST_KEYS: Final[tuple[str, ...]] = ("cost_5bps", "cost_10bps")
VALUATION_METHODS: Final[tuple[str, ...]] = (
    "adjusted_open",
    "terminal_adjusted_close",
)
CORE_ACCOUNT_NAMES: Final[tuple[str, ...]] = (
    "semantic",
    "baseline",
    "aapl_buy_and_hold",
    "no_filing_meaning",
    "no_gemma_channel",
)
FROZEN_CONTROL_IDS: Final[dict[str, tuple[str, ...]]] = {
    "development": tuple(item[0] for item in DEVELOPMENT_BLOCKS),
    "confirmation": ("through_2018",),
    "final": ("through_2023",),
}
STAGE_CUTOFFS: Final[dict[str, str]] = {
    "development": "2018-12-31",
    "confirmation": "2023-12-29",
    "final": "2026-07-09",
}
PERFORMANCE_WINDOWS: Final[dict[str, tuple[str, str]]] = {
    "development": ("2005-01-03", "2018-12-31"),
    "confirmation": ("2019-01-01", "2023-12-31"),
    "final": ("2024-01-01", "2026-07-09"),
}
COVERAGE_WINDOWS: Final[dict[str, tuple[str, str]]] = {
    "development": ("2000-01-03", "2018-12-31"),
    "confirmation": ("2019-01-01", "2023-12-31"),
    "final": ("2024-01-01", "2026-07-09"),
}

_LEDGER_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "policy_id",
        "cost_bps",
        "ordinal",
        "session",
        "adjusted_open_hex",
        "adjusted_close_hex",
        "target_row_sha256",
        "prior_exposure",
        "target_exposure",
        "transition",
        "changing_leg",
        "cash_hex",
        "shares_hex",
        "equity_before_fill_hex",
        "equity_open_hex",
        "period_factor_hex",
        "drawdown_hex",
        "previous_row_sha256",
        "ledger_row_sha256",
    }
)
_LEDGER_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "last_session",
        "exposure",
        "cash_hex",
        "shares_hex",
        "terminal_open_equity_hex",
        "terminal_close_equity_hex",
        "minimum_cash_hex",
        "minimum_shares_hex",
        "maximum_exposure",
        "ledger_row_count",
        "ledger_tip_sha256",
    }
)
_LEDGER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "policy_id",
        "cost_bps",
        "initial_cash_hex",
        "ledger_rows",
        "ledger_rows_sha256",
        "terminal_state",
        "terminal_state_sha256",
        "ledger_sha256",
    }
)
_POLICY_ACTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "policy_id",
        "action_ordinal",
        "prediction_row_sha256",
        "accession_number",
        "decision_session",
        "acceptance_datetime",
        "prediction_available",
        "learner_ready",
        "raw_gate_pass",
        "nonoverlap_blocked",
        "schedule_overlay",
        "effective_action_reason",
        "overlay_schedule_sha256",
        "policy_action_sha256",
    }
)
_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "decision_session",
        "accession_number",
        "acceptance_datetime",
        "feature_row_sha256",
        "arm",
        "fit_as_of_session",
        "frozen_fit_reuse",
        "learner_ready",
        "current_feature_available",
        "prediction_available",
        "fitted_prediction_available",
        "action_available",
        "raw_gate_pass",
        "gate_schedule_overlay",
        "gate_audit",
        "gate_audit_sha256",
        "fit_audit",
        "fit_audit_sha256",
        "prediction_row_sha256",
    }
)
_EPISODE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "overlay_schedule_sha256",
        "policy_id",
        "accession_number",
        "decision_session",
        "entry_session",
        "exit_session",
        "entry_position",
        "exit_position",
        "horizon_sessions",
        "realization_status",
        "prediction_row_sha256",
        "market_prefix_last_session",
        "market_sessions_sha256",
        "overlay_episode_observation_sha256",
    }
)
_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "stage",
        "ledgers",
        "frozen_control_ledgers",
        "feature_rows",
        "feature_rows_sha256",
        "semantic_predictions",
        "semantic_predictions_sha256",
        "no_filing_meaning_predictions",
        "no_filing_meaning_predictions_sha256",
        "no_gemma_channel_predictions",
        "no_gemma_channel_predictions_sha256",
        "frozen_predictions",
        "frozen_predictions_sha256",
        "learner_lessons",
        "learner_lessons_sha256",
        "semantic_policy_actions",
        "semantic_policy_actions_sha256",
        "no_filing_meaning_policy_actions",
        "no_filing_meaning_policy_actions_sha256",
        "no_gemma_channel_policy_actions",
        "no_gemma_channel_policy_actions_sha256",
        "frozen_policy_actions",
        "frozen_policy_actions_sha256",
        "semantic_overlay_episodes",
        "semantic_overlay_episodes_sha256",
        "stage_metrics_input_sha256",
    }
)


class SecGemmaOnlineRiskOverlayMetricsError(ValueError):
    """Raised when metric evidence or a gate report fails closed."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be a mapping"
        )
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be a sequence"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be a canonical ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be Boolean"
        )
    return value


def _binary(value: Any, location: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be canonical integer 0 or 1"
        )
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def _decode_hex(
    value: Any,
    location: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be canonical float.hex text"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be canonical float.hex text"
        ) from exc
    if not math.isfinite(number) or number.hex() != value:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be canonical finite float.hex text"
        )
    if positive and number <= 0.0:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be positive"
        )
    if nonnegative and number < 0.0:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be nonnegative"
        )
    return number


def _float_hex(value: float, location: str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must be finite"
        )
    return number.hex()


def _optional_hex(value: float | None, location: str) -> str | None:
    return None if value is None else _float_hex(value, location)


def _seal(body: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(body))
    result[hash_field] = canonical_sha256(result)
    return result


def _require_self_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    location: str,
) -> str:
    observed = _sha256(value.get(hash_field), f"{location}.{hash_field}")
    body = {key: item for key, item in value.items() if key != hash_field}
    if not hmac.compare_digest(observed, canonical_sha256(body)):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} self-hash changed"
        )
    return observed


def _validate_ledger(
    raw: Mapping[str, Any],
    *,
    expected_cost_bps: int,
    location: str,
) -> dict[str, Any]:
    """Validate one binary ledger using independent account arithmetic."""

    ledger = copy.deepcopy(dict(_mapping(raw, location)))
    if set(ledger) != _LEDGER_KEYS:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} keys changed"
        )
    if (
        ledger["schema_version"] != LEDGER_SCHEMA_VERSION
        or ledger["contract_version"] != CONTRACT_VERSION
        or ledger["contract_sha256"] != CONTRACT_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} experiment identity changed"
        )
    if (
        type(expected_cost_bps) is not int
        or expected_cost_bps not in (5, 10)
        or ledger["cost_bps"] != expected_cost_bps
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} cost changed"
        )
    policy_id = ledger["policy_id"]
    if not isinstance(policy_id, str) or not policy_id:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} policy_id is invalid"
        )
    if ledger["initial_cash_hex"] != INITIAL_CASH.hex():
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} initial cash changed"
        )
    rows = _sequence(ledger["ledger_rows"], f"{location}.ledger_rows")
    if not rows:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} ledger rows must not be empty"
        )
    if ledger["ledger_rows_sha256"] != canonical_sha256(rows):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} row-batch hash changed"
        )

    cash = INITIAL_CASH
    shares = 0.0
    exposure = 0
    previous_equity = INITIAL_CASH
    running_peak = INITIAL_CASH
    previous_hash = GENESIS_PREVIOUS_ROW_SHA256
    previous_session: str | None = None
    minimum_cash = math.inf
    minimum_shares = math.inf
    maximum_exposure = 0
    validated_rows: list[dict[str, Any]] = []
    cost = expected_cost_bps / 10_000.0
    for ordinal, item in enumerate(rows, start=1):
        row = copy.deepcopy(
            dict(_mapping(item, f"{location}.ledger_rows[{ordinal}]"))
        )
        if set(row) != _LEDGER_ROW_KEYS:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} ledger row keys changed"
            )
        if (
            row["schema_version"] != LEDGER_ROW_SCHEMA_VERSION
            or row["contract_version"] != CONTRACT_VERSION
            or row["contract_sha256"] != CONTRACT_SHA256
            or row["policy_id"] != policy_id
            or row["cost_bps"] != expected_cost_bps
            or row["ordinal"] != ordinal
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} ledger row identity changed"
            )
        session = _iso_date(
            row["session"], f"{location}.ledger_rows[{ordinal}].session"
        )
        if previous_session is not None and session <= previous_session:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} sessions are not strictly increasing"
            )
        adjusted_open = _decode_hex(
            row["adjusted_open_hex"],
            f"{location}.ledger_rows[{ordinal}].adjusted_open_hex",
            positive=True,
        )
        _decode_hex(
            row["adjusted_close_hex"],
            f"{location}.ledger_rows[{ordinal}].adjusted_close_hex",
            positive=True,
        )
        _sha256(
            row["target_row_sha256"],
            f"{location}.ledger_rows[{ordinal}].target_row_sha256",
        )
        prior = _binary(
            row["prior_exposure"],
            f"{location}.ledger_rows[{ordinal}].prior_exposure",
        )
        target = _binary(
            row["target_exposure"],
            f"{location}.ledger_rows[{ordinal}].target_exposure",
        )
        if ordinal == 1 and target != 1:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} genesis target is not LONG"
            )
        if prior != exposure:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} prior exposure chain changed"
            )
        expected_changing = prior != target
        if (
            _boolean(
                row["changing_leg"],
                f"{location}.ledger_rows[{ordinal}].changing_leg",
            )
            != expected_changing
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} changing-leg flag changed"
            )
        equity_before = cash + shares * adjusted_open
        if row["equity_before_fill_hex"] != equity_before.hex():
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} pre-fill equity arithmetic changed"
            )
        transition = (
            "SELL"
            if prior == 1 and target == 0
            else "BUY"
            if prior == 0 and target == 1
            else "HOLD_LONG"
            if target == 1
            else "HOLD_CASH"
        )
        if row["transition"] != transition:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} transition changed"
            )
        if transition == "SELL":
            cash = shares * adjusted_open * (1.0 - cost)
            shares = 0.0
        elif transition == "BUY":
            shares = cash / (adjusted_open * (1.0 + cost))
            cash = 0.0
        equity = cash + shares * adjusted_open
        if (
            row["cash_hex"] != cash.hex()
            or row["shares_hex"] != shares.hex()
            or row["equity_open_hex"] != equity.hex()
            or row["period_factor_hex"]
            != (equity / previous_equity).hex()
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} account arithmetic changed"
            )
        running_peak = max(running_peak, equity)
        expected_drawdown = equity / running_peak - 1.0
        if row["drawdown_hex"] != expected_drawdown.hex():
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} drawdown arithmetic changed"
            )
        if row["previous_row_sha256"] != previous_hash:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} row chain changed"
            )
        row_hash = _require_self_hash(
            row,
            hash_field="ledger_row_sha256",
            location=f"{location}.ledger_rows[{ordinal}]",
        )
        previous_hash = row_hash
        previous_equity = equity
        previous_session = session
        exposure = target
        minimum_cash = min(minimum_cash, cash)
        minimum_shares = min(minimum_shares, shares)
        maximum_exposure = max(maximum_exposure, target)
        validated_rows.append(row)

    state = copy.deepcopy(
        dict(_mapping(ledger["terminal_state"], f"{location}.terminal_state"))
    )
    if set(state) != _LEDGER_STATE_KEYS:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} terminal-state keys changed"
        )
    terminal_close = _decode_hex(
        validated_rows[-1]["adjusted_close_hex"],
        f"{location}.terminal adjusted close",
        positive=True,
    )
    terminal_close_equity = cash + shares * terminal_close
    expected_state = {
        "last_session": validated_rows[-1]["session"],
        "exposure": exposure,
        "cash_hex": cash.hex(),
        "shares_hex": shares.hex(),
        "terminal_open_equity_hex": previous_equity.hex(),
        "terminal_close_equity_hex": terminal_close_equity.hex(),
        "minimum_cash_hex": minimum_cash.hex(),
        "minimum_shares_hex": minimum_shares.hex(),
        "maximum_exposure": maximum_exposure,
        "ledger_row_count": len(validated_rows),
        "ledger_tip_sha256": previous_hash,
    }
    if state != expected_state:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} terminal state changed"
        )
    if ledger["terminal_state_sha256"] != canonical_sha256(state):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} terminal-state hash changed"
        )
    _require_self_hash(
        ledger,
        hash_field="ledger_sha256",
        location=location,
    )
    return ledger


def _validate_ledger_family(
    raw: Mapping[str, Any],
    *,
    location: str,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(_mapping(raw, location)))
    if set(value) != set(COST_KEYS):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} must contain exactly 5bps and 10bps ledgers"
        )
    result = {
        "cost_5bps": _validate_ledger(
            value["cost_5bps"],
            expected_cost_bps=5,
            location=f"{location}.cost_5bps",
        ),
        "cost_10bps": _validate_ledger(
            value["cost_10bps"],
            expected_cost_bps=10,
            location=f"{location}.cost_10bps",
        ),
    }
    left_rows = result["cost_5bps"]["ledger_rows"]
    right_rows = result["cost_10bps"]["ledger_rows"]
    if (
        result["cost_5bps"]["policy_id"]
        != result["cost_10bps"]["policy_id"]
        or len(left_rows) != len(right_rows)
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} cost ledgers have different policy identity or length"
        )
    immutable_fields = (
        "session",
        "adjusted_open_hex",
        "adjusted_close_hex",
        "target_row_sha256",
        "prior_exposure",
        "target_exposure",
        "transition",
        "changing_leg",
    )
    for ordinal, (left, right) in enumerate(
        zip(left_rows, right_rows, strict=True), start=1
    ):
        if any(left[name] != right[name] for name in immutable_fields):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} action stream differs by cost at row {ordinal}"
            )
    return result


def _validate_policy_actions(
    raw: Sequence[Mapping[str, Any]],
    *,
    location: str,
) -> list[dict[str, Any]]:
    rows = _sequence(raw, location)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_key: tuple[str, str, str] | None = None
    policy_id: str | None = None
    for ordinal, item in enumerate(rows, start=1):
        row = copy.deepcopy(dict(_mapping(item, f"{location}[{ordinal}]")))
        if set(row) != _POLICY_ACTION_KEYS:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} action keys changed"
            )
        if (
            row["schema_version"] != POLICY_ACTION_ROW_SCHEMA_VERSION
            or row["contract_version"] != CONTRACT_VERSION
            or row["contract_sha256"] != CONTRACT_SHA256
            or row["action_ordinal"] != ordinal
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} action identity changed"
            )
        if policy_id is None:
            policy_id = row["policy_id"]
        if (
            not isinstance(row["policy_id"], str)
            or not row["policy_id"]
            or row["policy_id"] != policy_id
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} policy identity changed"
            )
        accession = row["accession_number"]
        if (
            not isinstance(accession, str)
            or not accession
            or accession in seen
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} accession is invalid or duplicated"
            )
        seen.add(accession)
        session = _iso_date(
            row["decision_session"],
            f"{location}[{ordinal}].decision_session",
        )
        acceptance = row["acceptance_datetime"]
        if acceptance is not None and (
            not isinstance(acceptance, str)
            or len(acceptance) != 14
            or not acceptance.isascii()
            or not acceptance.isdigit()
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} acceptance timestamp is invalid"
            )
        key = (session, "" if acceptance is None else acceptance, accession)
        if previous_key is not None and key < previous_key:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} actions are not chronologically ordered"
            )
        previous_key = key
        for field in (
            "prediction_available",
            "learner_ready",
            "raw_gate_pass",
            "nonoverlap_blocked",
            "schedule_overlay",
        ):
            _boolean(row[field], f"{location}[{ordinal}].{field}")
        if row["schedule_overlay"] and (
            not row["prediction_available"]
            or not row["learner_ready"]
            or not row["raw_gate_pass"]
            or row["nonoverlap_blocked"]
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} scheduled action is inconsistent"
            )
        _sha256(
            row["prediction_row_sha256"],
            f"{location}[{ordinal}].prediction_row_sha256",
        )
        if row["schedule_overlay"]:
            _sha256(
                row["overlay_schedule_sha256"],
                f"{location}[{ordinal}].overlay_schedule_sha256",
            )
        elif row["overlay_schedule_sha256"] is not None:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} unscheduled action exposes a schedule hash"
            )
        _require_self_hash(
            row,
            hash_field="policy_action_sha256",
            location=f"{location}[{ordinal}]",
        )
        result.append(row)
    return result


def _validate_prediction_rows(
    raw: Sequence[Mapping[str, Any]],
    *,
    expected_arm: str,
    location: str,
) -> list[dict[str, Any]]:
    rows = _sequence(raw, location)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, item in enumerate(rows, start=1):
        row = copy.deepcopy(dict(_mapping(item, f"{location}[{ordinal}]")))
        if set(row) != _PREDICTION_KEYS:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} prediction keys changed"
            )
        if (
            row["contract_version"] != CONTRACT_VERSION
            or row["contract_sha256"] != CONTRACT_SHA256
            or row["arm"] != expected_arm
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} prediction identity changed"
            )
        if not isinstance(row["schema_version"], str) or not row[
            "schema_version"
        ]:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} prediction schema is invalid"
            )
        accession = row["accession_number"]
        if (
            not isinstance(accession, str)
            or not accession
            or accession in seen
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} prediction accession is invalid or duplicated"
            )
        seen.add(accession)
        _iso_date(
            row["decision_session"],
            f"{location}[{ordinal}].decision_session",
        )
        _iso_date(
            row["fit_as_of_session"],
            f"{location}[{ordinal}].fit_as_of_session",
        )
        _sha256(
            row["feature_row_sha256"],
            f"{location}[{ordinal}].feature_row_sha256",
        )
        for field in (
            "frozen_fit_reuse",
            "learner_ready",
            "current_feature_available",
            "prediction_available",
            "fitted_prediction_available",
            "action_available",
            "raw_gate_pass",
            "gate_schedule_overlay",
        ):
            _boolean(row[field], f"{location}[{ordinal}].{field}")
        if row["prediction_available"] != row["current_feature_available"]:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} prediction availability changed"
            )
        if (
            row["fitted_prediction_available"]
            != row["action_available"]
            or row["fitted_prediction_available"]
            != (
                row["learner_ready"] and row["prediction_available"]
            )
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} fitted prediction availability changed"
            )
        fit = _mapping(
            row["fit_audit"], f"{location}[{ordinal}].fit_audit"
        )
        fit_hash = _sha256(
            row["fit_audit_sha256"],
            f"{location}[{ordinal}].fit_audit_sha256",
        )
        if fit.get("fit_audit_sha256") != fit_hash:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} fit-audit binding changed"
            )
        _require_self_hash(
            fit,
            hash_field="fit_audit_sha256",
            location=f"{location}[{ordinal}].fit_audit",
        )
        gate = row["gate_audit"]
        if row["fitted_prediction_available"]:
            gate_map = _mapping(
                gate, f"{location}[{ordinal}].gate_audit"
            )
            gate_hash = _sha256(
                row["gate_audit_sha256"],
                f"{location}[{ordinal}].gate_audit_sha256",
            )
            if gate_map.get("gate_audit_sha256") != gate_hash:
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    f"{location} gate-audit binding changed"
                )
            _require_self_hash(
                gate_map,
                hash_field="gate_audit_sha256",
                location=f"{location}[{ordinal}].gate_audit",
            )
            probability = _decode_hex(
                gate_map.get("probability_hex"),
                f"{location}[{ordinal}].probability_hex",
            )
            if not 0.0 <= probability <= 1.0:
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    f"{location} probability lies outside [0, 1]"
                )
            if row["raw_gate_pass"] != bool(
                gate_map.get("overlay_gate_passed")
            ):
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    f"{location} raw gate differs from its audit"
                )
        elif gate is not None or row["gate_audit_sha256"] is not None:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} unavailable prediction exposes a gate audit"
            )
        _require_self_hash(
            row,
            hash_field="prediction_row_sha256",
            location=f"{location}[{ordinal}]",
        )
        result.append(row)
    return result


def _validate_episodes(
    raw: Sequence[Mapping[str, Any]],
    *,
    location: str,
) -> list[dict[str, Any]]:
    rows = _sequence(raw, location)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, item in enumerate(rows, start=1):
        row = copy.deepcopy(dict(_mapping(item, f"{location}[{ordinal}]")))
        if set(row) != _EPISODE_KEYS:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} episode keys changed"
            )
        if (
            row["schema_version"]
            != OVERLAY_EPISODE_OBSERVATION_SCHEMA_VERSION
            or row["contract_version"] != CONTRACT_VERSION
            or row["contract_sha256"] != CONTRACT_SHA256
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} episode identity changed"
            )
        schedule_hash = _sha256(
            row["overlay_schedule_sha256"],
            f"{location}[{ordinal}].overlay_schedule_sha256",
        )
        if schedule_hash in seen:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} episode schedule is duplicated"
            )
        seen.add(schedule_hash)
        _iso_date(
            row["decision_session"],
            f"{location}[{ordinal}].decision_session",
        )
        status = row["realization_status"]
        if status not in {
            "complete",
            "active_pending_exit",
            "pending_entry",
        }:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} episode realization status changed"
            )
        entry = row["entry_session"]
        exit_ = row["exit_session"]
        if entry is not None:
            _iso_date(entry, f"{location}[{ordinal}].entry_session")
        if exit_ is not None:
            _iso_date(exit_, f"{location}[{ordinal}].exit_session")
        if (
            (status == "complete" and (entry is None or exit_ is None))
            or (
                status == "active_pending_exit"
                and (entry is None or exit_ is not None)
            )
            or (
                status == "pending_entry"
                and (entry is not None or exit_ is not None)
            )
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} episode boundary status changed"
            )
        _integer(
            row["entry_position"],
            f"{location}[{ordinal}].entry_position",
        )
        _integer(
            row["exit_position"],
            f"{location}[{ordinal}].exit_position",
        )
        if row["horizon_sessions"] != HORIZON_SESSIONS:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} episode horizon changed"
            )
        _sha256(
            row["prediction_row_sha256"],
            f"{location}[{ordinal}].prediction_row_sha256",
        )
        _require_self_hash(
            row,
            hash_field="overlay_episode_observation_sha256",
            location=f"{location}[{ordinal}]",
        )
        result.append(row)
    return result


def _expected_episode_observations(
    *,
    scheduled_overlays: Sequence[Mapping[str, Any]],
    market_sessions: Sequence[str],
) -> list[dict[str, Any]]:
    sessions = list(market_sessions)
    positions = {session: position for position, session in enumerate(sessions)}
    result: list[dict[str, Any]] = []
    for schedule in scheduled_overlays:
        decision = schedule["decision_session"]
        if decision not in positions:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "scheduled overlay decision is outside the market prefix"
            )
        decision_position = positions[decision]
        entry_position = decision_position + 1
        exit_position = decision_position + LABEL_MATURITY_OFFSET
        entry_session = (
            sessions[entry_position]
            if entry_position < len(sessions)
            else None
        )
        exit_session = (
            sessions[exit_position]
            if exit_position < len(sessions)
            else None
        )
        status = (
            "pending_entry"
            if entry_session is None
            else "complete"
            if exit_session is not None
            else "active_pending_exit"
        )
        body = {
            "schema_version": OVERLAY_EPISODE_OBSERVATION_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "overlay_schedule_sha256": schedule[
                "overlay_schedule_sha256"
            ],
            "policy_id": schedule["policy_id"],
            "accession_number": schedule["accession_number"],
            "decision_session": decision,
            "entry_session": entry_session,
            "exit_session": exit_session,
            "entry_position": entry_position,
            "exit_position": exit_position,
            "horizon_sessions": HORIZON_SESSIONS,
            "realization_status": status,
            "prediction_row_sha256": schedule[
                "prediction_row_sha256"
            ],
            "market_prefix_last_session": sessions[-1],
            "market_sessions_sha256": canonical_sha256(sessions),
        }
        result.append(
            {
                **body,
                "overlay_episode_observation_sha256": canonical_sha256(
                    body
                ),
            }
        )
    return result


def _require_episode_observations_match_policy(
    *,
    episodes: Sequence[Mapping[str, Any]],
    scheduled_overlays: Sequence[Mapping[str, Any]],
    market_sessions: Sequence[str],
) -> None:
    expected = _expected_episode_observations(
        scheduled_overlays=scheduled_overlays,
        market_sessions=market_sessions,
    )
    if list(episodes) != expected:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "semantic episode observations differ from deterministic "
            "schedule realization"
        )


def build_stage_metrics_input(
    *,
    stage: str,
    ledgers: Mapping[str, Mapping[str, Mapping[str, Any]]],
    frozen_control_ledgers: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    feature_rows: Sequence[Mapping[str, Any]],
    semantic_predictions: Sequence[Mapping[str, Any]],
    no_filing_meaning_predictions: Sequence[Mapping[str, Any]],
    no_gemma_channel_predictions: Sequence[Mapping[str, Any]],
    frozen_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    learner_lessons: Sequence[Mapping[str, Any]],
    semantic_policy_actions: Sequence[Mapping[str, Any]],
    no_filing_meaning_policy_actions: Sequence[Mapping[str, Any]],
    no_gemma_channel_policy_actions: Sequence[Mapping[str, Any]],
    frozen_policy_actions: Mapping[str, Sequence[Mapping[str, Any]]],
    semantic_overlay_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the exact pure inputs consumed by the metric layer."""

    body = {
        "schema_version": METRICS_INPUT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "ledgers": copy.deepcopy(dict(ledgers)),
        "frozen_control_ledgers": copy.deepcopy(
            dict(frozen_control_ledgers)
        ),
        "feature_rows": copy.deepcopy(list(feature_rows)),
        "feature_rows_sha256": canonical_sha256(feature_rows),
        "semantic_predictions": copy.deepcopy(list(semantic_predictions)),
        "semantic_predictions_sha256": canonical_sha256(
            semantic_predictions
        ),
        "no_filing_meaning_predictions": copy.deepcopy(
            list(no_filing_meaning_predictions)
        ),
        "no_filing_meaning_predictions_sha256": canonical_sha256(
            no_filing_meaning_predictions
        ),
        "no_gemma_channel_predictions": copy.deepcopy(
            list(no_gemma_channel_predictions)
        ),
        "no_gemma_channel_predictions_sha256": canonical_sha256(
            no_gemma_channel_predictions
        ),
        "frozen_predictions": copy.deepcopy(dict(frozen_predictions)),
        "frozen_predictions_sha256": canonical_sha256(
            frozen_predictions
        ),
        "learner_lessons": copy.deepcopy(list(learner_lessons)),
        "learner_lessons_sha256": canonical_sha256(learner_lessons),
        "semantic_policy_actions": copy.deepcopy(
            list(semantic_policy_actions)
        ),
        "semantic_policy_actions_sha256": canonical_sha256(
            semantic_policy_actions
        ),
        "no_filing_meaning_policy_actions": copy.deepcopy(
            list(no_filing_meaning_policy_actions)
        ),
        "no_filing_meaning_policy_actions_sha256": canonical_sha256(
            no_filing_meaning_policy_actions
        ),
        "no_gemma_channel_policy_actions": copy.deepcopy(
            list(no_gemma_channel_policy_actions)
        ),
        "no_gemma_channel_policy_actions_sha256": canonical_sha256(
            no_gemma_channel_policy_actions
        ),
        "frozen_policy_actions": copy.deepcopy(
            dict(frozen_policy_actions)
        ),
        "frozen_policy_actions_sha256": canonical_sha256(
            frozen_policy_actions
        ),
        "semantic_overlay_episodes": copy.deepcopy(
            list(semantic_overlay_episodes)
        ),
        "semantic_overlay_episodes_sha256": canonical_sha256(
            semantic_overlay_episodes
        ),
    }
    sealed = _seal(body, "stage_metrics_input_sha256")
    _validate_stage_metrics_input(
        sealed,
        expected_stage_metrics_input_sha256=sealed[
            "stage_metrics_input_sha256"
        ],
    )
    return sealed


def _validate_stage_metrics_input(
    raw: Mapping[str, Any],
    *,
    expected_stage_metrics_input_sha256: str,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(_mapping(raw, "stage metrics input")))
    if set(value) != _INPUT_KEYS:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage metrics input keys changed"
        )
    if (
        value["schema_version"] != METRICS_INPUT_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"] != CONTRACT_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage metrics input identity changed"
        )
    stage = value["stage"]
    if stage not in STAGES:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "unknown metric stage"
        )
    observed_input_hash = _require_self_hash(
        value,
        hash_field="stage_metrics_input_sha256",
        location="stage metrics input",
    )
    expected_input_hash = _sha256(
        expected_stage_metrics_input_sha256,
        "expected stage metrics input hash",
    )
    if not hmac.compare_digest(observed_input_hash, expected_input_hash):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage metrics input is not externally pinned"
        )

    ledgers_raw = copy.deepcopy(
        dict(_mapping(value["ledgers"], "stage ledgers"))
    )
    if set(ledgers_raw) != set(CORE_ACCOUNT_NAMES):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage ledgers must contain the exact five declared accounts"
        )
    ledgers = {
        name: _validate_ledger_family(
            ledgers_raw[name], location=f"stage ledgers.{name}"
        )
        for name in CORE_ACCOUNT_NAMES
    }
    frozen_raw = copy.deepcopy(
        dict(
            _mapping(
                value["frozen_control_ledgers"],
                "frozen control ledgers",
            )
        )
    )
    expected_controls = set(FROZEN_CONTROL_IDS[stage])
    if set(frozen_raw) != expected_controls:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "frozen control ledger identities changed"
        )
    frozen_ledgers = {
        control_id: _validate_ledger_family(
            frozen_raw[control_id],
            location=f"frozen control ledgers.{control_id}",
        )
        for control_id in FROZEN_CONTROL_IDS[stage]
    }

    reference_rows = ledgers["semantic"]["cost_5bps"]["ledger_rows"]
    reference_sessions = [row["session"] for row in reference_rows]
    if (
        reference_sessions[0] != "2000-01-03"
        or reference_sessions[-1] != STAGE_CUTOFFS[stage]
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "continuous ledger genesis or stage cutoff changed"
        )
    for cost_key in COST_KEYS:
        if any(
            row["target_exposure"] != 1
            for row in ledgers["aapl_buy_and_hold"][cost_key][
                "ledger_rows"
            ]
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "same-ledger AAPL benchmark is not continuously LONG"
            )
    alignment_fields = ("session", "adjusted_open_hex", "adjusted_close_hex")
    for name, family in {
        **ledgers,
        **{
            f"frozen:{key}": item
            for key, item in frozen_ledgers.items()
        },
    }.items():
        for cost_key in COST_KEYS:
            rows = family[cost_key]["ledger_rows"]
            if len(rows) != len(reference_rows) or any(
                any(left[field] != right[field] for field in alignment_fields)
                for left, right in zip(
                    rows, reference_rows, strict=True
                )
            ):
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    f"{name} is not same-ledger aligned"
                )

    feature_rows_raw = _sequence(
        value["feature_rows"], "feature rows"
    )
    if value["feature_rows_sha256"] != canonical_sha256(feature_rows_raw):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "feature row batch hash changed"
        )
    feature_rows: list[dict[str, Any]] = []
    feature_accessions: set[str] = set()
    for ordinal, raw_feature in enumerate(feature_rows_raw, start=1):
        feature_hash = _sha256(
            _mapping(raw_feature, f"feature row {ordinal}").get(
                "feature_row_sha256"
            ),
            f"feature row {ordinal}.feature_row_sha256",
        )
        feature = validate_online_overlay_feature_row(
            raw_feature,
            expected_feature_row_sha256=feature_hash,
        )
        accession = feature["accession_number"]
        if accession in feature_accessions:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "feature accession is duplicated"
            )
        feature_accessions.add(accession)
        if feature["decision_session"] not in set(reference_sessions):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "feature decision lies outside the continuous ledger"
            )
        feature_rows.append(feature)

    semantic_predictions_raw = _sequence(
        value["semantic_predictions"], "semantic predictions"
    )
    no_meaning_predictions_raw = _sequence(
        value["no_filing_meaning_predictions"],
        "no-filing-meaning predictions",
    )
    no_gemma_predictions_raw = _sequence(
        value["no_gemma_channel_predictions"],
        "no-Gemma-channel predictions",
    )
    if (
        value["semantic_predictions_sha256"]
        != canonical_sha256(semantic_predictions_raw)
        or value["no_filing_meaning_predictions_sha256"]
        != canonical_sha256(no_meaning_predictions_raw)
        or value["no_gemma_channel_predictions_sha256"]
        != canonical_sha256(no_gemma_predictions_raw)
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "prediction batch hash changed"
        )
    semantic_predictions = _validate_prediction_rows(
        semantic_predictions_raw,
        expected_arm="semantic",
        location="semantic predictions",
    )
    no_meaning_predictions = _validate_prediction_rows(
        no_meaning_predictions_raw,
        expected_arm="no_filing_meaning",
        location="no-filing-meaning predictions",
    )
    no_gemma_predictions = _validate_prediction_rows(
        no_gemma_predictions_raw,
        expected_arm="no_gemma_channel",
        location="no-Gemma-channel predictions",
    )
    frozen_predictions_raw = copy.deepcopy(
        dict(
            _mapping(
                value["frozen_predictions"],
                "frozen predictions",
            )
        )
    )
    if (
        set(frozen_predictions_raw) != expected_controls
        or value["frozen_predictions_sha256"]
        != canonical_sha256(frozen_predictions_raw)
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "frozen prediction controls or batch hash changed"
        )
    frozen_predictions = {
        control_id: _validate_prediction_rows(
            frozen_predictions_raw[control_id],
            expected_arm="semantic",
            location=f"frozen predictions.{control_id}",
        )
        for control_id in FROZEN_CONTROL_IDS[stage]
    }
    semantic_by_accession = {
        row["accession_number"]: row for row in semantic_predictions
    }
    no_meaning_by_accession = {
        row["accession_number"]: row
        for row in no_meaning_predictions
    }
    no_gemma_by_accession = {
        row["accession_number"]: row for row in no_gemma_predictions
    }
    frozen_prediction_maps = {
        control_id: {
            row["accession_number"]: row for row in control_predictions
        }
        for control_id, control_predictions in frozen_predictions.items()
    }
    if (
        set(semantic_by_accession) != feature_accessions
        or set(no_meaning_by_accession) != feature_accessions
        or set(no_gemma_by_accession) != feature_accessions
        or any(
            set(control_map) != feature_accessions
            for control_map in frozen_prediction_maps.values()
        )
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "prediction support differs from the eligible feature universe"
        )
    feature_by_accession = {
        row["accession_number"]: row for row in feature_rows
    }
    for accession in feature_accessions:
        feature = feature_by_accession[accession]
        semantic = semantic_by_accession[accession]
        no_meaning = no_meaning_by_accession[accession]
        no_gemma = no_gemma_by_accession[accession]
        identity = (
            "decision_session",
            "acceptance_datetime",
            "feature_row_sha256",
            "prediction_available",
            "learner_ready",
        )
        if any(
            semantic[field] != comparison[field]
            for comparison in (no_meaning, no_gemma)
            for field in identity
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "semantic prediction supports are not identical"
            )
        if (
            semantic["decision_session"] != feature["decision_session"]
            or semantic["feature_row_sha256"]
            != feature["feature_row_sha256"]
            or semantic["prediction_available"]
            != feature["prediction_available"]
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "prediction differs from its feature row"
            )
        for prediction in (semantic, no_meaning, no_gemma):
            validate_online_overlay_prediction_from_fit(
                prediction,
                expected_prediction_row_sha256=prediction[
                    "prediction_row_sha256"
                ],
                feature_row=feature,
                expected_feature_row_sha256=feature[
                    "feature_row_sha256"
                ],
                fit_audit=prediction["fit_audit"],
                expected_fit_audit_sha256=prediction[
                    "fit_audit_sha256"
                ],
            )
        for control_id, frozen_by_accession in (
            frozen_prediction_maps.items()
        ):
            frozen = frozen_by_accession[accession]
            if (
                frozen["decision_session"] != feature["decision_session"]
                or frozen["feature_row_sha256"]
                != feature["feature_row_sha256"]
                or frozen["prediction_available"]
                != feature["prediction_available"]
            ):
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    "frozen prediction differs from its feature row"
                )
            validate_online_overlay_prediction_from_fit(
                frozen,
                expected_prediction_row_sha256=frozen[
                    "prediction_row_sha256"
                ],
                feature_row=feature,
                expected_feature_row_sha256=feature[
                    "feature_row_sha256"
                ],
                fit_audit=frozen["fit_audit"],
                expected_fit_audit_sha256=frozen[
                    "fit_audit_sha256"
                ],
            )

    lessons_raw = _sequence(value["learner_lessons"], "learner lessons")
    if value["learner_lessons_sha256"] != canonical_sha256(lessons_raw):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "learner lesson batch hash changed"
        )
    lessons: list[dict[str, Any]] = []
    lesson_accessions: set[str] = set()
    for ordinal, raw_lesson in enumerate(lessons_raw, start=1):
        lesson_hash = _sha256(
            _mapping(raw_lesson, f"learner lesson {ordinal}").get(
                "lesson_row_sha256"
            ),
            f"learner lesson {ordinal}.lesson_row_sha256",
        )
        lesson = validate_online_overlay_lesson(
            raw_lesson,
            expected_lesson_row_sha256=lesson_hash,
        )
        accession = lesson["accession_number"]
        if (
            accession in lesson_accessions
            or accession not in feature_accessions
            or lesson["feature_row_sha256"]
            != feature_by_accession[accession]["feature_row_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "learner lesson support changed"
            )
        lesson_accessions.add(accession)
        lessons.append(lesson)

    action_batches = {
        "semantic": (
            value["semantic_policy_actions"],
            value["semantic_policy_actions_sha256"],
        ),
        "no_filing_meaning": (
            value["no_filing_meaning_policy_actions"],
            value["no_filing_meaning_policy_actions_sha256"],
        ),
        "no_gemma_channel": (
            value["no_gemma_channel_policy_actions"],
            value["no_gemma_channel_policy_actions_sha256"],
        ),
    }
    actions: dict[str, list[dict[str, Any]]] = {}
    for name, (raw_actions, batch_hash) in action_batches.items():
        if batch_hash != canonical_sha256(raw_actions):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{name} action batch hash changed"
            )
        actions[name] = _validate_policy_actions(
            raw_actions, location=f"{name} policy actions"
        )
    frozen_actions_raw = copy.deepcopy(
        dict(
            _mapping(
                value["frozen_policy_actions"],
                "frozen policy actions",
            )
        )
    )
    if (
        set(frozen_actions_raw) != expected_controls
        or value["frozen_policy_actions_sha256"]
        != canonical_sha256(frozen_actions_raw)
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "frozen action controls or batch hash changed"
        )
    frozen_actions = {
        control_id: _validate_policy_actions(
            frozen_actions_raw[control_id],
            location=f"frozen policy actions.{control_id}",
        )
        for control_id in FROZEN_CONTROL_IDS[stage]
    }
    expected_action_accessions = feature_accessions
    for name, batch in {
        **actions,
        **{
            f"frozen:{key}": item
            for key, item in frozen_actions.items()
        },
    }.items():
        if {row["accession_number"] for row in batch} != (
            expected_action_accessions
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{name} action support differs from the feature universe"
            )
    prediction_batches = {
        "semantic": semantic_by_accession,
        "no_filing_meaning": no_meaning_by_accession,
        "no_gemma_channel": no_gemma_by_accession,
        **{
            f"frozen:{key}": item
            for key, item in frozen_prediction_maps.items()
        },
    }
    action_batch_map = {
        "semantic": actions["semantic"],
        "no_filing_meaning": actions["no_filing_meaning"],
        "no_gemma_channel": actions["no_gemma_channel"],
        **{
            f"frozen:{key}": item
            for key, item in frozen_actions.items()
        },
    }
    prediction_sequence_batches = {
        "semantic": semantic_predictions,
        "no_filing_meaning": no_meaning_predictions,
        "no_gemma_channel": no_gemma_predictions,
        **{
            f"frozen:{key}": item
            for key, item in frozen_predictions.items()
        },
    }
    policy_ids = {
        "semantic": ledgers["semantic"]["cost_5bps"]["policy_id"],
        "no_filing_meaning": ledgers["no_filing_meaning"][
            "cost_5bps"
        ]["policy_id"],
        "no_gemma_channel": ledgers["no_gemma_channel"][
            "cost_5bps"
        ]["policy_id"],
        **{
            f"frozen:{key}": item["cost_5bps"]["policy_id"]
            for key, item in frozen_ledgers.items()
        },
    }
    rebuilt_policies: dict[str, dict[str, Any]] = {}
    for name, batch in action_batch_map.items():
        prediction_map = prediction_batches[name]
        for action in batch:
            prediction = prediction_map[action["accession_number"]]
            if (
                action["prediction_row_sha256"]
                != prediction["prediction_row_sha256"]
                or action["decision_session"]
                != prediction["decision_session"]
                or action["acceptance_datetime"]
                != prediction["acceptance_datetime"]
                or action["prediction_available"]
                != prediction["prediction_available"]
                or action["learner_ready"]
                != prediction["learner_ready"]
                or action["raw_gate_pass"]
                != prediction["raw_gate_pass"]
            ):
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    f"{name} action differs from its prediction"
                )
        policy_id = policy_ids[name]
        if batch and batch[0]["policy_id"] != policy_id:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{name} action and ledger policy identities differ"
            )
        rebuilt_policy = replay_nonoverlapping_overlay_policy(
            market_sessions=reference_sessions,
            prediction_rows=prediction_sequence_batches[name],
            policy_id=policy_id,
        )
        if batch != rebuilt_policy["policy_actions"]:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{name} actions differ from deterministic policy replay"
            )
        rebuilt_policies[name] = rebuilt_policy
    for accession in feature_accessions:
        sem = next(
            row
            for row in actions["semantic"]
            if row["accession_number"] == accession
        )
        abl = next(
            row
            for row in actions["no_filing_meaning"]
            if row["accession_number"] == accession
        )
        diagnostic = next(
            row
            for row in actions["no_gemma_channel"]
            if row["accession_number"] == accession
        )
        if any(
            sem[field] != comparison[field]
            for comparison in (abl, diagnostic)
            for field in (
                "decision_session",
                "acceptance_datetime",
                "prediction_available",
                "learner_ready",
            )
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "semantic action supports are not identical"
            )

    episodes_raw = _sequence(
        value["semantic_overlay_episodes"], "semantic overlay episodes"
    )
    if (
        value["semantic_overlay_episodes_sha256"]
        != canonical_sha256(episodes_raw)
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "semantic episode batch hash changed"
        )
    episodes = _validate_episodes(
        episodes_raw, location="semantic overlay episodes"
    )
    session_positions = {
        session: position
        for position, session in enumerate(reference_sessions)
    }
    for episode in episodes:
        entry = episode["entry_session"]
        exit_ = episode["exit_session"]
        if (
            entry is not None
            and (
                entry not in session_positions
                or episode["entry_position"] != session_positions[entry]
            )
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "episode entry position differs from the continuous ledger"
            )
        if (
            exit_ is not None
            and (
                exit_ not in session_positions
                or episode["exit_position"] != session_positions[exit_]
            )
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "episode exit position differs from the continuous ledger"
            )
        if (
            episode["market_prefix_last_session"] != reference_sessions[-1]
            or episode["market_sessions_sha256"]
            != canonical_sha256(reference_sessions)
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "episode market-prefix cutoff changed"
            )
    _require_episode_observations_match_policy(
        episodes=episodes,
        scheduled_overlays=rebuilt_policies["semantic"][
            "scheduled_overlays"
        ],
        market_sessions=reference_sessions,
    )

    value["ledgers"] = ledgers
    value["frozen_control_ledgers"] = frozen_ledgers
    value["feature_rows"] = feature_rows
    value["semantic_predictions"] = semantic_predictions
    value["no_filing_meaning_predictions"] = no_meaning_predictions
    value["no_gemma_channel_predictions"] = no_gemma_predictions
    value["frozen_predictions"] = frozen_predictions
    value["learner_lessons"] = lessons
    value["semantic_policy_actions"] = actions["semantic"]
    value["no_filing_meaning_policy_actions"] = actions[
        "no_filing_meaning"
    ]
    value["no_gemma_channel_policy_actions"] = actions[
        "no_gemma_channel"
    ]
    value["frozen_policy_actions"] = frozen_actions
    value["semantic_overlay_episodes"] = episodes
    return value


def validate_stage_metrics_input(
    value: Mapping[str, Any],
    *,
    expected_stage_metrics_input_sha256: str,
) -> str:
    """Validate an externally pinned metric-input artifact."""

    validated = _validate_stage_metrics_input(
        value,
        expected_stage_metrics_input_sha256=(
            expected_stage_metrics_input_sha256
        ),
    )
    return validated["stage_metrics_input_sha256"]


def _window_definitions(stage: str) -> dict[str, tuple[str, str]]:
    if stage == "development":
        result = {
            "development_total": PERFORMANCE_WINDOWS["development"]
        }
        result.update(
            {
                block_id: (first, last)
                for block_id, first, last in DEVELOPMENT_BLOCKS
            }
        )
        result.update(
            {
                f"year_{year}": (f"{year}-01-01", f"{year}-12-31")
                for year in range(2005, 2019)
            }
        )
        return result
    if stage == "confirmation":
        result = {
            "confirmation_total": PERFORMANCE_WINDOWS["confirmation"]
        }
        result.update(
            {
                f"year_{year}": (f"{year}-01-01", f"{year}-12-31")
                for year in range(2019, 2024)
            }
        )
        return result
    return {
        "final_continuous": PERFORMANCE_WINDOWS["final"],
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026_ytd": ("2026-01-01", "2026-07-09"),
    }


def _aligned_rows(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_rows = list(left["ledger_rows"])
    right_rows = list(right["ledger_rows"])
    if len(left_rows) != len(right_rows) or any(
        left_row["session"] != right_row["session"]
        or left_row["adjusted_open_hex"]
        != right_row["adjusted_open_hex"]
        or left_row["adjusted_close_hex"]
        != right_row["adjusted_close_hex"]
        for left_row, right_row in zip(
            left_rows, right_rows, strict=True
        )
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "comparison ledgers are not same-ledger aligned"
        )
    return left_rows, right_rows


def _window_positions(
    rows: Sequence[Mapping[str, Any]],
    *,
    first: str,
    last: str,
    location: str,
) -> list[int]:
    first = _iso_date(first, f"{location}.first")
    last = _iso_date(last, f"{location}.last")
    if first > last:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} window is reversed"
        )
    positions = [
        position
        for position, row in enumerate(rows)
        if first <= row["session"] <= last
    ]
    if not positions:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} has no physical ledger support"
        )
    return positions


def _path_metrics(
    rows: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
    *,
    valuation_method: str,
    location: str,
) -> tuple[float, float, float]:
    if valuation_method not in VALUATION_METHODS:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{location} valuation method changed"
        )
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    log_terms: list[float] = []
    for position in positions:
        factor = _decode_hex(
            rows[position]["period_factor_hex"],
            f"{location}.period_factor[{position}]",
            positive=True,
        )
        wealth *= factor
        if not math.isfinite(wealth) or wealth <= 0.0:
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{location} normalized wealth is invalid"
            )
        log_terms.append(math.log(factor))
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
    if valuation_method == "terminal_adjusted_close":
        final_row = rows[positions[-1]]
        close_factor = (
            _decode_hex(
                final_row["adjusted_close_hex"],
                f"{location}.terminal adjusted close",
                positive=True,
            )
            / _decode_hex(
                final_row["adjusted_open_hex"],
                f"{location}.terminal adjusted open",
                positive=True,
            )
            if final_row["target_exposure"] == 1
            else 1.0
        )
        wealth *= close_factor
        log_terms.append(math.log(close_factor))
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
    total_log_return = math.fsum(log_terms)
    return total_log_return, wealth, drawdown


def _window_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison_id: str,
    first: str,
    last: str,
    valuation_method: str,
) -> dict[str, Any]:
    left_rows, right_rows = _aligned_rows(left, right)
    positions = _window_positions(
        left_rows,
        first=first,
        last=last,
        location=f"window {comparison_id}",
    )
    left_return, left_wealth, left_mdd = _path_metrics(
        left_rows,
        positions,
        valuation_method=valuation_method,
        location=f"window {comparison_id}.left",
    )
    right_return, right_wealth, right_mdd = _path_metrics(
        right_rows,
        positions,
        valuation_method=valuation_method,
        location=f"window {comparison_id}.right",
    )
    edge_terms = [
        math.log(
            _decode_hex(
                left_rows[position]["period_factor_hex"],
                "left window edge factor",
                positive=True,
            )
        )
        - math.log(
            _decode_hex(
                right_rows[position]["period_factor_hex"],
                "right window edge factor",
                positive=True,
            )
        )
        for position in positions
    ]
    if valuation_method == "terminal_adjusted_close":
        left_final = left_rows[positions[-1]]
        right_final = right_rows[positions[-1]]
        left_close_factor = (
            _decode_hex(
                left_final["adjusted_close_hex"],
                "left window final close",
                positive=True,
            )
            / _decode_hex(
                left_final["adjusted_open_hex"],
                "left window final open",
                positive=True,
            )
            if left_final["target_exposure"] == 1
            else 1.0
        )
        right_close_factor = (
            _decode_hex(
                right_final["adjusted_close_hex"],
                "right window final close",
                positive=True,
            )
            / _decode_hex(
                right_final["adjusted_open_hex"],
                "right window final open",
                positive=True,
            )
            if right_final["target_exposure"] == 1
            else 1.0
        )
        edge_terms.append(
            math.log(left_close_factor) - math.log(right_close_factor)
        )
    log_edge = math.fsum(edge_terms)
    body = {
        "schema_version": WINDOW_COMPARISON_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "comparison_id": comparison_id,
        "valuation_method": valuation_method,
        "requested_first_session": first,
        "requested_last_session": last,
        "first_physical_session": left_rows[positions[0]]["session"],
        "last_physical_session": left_rows[positions[-1]]["session"],
        "assigned_ledger_row_count": len(positions),
        "left_ledger_sha256": left["ledger_sha256"],
        "right_ledger_sha256": right["ledger_sha256"],
        "left_log_return_hex": _float_hex(
            left_return, "left window log return"
        ),
        "right_log_return_hex": _float_hex(
            right_return, "right window log return"
        ),
        "log_edge_hex": _float_hex(
            log_edge, "window log edge"
        ),
        "left_terminal_wealth_hex": _float_hex(
            left_wealth, "left window terminal wealth"
        ),
        "right_terminal_wealth_hex": _float_hex(
            right_wealth, "right window terminal wealth"
        ),
        "left_maximum_drawdown_hex": _float_hex(
            left_mdd, "left window MDD"
        ),
        "right_maximum_drawdown_hex": _float_hex(
            right_mdd, "right window MDD"
        ),
    }
    return _seal(body, "window_comparison_sha256")


def _build_window_metrics(
    validated_input: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = validated_input["stage"]
    windows = _window_definitions(stage)
    ledgers = validated_input["ledgers"]
    comparisons = {
        "semantic_vs_aapl": ("semantic", "aapl_buy_and_hold"),
        "semantic_vs_baseline": ("semantic", "baseline"),
        "semantic_vs_no_filing_meaning": (
            "semantic",
            "no_filing_meaning",
        ),
        "semantic_vs_no_gemma_channel": (
            "semantic",
            "no_gemma_channel",
        ),
        "no_gemma_channel_vs_aapl": (
            "no_gemma_channel",
            "aapl_buy_and_hold",
        ),
        "no_gemma_channel_vs_baseline": (
            "no_gemma_channel",
            "baseline",
        ),
    }
    window_metrics: dict[str, Any] = {}
    for valuation in VALUATION_METHODS:
        by_cost: dict[str, Any] = {}
        for cost_key in COST_KEYS:
            by_comparison: dict[str, Any] = {}
            for comparison_id, (left_name, right_name) in comparisons.items():
                by_comparison[comparison_id] = {
                    window_id: _window_comparison(
                        ledgers[left_name][cost_key],
                        ledgers[right_name][cost_key],
                        comparison_id=comparison_id,
                        first=first,
                        last=last,
                        valuation_method=valuation,
                    )
                    for window_id, (first, last) in windows.items()
                }
            by_cost[cost_key] = by_comparison
        window_metrics[valuation] = by_cost

    frozen_metrics: dict[str, Any] = {}
    if stage == "development":
        for valuation in VALUATION_METHODS:
            frozen_metrics[valuation] = {}
            for cost_key in COST_KEYS:
                frozen_metrics[valuation][cost_key] = {
                    block_id: _window_comparison(
                        ledgers["semantic"][cost_key],
                        validated_input["frozen_control_ledgers"][
                            block_id
                        ][cost_key],
                        comparison_id=(
                            f"semantic_vs_block_frozen:{block_id}"
                        ),
                        first=first,
                        last=last,
                        valuation_method=valuation,
                    )
                    for block_id, first, last in DEVELOPMENT_BLOCKS
                }
    else:
        control_id = FROZEN_CONTROL_IDS[stage][0]
        total_id = (
            "confirmation_total"
            if stage == "confirmation"
            else "final_continuous"
        )
        first, last = windows[total_id]
        for valuation in VALUATION_METHODS:
            frozen_metrics[valuation] = {}
            for cost_key in COST_KEYS:
                frozen_metrics[valuation][cost_key] = {
                    total_id: _window_comparison(
                        ledgers["semantic"][cost_key],
                        validated_input["frozen_control_ledgers"][
                            control_id
                        ][cost_key],
                        comparison_id=f"semantic_vs_frozen:{control_id}",
                        first=first,
                        last=last,
                        valuation_method=valuation,
                    )
                }
    return window_metrics, frozen_metrics


def _calendar_and_block_diagnostics(
    *,
    stage: str,
    window_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    calendar_ids = (
        [f"year_{year}" for year in range(2005, 2019)]
        if stage == "development"
        else [f"year_{year}" for year in range(2019, 2024)]
        if stage == "confirmation"
        else ["2024", "2025", "2026_ytd"]
    )
    calendar: dict[str, Any] = {}
    for valuation in VALUATION_METHODS:
        calendar[valuation] = {}
        for cost in COST_KEYS:
            rows: list[dict[str, Any]] = []
            for window_id in calendar_ids:
                active = window_metrics[valuation][cost][
                    "semantic_vs_aapl"
                ][window_id]
                baseline = window_metrics[valuation][cost][
                    "semantic_vs_baseline"
                ][window_id]
                ablation = window_metrics[valuation][cost][
                    "semantic_vs_no_filing_meaning"
                ][window_id]
                diagnostic = window_metrics[valuation][cost][
                    "semantic_vs_no_gemma_channel"
                ][window_id]
                aapl_return = _decode_hex(
                    active["right_log_return_hex"],
                    "calendar AAPL return",
                )
                active_edge = _decode_hex(
                    active["log_edge_hex"], "calendar active edge"
                )
                body = {
                    "window_id": window_id,
                    "first_physical_session": active[
                        "first_physical_session"
                    ],
                    "last_physical_session": active[
                        "last_physical_session"
                    ],
                    "semantic_log_return_hex": active[
                        "left_log_return_hex"
                    ],
                    "aapl_log_return_hex": active[
                        "right_log_return_hex"
                    ],
                    "active_log_edge_hex": active["log_edge_hex"],
                    "incremental_vs_baseline_log_edge_hex": baseline[
                        "log_edge_hex"
                    ],
                    "incremental_vs_no_filing_meaning_log_edge_hex": (
                        ablation["log_edge_hex"]
                    ),
                    "incremental_vs_no_gemma_channel_log_edge_hex": (
                        diagnostic["log_edge_hex"]
                    ),
                    "negative_aapl": (
                        aapl_return < -POSITIVE_EDGE_TOLERANCE
                    ),
                    "active_edge_strictly_positive": (
                        active_edge > POSITIVE_EDGE_TOLERANCE
                    ),
                }
                rows.append(_seal(body, "calendar_row_sha256"))
            negative_rows = [
                row for row in rows if row["negative_aapl"]
            ]
            body = {
                "rows": rows,
                "rows_sha256": canonical_sha256(rows),
                "window_count": len(rows),
                "positive_active_window_count": sum(
                    row["active_edge_strictly_positive"] for row in rows
                ),
                "negative_aapl_window_count": len(negative_rows),
                "positive_active_negative_aapl_window_count": sum(
                    row["active_edge_strictly_positive"]
                    for row in negative_rows
                ),
            }
            calendar[valuation][cost] = _seal(
                body, "calendar_slice_sha256"
            )
    calendar_body = {
        "schema_version": CALENDAR_DIAGNOSTICS_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "kind": "calendar_years_or_final_periods",
        "by_valuation_and_cost": calendar,
        "by_valuation_and_cost_sha256": canonical_sha256(calendar),
    }
    calendar_report = _seal(
        calendar_body, "calendar_diagnostics_sha256"
    )

    if stage != "development":
        return calendar_report, None
    blocks: dict[str, Any] = {}
    for valuation in VALUATION_METHODS:
        blocks[valuation] = {}
        for cost in COST_KEYS:
            rows = []
            for block_id, _, _ in DEVELOPMENT_BLOCKS:
                active = window_metrics[valuation][cost][
                    "semantic_vs_aapl"
                ][block_id]
                baseline = window_metrics[valuation][cost][
                    "semantic_vs_baseline"
                ][block_id]
                ablation = window_metrics[valuation][cost][
                    "semantic_vs_no_filing_meaning"
                ][block_id]
                diagnostic = window_metrics[valuation][cost][
                    "semantic_vs_no_gemma_channel"
                ][block_id]
                body = {
                    "block_id": block_id,
                    "first_physical_session": active[
                        "first_physical_session"
                    ],
                    "last_physical_session": active[
                        "last_physical_session"
                    ],
                    "active_log_edge_hex": active["log_edge_hex"],
                    "incremental_vs_baseline_log_edge_hex": baseline[
                        "log_edge_hex"
                    ],
                    "incremental_vs_no_filing_meaning_log_edge_hex": (
                        ablation["log_edge_hex"]
                    ),
                    "incremental_vs_no_gemma_channel_log_edge_hex": (
                        diagnostic["log_edge_hex"]
                    ),
                    "active_edge_strictly_positive": (
                        _decode_hex(
                            active["log_edge_hex"],
                            "development block edge",
                        )
                        > POSITIVE_EDGE_TOLERANCE
                    ),
                }
                rows.append(_seal(body, "block_diagnostic_row_sha256"))
            slice_body = {
                "rows": rows,
                "rows_sha256": canonical_sha256(rows),
                "block_count": len(rows),
                "positive_active_block_count": sum(
                    row["active_edge_strictly_positive"] for row in rows
                ),
            }
            blocks[valuation][cost] = _seal(
                slice_body, "block_diagnostic_slice_sha256"
            )
    block_body = {
        "schema_version": CALENDAR_DIAGNOSTICS_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "kind": "development_blocks",
        "by_valuation_and_cost": blocks,
        "by_valuation_and_cost_sha256": canonical_sha256(blocks),
    }
    return calendar_report, _seal(
        block_body, "calendar_diagnostics_sha256"
    )


def _action_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {row["accession_number"]: row for row in rows}


def _action_differences(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    comparison_id: str,
    first: str,
    last: str,
    cutoff_session: str,
) -> dict[str, Any]:
    left = _action_map(left_rows)
    right = _action_map(right_rows)
    if set(left) != set(right):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            f"{comparison_id} action supports differ"
        )
    differences: list[dict[str, Any]] = []
    for accession in sorted(
        left,
        key=lambda item: (
            left[item]["decision_session"],
            ""
            if left[item]["acceptance_datetime"] is None
            else left[item]["acceptance_datetime"],
            item,
        ),
    ):
        left_row = left[accession]
        right_row = right[accession]
        if any(
            left_row[field] != right_row[field]
            for field in (
                "decision_session",
                "acceptance_datetime",
                "prediction_available",
            )
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"{comparison_id} action identity differs"
            )
        session = left_row["decision_session"]
        if (
            not first <= session <= last
            or session >= cutoff_session
            or not left_row["prediction_available"]
        ):
            continue
        if (
            left_row["schedule_overlay"]
            == right_row["schedule_overlay"]
        ):
            continue
        block_id = next(
            (
                block
                for block, block_first, block_last in DEVELOPMENT_BLOCKS
                if block_first <= session <= block_last
            ),
            None,
        )
        body = {
            "accession_number": accession,
            "decision_session": session,
            "left_policy_action_sha256": left_row[
                "policy_action_sha256"
            ],
            "right_policy_action_sha256": right_row[
                "policy_action_sha256"
            ],
            "left_schedule_overlay": left_row["schedule_overlay"],
            "right_schedule_overlay": right_row["schedule_overlay"],
            "development_block": block_id,
            "calendar_year": int(session[:4]),
        }
        differences.append(_seal(body, "action_difference_sha256"))
    body = {
        "schema_version": ACTION_DIFFERENCE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "comparison_id": comparison_id,
        "first_decision_session": first,
        "last_decision_session": last,
        "pending_action_cutoff_session": cutoff_session,
        "differences": differences,
        "differences_sha256": canonical_sha256(differences),
        "action_difference_count": len(differences),
        "difference_development_blocks": sorted(
            {
                row["development_block"]
                for row in differences
                if row["development_block"] is not None
            }
        ),
        "difference_development_block_count": len(
            {
                row["development_block"]
                for row in differences
                if row["development_block"] is not None
            }
        ),
        "difference_calendar_years": sorted(
            {row["calendar_year"] for row in differences}
        ),
        "difference_calendar_year_count": len(
            {row["calendar_year"] for row in differences}
        ),
    }
    return _seal(body, "action_differences_sha256")


def _development_frozen_action_differences(
    validated_input: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    semantic = validated_input["semantic_policy_actions"]
    for block_id, first, last in DEVELOPMENT_BLOCKS:
        report = _action_differences(
            semantic,
            validated_input["frozen_policy_actions"][block_id],
            comparison_id=f"semantic_vs_block_frozen:{block_id}",
            first=first,
            last=last,
            cutoff_session=STAGE_CUTOFFS["development"],
        )
        rows.extend(report["differences"])
    if len({row["accession_number"] for row in rows}) != len(rows):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "development frozen action differences overlap blocks"
        )
    body = {
        "schema_version": ACTION_DIFFERENCE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "comparison_id": "semantic_vs_block_frozen",
        "first_decision_session": PERFORMANCE_WINDOWS["development"][0],
        "last_decision_session": PERFORMANCE_WINDOWS["development"][1],
        "pending_action_cutoff_session": STAGE_CUTOFFS["development"],
        "differences": rows,
        "differences_sha256": canonical_sha256(rows),
        "action_difference_count": len(rows),
        "difference_development_blocks": sorted(
            {
                row["development_block"]
                for row in rows
                if row["development_block"] is not None
            }
        ),
        "difference_development_block_count": len(
            {
                row["development_block"]
                for row in rows
                if row["development_block"] is not None
            }
        ),
        "difference_calendar_years": sorted(
            {row["calendar_year"] for row in rows}
        ),
        "difference_calendar_year_count": len(
            {row["calendar_year"] for row in rows}
        ),
    }
    return _seal(body, "action_differences_sha256")


def _log_factor_edge(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
) -> float:
    return math.fsum(
        math.log(
            _decode_hex(
                left_rows[position]["period_factor_hex"],
                "left attribution factor",
                positive=True,
            )
        )
        - math.log(
            _decode_hex(
                right_rows[position]["period_factor_hex"],
                "right attribution factor",
                positive=True,
            )
        )
        for position in positions
    )


def _contribution_stats(
    complete_contributions: Sequence[float],
) -> dict[str, Any]:
    positive = [
        value
        for value in complete_contributions
        if value > POSITIVE_EDGE_TOLERANCE
    ]
    win_rate = (
        None
        if not complete_contributions
        else len(positive) / len(complete_contributions)
    )
    median_edge = (
        None
        if not complete_contributions
        else float(median(complete_contributions))
    )
    positive_sum = math.fsum(positive)
    concentration = (
        None
        if not positive
        else max(positive) / positive_sum
    )
    best_positive = None if not positive else max(positive)
    return {
        "complete_count": len(complete_contributions),
        "strictly_positive_count": len(positive),
        "win_rate_hex": _optional_hex(win_rate, "contribution win rate"),
        "median_contribution_hex": _optional_hex(
            median_edge, "median contribution"
        ),
        "positive_contribution_sum_hex": _float_hex(
            positive_sum, "positive contribution sum"
        ),
        "largest_positive_contribution_hex": _optional_hex(
            best_positive, "largest positive contribution"
        ),
        "largest_positive_share_hex": _optional_hex(
            concentration, "largest positive share"
        ),
    }


def _xor_attribution(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison_id: str,
    first: str,
    last: str,
) -> dict[str, Any]:
    left_rows, right_rows = _aligned_rows(left, right)
    window_positions = _window_positions(
        left_rows,
        first=first,
        last=last,
        location=f"XOR {comparison_id}",
    )
    first_position = window_positions[0]
    last_position = window_positions[-1]
    intervals: list[dict[str, Any]] = []
    position = first_position
    while position <= last_position:
        different = (
            left_rows[position]["target_exposure"]
            != right_rows[position]["target_exposure"]
        )
        if not different:
            position += 1
            continue
        carried = (
            position == first_position
            and position > 0
            and left_rows[position - 1]["target_exposure"]
            != right_rows[position - 1]["target_exposure"]
        )
        start = position
        end_equal: int | None = None
        position += 1
        while position <= last_position:
            if (
                left_rows[position]["target_exposure"]
                == right_rows[position]["target_exposure"]
            ):
                end_equal = position
                break
            position += 1
        attribution_end = (
            last_position if end_equal is None else end_equal
        )
        contribution_positions = list(range(start, attribution_end + 1))
        contribution = _log_factor_edge(
            left_rows, right_rows, contribution_positions
        )
        complete = not carried and end_equal is not None
        body = {
            "xor_ordinal": len(intervals) + 1,
            "start_session": left_rows[start]["session"],
            "return_to_equality_session": (
                None
                if end_equal is None
                else left_rows[end_equal]["session"]
            ),
            "started_inside_window": not carried,
            "returned_to_equality_inside_window": end_equal is not None,
            "complete": complete,
            "assigned_row_count": len(contribution_positions),
            "contribution_hex": _float_hex(
                contribution, "XOR contribution"
            ),
        }
        intervals.append(_seal(body, "xor_interval_sha256"))
        if end_equal is None:
            break
        position = end_equal + 1
    complete_contributions = [
        _decode_hex(row["contribution_hex"], "complete XOR contribution")
        for row in intervals
        if row["complete"]
    ]
    body = {
        "schema_version": XOR_ATTRIBUTION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "comparison_id": comparison_id,
        "first_session": first,
        "last_session": last,
        "left_ledger_sha256": left["ledger_sha256"],
        "right_ledger_sha256": right["ledger_sha256"],
        "intervals": intervals,
        "intervals_sha256": canonical_sha256(intervals),
        **_contribution_stats(complete_contributions),
    }
    return _seal(body, "xor_attribution_sha256")


def _episode_attribution(
    semantic: Mapping[str, Any],
    baseline: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    *,
    first: str,
    last: str,
) -> dict[str, Any]:
    semantic_rows, baseline_rows = _aligned_rows(semantic, baseline)
    sessions = [row["session"] for row in semantic_rows]
    positions = {session: index for index, session in enumerate(sessions)}
    attributed: list[dict[str, Any]] = []
    for episode in episodes:
        entry = episode["entry_session"]
        exit_ = episode["exit_session"]
        complete = bool(
            entry is not None
            and exit_ is not None
            and first <= entry <= last
            and first <= exit_ <= last
        )
        contribution: float | None = None
        assigned_count = 0
        if complete:
            if entry not in positions or exit_ not in positions:
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    "complete episode boundary is absent from ledger"
                )
            start_position = positions[entry]
            exit_position = positions[exit_]
            if (
                episode["entry_position"] != start_position
                or episode["exit_position"] != exit_position
                or exit_position <= start_position
            ):
                raise SecGemmaOnlineRiskOverlayMetricsError(
                    "episode positions differ from ledger boundaries"
                )
            contribution_positions = list(
                range(start_position, exit_position + 1)
            )
            assigned_count = len(contribution_positions)
            contribution = _log_factor_edge(
                semantic_rows,
                baseline_rows,
                contribution_positions,
            )
        body = {
            "overlay_schedule_sha256": episode[
                "overlay_schedule_sha256"
            ],
            "accession_number": episode["accession_number"],
            "decision_session": episode["decision_session"],
            "entry_session": entry,
            "exit_session": exit_,
            "complete_inside_window": complete,
            "assigned_row_count": assigned_count,
            "contribution_hex": _optional_hex(
                contribution, "episode contribution"
            ),
        }
        attributed.append(_seal(body, "episode_attribution_sha256"))
    complete_contributions = [
        _decode_hex(
            row["contribution_hex"], "complete episode contribution"
        )
        for row in attributed
        if row["complete_inside_window"]
    ]
    body = {
        "schema_version": EPISODE_ATTRIBUTION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "first_session": first,
        "last_session": last,
        "semantic_ledger_sha256": semantic["ledger_sha256"],
        "baseline_ledger_sha256": baseline["ledger_sha256"],
        "episodes": attributed,
        "episodes_sha256": canonical_sha256(attributed),
        **_contribution_stats(complete_contributions),
    }
    return _seal(body, "episode_attribution_sha256")


def _coverage_metrics(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    first, last = COVERAGE_WINDOWS[stage]
    rows = [
        row
        for row in feature_rows
        if first <= row["decision_session"] <= last
    ]
    denominator = len(rows)
    valid = sum(bool(row["schema_valid_extraction"]) for row in rows)
    nonzero = sum(bool(row["meaning_nonzero"]) for row in rows)
    rate = None if denominator == 0 else valid / denominator
    body = {
        "schema_version": COVERAGE_METRICS_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "first_decision_session": first,
        "last_decision_session": last,
        "eligible_call_count": denominator,
        "schema_valid_extraction_count": valid,
        "schema_valid_extraction_rate_hex": _optional_hex(
            rate, "schema-valid extraction rate"
        ),
        "nonzero_filing_meaning_row_count": nonzero,
        "included_feature_row_sha256s": [
            row["feature_row_sha256"] for row in rows
        ],
        "included_feature_rows_sha256": canonical_sha256(
            [row["feature_row_sha256"] for row in rows]
        ),
    }
    return _seal(body, "coverage_metrics_sha256")


def _brier_metrics(
    *,
    stage: str,
    semantic_predictions: Sequence[Mapping[str, Any]],
    no_meaning_predictions: Sequence[Mapping[str, Any]],
    no_gemma_predictions: Sequence[Mapping[str, Any]],
    lessons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first, last = PERFORMANCE_WINDOWS[stage]
    cutoff = STAGE_CUTOFFS[stage]
    semantic = _action_map(semantic_predictions)
    ablation = _action_map(no_meaning_predictions)
    diagnostic = _action_map(no_gemma_predictions)
    lesson_by_accession = {
        row["accession_number"]: row for row in lessons
    }
    rows: list[dict[str, Any]] = []
    semantic_errors: list[float] = []
    ablation_errors: list[float] = []
    diagnostic_errors: list[float] = []
    for accession in sorted(semantic):
        sem = semantic[accession]
        abl = ablation[accession]
        no_gemma = diagnostic[accession]
        decision = sem["decision_session"]
        lesson = lesson_by_accession.get(accession)
        if (
            not first <= decision <= last
            or not sem["fitted_prediction_available"]
            or not abl["fitted_prediction_available"]
            or not no_gemma["fitted_prediction_available"]
            or lesson is None
            or lesson["maturity_session"] > cutoff
            or not lesson["trainable"]
        ):
            continue
        sem_probability = _decode_hex(
            sem["gate_audit"]["probability_hex"],
            "semantic Brier probability",
        )
        abl_probability = _decode_hex(
            abl["gate_audit"]["probability_hex"],
            "ablation Brier probability",
        )
        diagnostic_probability = _decode_hex(
            no_gemma["gate_audit"]["probability_hex"],
            "no-Gemma Brier probability",
        )
        target = _binary(
            lesson["binary_cash_win_target"], "Brier binary target"
        )
        sem_error = (sem_probability - target) ** 2
        abl_error = (abl_probability - target) ** 2
        diagnostic_error = (diagnostic_probability - target) ** 2
        semantic_errors.append(sem_error)
        ablation_errors.append(abl_error)
        diagnostic_errors.append(diagnostic_error)
        body = {
            "accession_number": accession,
            "decision_session": decision,
            "maturity_session": lesson["maturity_session"],
            "semantic_prediction_row_sha256": sem[
                "prediction_row_sha256"
            ],
            "ablation_prediction_row_sha256": abl[
                "prediction_row_sha256"
            ],
            "no_gemma_prediction_row_sha256": no_gemma[
                "prediction_row_sha256"
            ],
            "lesson_row_sha256": lesson["lesson_row_sha256"],
            "target": target,
            "semantic_probability_hex": sem_probability.hex(),
            "ablation_probability_hex": abl_probability.hex(),
            "no_gemma_probability_hex": diagnostic_probability.hex(),
            "semantic_squared_error_hex": sem_error.hex(),
            "ablation_squared_error_hex": abl_error.hex(),
            "no_gemma_squared_error_hex": diagnostic_error.hex(),
        }
        rows.append(_seal(body, "brier_row_sha256"))
    support = len(rows)
    semantic_brier = (
        None if not support else math.fsum(semantic_errors) / support
    )
    ablation_brier = (
        None if not support else math.fsum(ablation_errors) / support
    )
    diagnostic_brier = (
        None if not support else math.fsum(diagnostic_errors) / support
    )
    relative_improvement = (
        None
        if semantic_brier is None or ablation_brier is None
        else (
            (ablation_brier - semantic_brier)
            / max(ablation_brier, POSITIVE_EDGE_TOLERANCE)
        )
    )
    body = {
        "schema_version": BRIER_METRICS_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "first_decision_session": first,
        "last_decision_session": last,
        "maturity_cutoff_session": cutoff,
        "support_count": support,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
        "semantic_brier_score_hex": _optional_hex(
            semantic_brier, "semantic Brier score"
        ),
        "ablation_brier_score_hex": _optional_hex(
            ablation_brier, "ablation Brier score"
        ),
        "no_gemma_brier_score_hex": _optional_hex(
            diagnostic_brier, "no-Gemma Brier score"
        ),
        "relative_improvement_hex": _optional_hex(
            relative_improvement, "Brier relative improvement"
        ),
    }
    return _seal(body, "brier_metrics_sha256")


def build_stage_metrics(
    metrics_input: Mapping[str, Any],
    *,
    expected_stage_metrics_input_sha256: str,
) -> dict[str, Any]:
    """Compute one exact stage metric artifact from sealed continuous inputs."""

    value = _validate_stage_metrics_input(
        metrics_input,
        expected_stage_metrics_input_sha256=(
            expected_stage_metrics_input_sha256
        ),
    )
    stage = value["stage"]
    first, last = PERFORMANCE_WINDOWS[stage]
    window_metrics, frozen_window_metrics = _build_window_metrics(value)
    calendar_diagnostics, block_diagnostics = (
        _calendar_and_block_diagnostics(
            stage=stage,
            window_metrics=window_metrics,
        )
    )
    semantic_vs_no_meaning_actions = _action_differences(
        value["semantic_policy_actions"],
        value["no_filing_meaning_policy_actions"],
        comparison_id="semantic_vs_no_filing_meaning",
        first=first,
        last=last,
        cutoff_session=STAGE_CUTOFFS[stage],
    )
    semantic_vs_no_gemma_actions = _action_differences(
        value["semantic_policy_actions"],
        value["no_gemma_channel_policy_actions"],
        comparison_id="semantic_vs_no_gemma_channel",
        first=first,
        last=last,
        cutoff_session=STAGE_CUTOFFS[stage],
    )
    if stage == "development":
        online_vs_frozen_actions = (
            _development_frozen_action_differences(value)
        )
    else:
        control_id = FROZEN_CONTROL_IDS[stage][0]
        online_vs_frozen_actions = _action_differences(
            value["semantic_policy_actions"],
            value["frozen_policy_actions"][control_id],
            comparison_id=f"semantic_vs_frozen:{control_id}",
            first=first,
            last=last,
            cutoff_session=STAGE_CUTOFFS[stage],
        )
    xor_attribution = _xor_attribution(
        value["ledgers"]["semantic"]["cost_10bps"],
        value["ledgers"]["no_filing_meaning"]["cost_10bps"],
        comparison_id="semantic_vs_no_filing_meaning",
        first=first,
        last=last,
    )
    no_gemma_xor_attribution = _xor_attribution(
        value["ledgers"]["semantic"]["cost_10bps"],
        value["ledgers"]["no_gemma_channel"]["cost_10bps"],
        comparison_id="semantic_vs_no_gemma_channel",
        first=first,
        last=last,
    )
    episode_attribution = _episode_attribution(
        value["ledgers"]["semantic"]["cost_10bps"],
        value["ledgers"]["baseline"]["cost_10bps"],
        value["semantic_overlay_episodes"],
        first=first,
        last=last,
    )
    coverage = _coverage_metrics(value["feature_rows"], stage=stage)
    brier = _brier_metrics(
        stage=stage,
        semantic_predictions=value["semantic_predictions"],
        no_meaning_predictions=value[
            "no_filing_meaning_predictions"
        ],
        no_gemma_predictions=value["no_gemma_channel_predictions"],
        lessons=value["learner_lessons"],
    )
    body = {
        "schema_version": STAGE_METRICS_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "stage_metrics_input_sha256": value[
            "stage_metrics_input_sha256"
        ],
        "performance_first_session": first,
        "performance_last_session": last,
        "valuation_methods": list(VALUATION_METHODS),
        "window_metrics": window_metrics,
        "window_metrics_sha256": canonical_sha256(window_metrics),
        "frozen_window_metrics": frozen_window_metrics,
        "frozen_window_metrics_sha256": canonical_sha256(
            frozen_window_metrics
        ),
        "calendar_diagnostics": calendar_diagnostics,
        "calendar_diagnostics_sha256": calendar_diagnostics[
            "calendar_diagnostics_sha256"
        ],
        "development_block_diagnostics": block_diagnostics,
        "development_block_diagnostics_sha256": (
            None
            if block_diagnostics is None
            else block_diagnostics["calendar_diagnostics_sha256"]
        ),
        "semantic_vs_no_filing_meaning_action_differences": (
            semantic_vs_no_meaning_actions
        ),
        "semantic_vs_no_filing_meaning_action_differences_sha256": (
            semantic_vs_no_meaning_actions["action_differences_sha256"]
        ),
        "semantic_vs_no_gemma_channel_action_differences": (
            semantic_vs_no_gemma_actions
        ),
        "semantic_vs_no_gemma_channel_action_differences_sha256": (
            semantic_vs_no_gemma_actions["action_differences_sha256"]
        ),
        "online_vs_frozen_action_differences": (
            online_vs_frozen_actions
        ),
        "online_vs_frozen_action_differences_sha256": (
            online_vs_frozen_actions["action_differences_sha256"]
        ),
        "semantic_vs_no_filing_meaning_xor_attribution_10bps": (
            xor_attribution
        ),
        "semantic_vs_no_filing_meaning_xor_attribution_10bps_sha256": (
            xor_attribution["xor_attribution_sha256"]
        ),
        "semantic_vs_no_gemma_channel_xor_attribution_10bps": (
            no_gemma_xor_attribution
        ),
        "semantic_vs_no_gemma_channel_xor_attribution_10bps_sha256": (
            no_gemma_xor_attribution["xor_attribution_sha256"]
        ),
        "semantic_overlay_episode_attribution_10bps": (
            episode_attribution
        ),
        "semantic_overlay_episode_attribution_10bps_sha256": (
            episode_attribution["episode_attribution_sha256"]
        ),
        "coverage_metrics": coverage,
        "coverage_metrics_sha256": coverage["coverage_metrics_sha256"],
        "brier_metrics": brier,
        "brier_metrics_sha256": brier["brier_metrics_sha256"],
    }
    return _seal(body, "stage_metrics_sha256")


def _metric_edge(
    metrics: Mapping[str, Any],
    *,
    valuation: str,
    cost: str,
    comparison: str,
    window: str,
) -> float:
    try:
        value = metrics["window_metrics"][valuation][cost][comparison][
            window
        ]["log_edge_hex"]
    except (KeyError, TypeError) as exc:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "required window metric is missing"
        ) from exc
    return _decode_hex(value, "window edge")


def _frozen_edge(
    metrics: Mapping[str, Any],
    *,
    valuation: str,
    cost: str,
    window: str,
) -> float:
    try:
        value = metrics["frozen_window_metrics"][valuation][cost][window][
            "log_edge_hex"
        ]
    except (KeyError, TypeError) as exc:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "required frozen metric is missing"
        ) from exc
    return _decode_hex(value, "frozen edge")


def _both_valuations(predicate: Any) -> bool:
    return all(bool(predicate(valuation)) for valuation in VALUATION_METHODS)


def _positive(value: float) -> bool:
    return value > POSITIVE_EDGE_TOLERANCE


def _optional_metric(value: Any, location: str) -> float | None:
    return None if value is None else _decode_hex(value, location)


def _development_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    blocks = [item[0] for item in DEVELOPMENT_BLOCKS]
    years = [f"year_{year}" for year in range(2005, 2019)]

    def total(valuation: str, comparison: str) -> float:
        return _metric_edge(
            metrics,
            valuation=valuation,
            cost="cost_10bps",
            comparison=comparison,
            window="development_total",
        )

    def block_edges(valuation: str, comparison: str) -> list[float]:
        return [
            _metric_edge(
                metrics,
                valuation=valuation,
                cost="cost_10bps",
                comparison=comparison,
                window=block,
            )
            for block in blocks
        ]

    def year_edges(valuation: str, comparison: str) -> list[float]:
        return [
            _metric_edge(
                metrics,
                valuation=valuation,
                cost="cost_10bps",
                comparison=comparison,
                window=year,
            )
            for year in years
        ]

    qualification_valuation = "adjusted_open"
    active_years = year_edges(
        qualification_valuation, "semantic_vs_aapl"
    )
    aapl_years = [
        _decode_hex(
            metrics["window_metrics"][qualification_valuation][
                "cost_10bps"
            ]["semantic_vs_aapl"][year]["right_log_return_hex"],
            "AAPL year return",
        )
        for year in years
    ]
    annual_rate = (
        sum(_positive(edge) for edge in active_years) / len(active_years)
    )
    negative_positions = [
        index
        for index, value in enumerate(aapl_years)
        if value < -POSITIVE_EDGE_TOLERANCE
    ]
    negative_year_rate = (
        None
        if not negative_positions
        else sum(
            _positive(active_years[index])
            for index in negative_positions
        )
        / len(negative_positions)
    )

    episode = metrics["semantic_overlay_episode_attribution_10bps"]
    xor = metrics[
        "semantic_vs_no_filing_meaning_xor_attribution_10bps"
    ]
    coverage = metrics["coverage_metrics"]
    brier = metrics["brier_metrics"]
    semantic_actions = metrics[
        "semantic_vs_no_filing_meaning_action_differences"
    ]
    frozen_actions = metrics["online_vs_frozen_action_differences"]
    episode_win_rate = _optional_metric(
        episode["win_rate_hex"], "episode win rate"
    )
    episode_median = _optional_metric(
        episode["median_contribution_hex"], "episode median"
    )
    episode_share = _optional_metric(
        episode["largest_positive_share_hex"],
        "episode concentration",
    )
    coverage_rate = _optional_metric(
        coverage["schema_valid_extraction_rate_hex"],
        "coverage rate",
    )
    brier_improvement = _optional_metric(
        brier["relative_improvement_hex"], "Brier improvement"
    )
    best_xor = _optional_metric(
        xor["largest_positive_contribution_hex"], "best XOR"
    )
    checks = {
        "combined_total_active_log_edge_10bps_at_least": (
            total(qualification_valuation, "semantic_vs_aapl") >= 0.02
        ),
        "combined_edge_without_best_block_10bps_at_least": (
            total(qualification_valuation, "semantic_vs_aapl")
            - max(
                block_edges(
                    qualification_valuation, "semantic_vs_aapl"
                )
            )
            >= 0.005
        ),
        "positive_combined_blocks_10bps_at_least": (
            sum(
                _positive(value)
                for value in block_edges(
                    qualification_valuation, "semantic_vs_aapl"
                )
            )
            >= 4
        ),
        "annual_win_rate_10bps_at_least": annual_rate >= 0.55,
        "negative_aapl_year_win_rate_10bps_at_least": (
            negative_year_rate is not None
            and negative_year_rate >= 0.60
        ),
        "complete_sec_overlay_episodes_at_least": (
            episode["complete_count"] >= 12
        ),
        "overlay_episode_win_rate_10bps_at_least": (
            episode_win_rate is not None
            and episode_win_rate >= 0.55
        ),
        "overlay_median_edge_10bps_strictly_positive": (
            episode_median is not None and _positive(episode_median)
        ),
        "largest_positive_episode_share_10bps_at_most": (
            episode_share is not None and episode_share <= 0.35
        ),
        "schema_valid_extraction_rate_at_least": (
            coverage_rate is not None and coverage_rate >= 0.90
        ),
        "nonzero_filing_meaning_rows_at_least": (
            coverage["nonzero_filing_meaning_row_count"] >= 24
        ),
        "incremental_vs_baseline_10bps_at_least": (
            total(qualification_valuation, "semantic_vs_baseline")
            >= 0.005
        ),
        "incremental_vs_baseline_without_best_block_10bps_strictly_positive": (
            _positive(
                total(qualification_valuation, "semantic_vs_baseline")
                - max(
                    block_edges(
                        qualification_valuation,
                        "semantic_vs_baseline",
                    )
                )
            )
        ),
        "online_vs_block_frozen_action_differences_at_least": (
            frozen_actions["action_difference_count"] >= 5
        ),
        "online_vs_block_frozen_difference_blocks_at_least": (
            frozen_actions["difference_development_block_count"] >= 3
        ),
        "online_vs_block_frozen_10bps_edge_strictly_positive": (
            _positive(
                math.fsum(
                    _frozen_edge(
                        metrics,
                        valuation=qualification_valuation,
                        cost="cost_10bps",
                        window=block,
                    )
                    for block in blocks
                )
            )
        ),
        "semantic_vs_no_filing_meaning_action_differences_at_least": (
            semantic_actions["action_difference_count"] >= 5
        ),
        "semantic_vs_no_filing_meaning_difference_blocks_at_least": (
            semantic_actions["difference_development_block_count"] >= 3
        ),
        "semantic_vs_no_filing_meaning_complete_xor_intervals_at_least": (
            xor["complete_count"] >= 4
        ),
        "semantic_vs_no_filing_meaning_10bps_edge_at_least": (
            total(
                qualification_valuation,
                "semantic_vs_no_filing_meaning",
            )
            >= 0.005
        ),
        "semantic_edge_without_best_xor_10bps_strictly_positive": (
            best_xor is not None
            and _positive(
                total(
                    qualification_valuation,
                    "semantic_vs_no_filing_meaning",
                )
                - best_xor
            )
        ),
        "semantic_brier_relative_improvement_at_least": (
            brier_improvement is not None
            and brier_improvement >= 0.01
        ),
    }
    return checks


def _confirmation_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    years = [f"year_{year}" for year in range(2019, 2024)]
    qualification_valuation = "adjusted_open"

    def edge(
        valuation: str,
        cost: str,
        comparison: str,
        window: str = "confirmation_total",
    ) -> float:
        return _metric_edge(
            metrics,
            valuation=valuation,
            cost=cost,
            comparison=comparison,
            window=window,
        )

    episode = metrics["semantic_overlay_episode_attribution_10bps"]
    coverage = metrics["coverage_metrics"]
    semantic_actions = metrics[
        "semantic_vs_no_filing_meaning_action_differences"
    ]
    frozen_actions = metrics["online_vs_frozen_action_differences"]
    coverage_rate = _optional_metric(
        coverage["schema_valid_extraction_rate_hex"],
        "coverage rate",
    )
    best_episode = _optional_metric(
        episode["largest_positive_contribution_hex"], "best episode"
    )
    checks = {
        "combined_active_log_edge_positive_at_5_and_10bps": all(
            _positive(
                edge(
                    qualification_valuation,
                    cost,
                    "semantic_vs_aapl",
                )
            )
            for cost in COST_KEYS
        ),
        "combined_positive_years_at_least_3_at_both_5_and_10bps": all(
            sum(
                _positive(
                    edge(
                        qualification_valuation,
                        cost,
                        "semantic_vs_aapl",
                        year,
                    )
                )
                for year in years
            )
            >= 3
            for cost in COST_KEYS
        ),
        "schema_valid_extraction_rate_at_least": (
            coverage_rate is not None and coverage_rate >= 0.90
        ),
        "nonzero_filing_meaning_rows_at_least": (
            coverage["nonzero_filing_meaning_row_count"] >= 6
        ),
        "incremental_vs_baseline_10bps_at_least": (
            edge(
                qualification_valuation,
                "cost_10bps",
                "semantic_vs_baseline",
            )
            >= 0.0025
        ),
        "incremental_without_best_episode_10bps_strictly_positive": (
            best_episode is not None
            and _positive(
                edge(
                    qualification_valuation,
                    "cost_10bps",
                    "semantic_vs_baseline",
                )
                - best_episode
            )
        ),
        "semantic_vs_no_filing_meaning_action_differences_at_least": (
            semantic_actions["action_difference_count"] >= 3
        ),
        "semantic_difference_years_at_least": (
            semantic_actions["difference_calendar_year_count"] >= 2
        ),
        "semantic_vs_no_filing_meaning_10bps_edge_at_least": (
            edge(
                qualification_valuation,
                "cost_10bps",
                "semantic_vs_no_filing_meaning",
            )
            >= 0.0025
        ),
        "online_vs_frozen_action_differences_at_least": (
            frozen_actions["action_difference_count"] >= 2
        ),
        "online_vs_frozen_10bps_edge_strictly_positive": (
            _positive(
                _frozen_edge(
                    metrics,
                    valuation=qualification_valuation,
                    cost="cost_10bps",
                    window="confirmation_total",
                )
            )
        ),
    }
    return checks


def _final_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    periods = ("2024", "2025", "2026_ytd")

    def edge(
        valuation: str,
        cost: str,
        comparison: str,
        window: str,
    ) -> float:
        return _metric_edge(
            metrics,
            valuation=valuation,
            cost=cost,
            comparison=comparison,
            window=window,
        )

    episode = metrics["semantic_overlay_episode_attribution_10bps"]
    coverage = metrics["coverage_metrics"]
    semantic_actions = metrics[
        "semantic_vs_no_filing_meaning_action_differences"
    ]
    frozen_actions = metrics["online_vs_frozen_action_differences"]
    coverage_rate = _optional_metric(
        coverage["schema_valid_extraction_rate_hex"],
        "coverage rate",
    )
    episode_win_rate = _optional_metric(
        episode["win_rate_hex"], "episode win rate"
    )
    episode_share = _optional_metric(
        episode["largest_positive_share_hex"],
        "episode concentration",
    )

    drawdown_pass = True
    for valuation in VALUATION_METHODS:
        for cost in COST_KEYS:
            window = metrics["window_metrics"][valuation][cost][
                "semantic_vs_aapl"
            ]["final_continuous"]
            strategy_mdd = _decode_hex(
                window["left_maximum_drawdown_hex"],
                "strategy final MDD",
            )
            aapl_mdd = _decode_hex(
                window["right_maximum_drawdown_hex"],
                "AAPL final MDD",
            )
            drawdown_pass = drawdown_pass and (
                strategy_mdd >= aapl_mdd - 0.01
            )

    checks = {
        "active_edge_5bps_each_2024_2025_2026_ytd_at_least": all(
            edge(
                valuation,
                "cost_5bps",
                "semantic_vs_aapl",
                period,
            )
            >= 0.005
            for valuation in VALUATION_METHODS
            for period in periods
        ),
        "continuous_active_edge_5bps_at_least": (
            _both_valuations(
                lambda valuation: edge(
                    valuation,
                    "cost_5bps",
                    "semantic_vs_aapl",
                    "final_continuous",
                )
                >= 0.02
            )
        ),
        "active_edge_positive_each_period_at_10bps": all(
            _positive(
                edge(
                    valuation,
                    "cost_10bps",
                    "semantic_vs_aapl",
                    period,
                )
            )
            for valuation in VALUATION_METHODS
            for period in periods
        ),
        "schema_valid_extraction_rate_at_least": (
            coverage_rate is not None and coverage_rate >= 0.90
        ),
        "nonzero_filing_meaning_rows_at_least": (
            coverage["nonzero_filing_meaning_row_count"] >= 3
        ),
        "incremental_vs_baseline_continuous_10bps_strictly_positive": (
            _both_valuations(
                lambda valuation: _positive(
                    edge(
                        valuation,
                        "cost_10bps",
                        "semantic_vs_baseline",
                        "final_continuous",
                    )
                )
            )
        ),
        "incremental_vs_baseline_positive_periods_10bps_at_least": all(
            sum(
                _positive(
                    edge(
                        valuation,
                        "cost_10bps",
                        "semantic_vs_baseline",
                        period,
                    )
                )
                for period in periods
            )
            >= 2
            for valuation in VALUATION_METHODS
        ),
        "semantic_vs_no_filing_meaning_action_differences_at_least": (
            semantic_actions["action_difference_count"] >= 2
        ),
        "semantic_vs_no_filing_meaning_continuous_10bps_strictly_positive": (
            _both_valuations(
                lambda valuation: _positive(
                    edge(
                        valuation,
                        "cost_10bps",
                        "semantic_vs_no_filing_meaning",
                        "final_continuous",
                    )
                )
            )
        ),
        "online_vs_frozen_continuous_10bps_strictly_positive": (
            _both_valuations(
                lambda valuation: _positive(
                    _frozen_edge(
                        metrics,
                        valuation=valuation,
                        cost="cost_10bps",
                        window="final_continuous",
                    )
                )
            )
        ),
        "post_2023_online_vs_frozen_action_differences_at_least": (
            frozen_actions["action_difference_count"] >= 3
        ),
        "complete_sec_overlay_episodes_at_least": (
            episode["complete_count"] >= 6
        ),
        "overlay_episode_win_rate_10bps_at_least": (
            episode_win_rate is not None
            and episode_win_rate >= 0.55
        ),
        "largest_positive_episode_share_10bps_at_most": (
            episode_share is not None and episode_share <= 0.50
        ),
        "max_drawdown_not_worse_than_aapl_by_more_than_at_5_and_10bps": (
            drawdown_pass
        ),
        "terminal_valuation_methods_required": (
            metrics["valuation_methods"] == list(VALUATION_METHODS)
        ),
        "all_return_edge_and_drawdown_gates_pass_under_both_terminal_valuations": True,
        "undefined_metric_fails": True,
    }
    return checks


def _gate_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    stage = metrics["stage"]
    if stage == "development":
        return _development_checks(metrics)
    if stage == "confirmation":
        return _confirmation_checks(metrics)
    if stage == "final":
        return _final_checks(metrics)
    raise SecGemmaOnlineRiskOverlayMetricsError("unknown metric stage")


def build_stage_gate_report(
    metrics: Mapping[str, Any],
    *,
    expected_stage_metrics_sha256: str,
) -> dict[str, Any]:
    """Apply the exact frozen gate set to one validated metric artifact."""

    value = _validate_stage_metrics_structure(
        metrics,
        expected_stage_metrics_sha256=expected_stage_metrics_sha256,
    )
    checks = _gate_checks(value)
    gate_definitions = copy.deepcopy(
        build_contract_manifest()["gates"][value["stage"]]
    )
    if not checks or any(type(item) is not bool for item in checks.values()):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "gate checks are incomplete or non-Boolean"
        )
    if set(checks) != set(gate_definitions):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "gate checks differ from the literal preregistration"
        )
    failures = [name for name, passed in checks.items() if not passed]
    body = {
        "schema_version": GATE_REPORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": value["stage"],
        "stage_metrics_sha256": value["stage_metrics_sha256"],
        "gate_definitions": gate_definitions,
        "gate_definitions_sha256": canonical_sha256(gate_definitions),
        "checks": checks,
        "checks_sha256": canonical_sha256(checks),
        "passed": not failures,
        "failed_checks": failures,
        "failed_checks_sha256": canonical_sha256(failures),
    }
    return _seal(body, "gate_report_sha256")


def _validate_stage_metrics_structure(
    raw: Mapping[str, Any],
    *,
    expected_stage_metrics_sha256: str,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(_mapping(raw, "stage metrics")))
    expected_keys = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "stage",
        "stage_metrics_input_sha256",
        "performance_first_session",
        "performance_last_session",
        "valuation_methods",
        "window_metrics",
        "window_metrics_sha256",
        "frozen_window_metrics",
        "frozen_window_metrics_sha256",
        "calendar_diagnostics",
        "calendar_diagnostics_sha256",
        "development_block_diagnostics",
        "development_block_diagnostics_sha256",
        "semantic_vs_no_filing_meaning_action_differences",
        "semantic_vs_no_filing_meaning_action_differences_sha256",
        "semantic_vs_no_gemma_channel_action_differences",
        "semantic_vs_no_gemma_channel_action_differences_sha256",
        "online_vs_frozen_action_differences",
        "online_vs_frozen_action_differences_sha256",
        "semantic_vs_no_filing_meaning_xor_attribution_10bps",
        "semantic_vs_no_filing_meaning_xor_attribution_10bps_sha256",
        "semantic_vs_no_gemma_channel_xor_attribution_10bps",
        "semantic_vs_no_gemma_channel_xor_attribution_10bps_sha256",
        "semantic_overlay_episode_attribution_10bps",
        "semantic_overlay_episode_attribution_10bps_sha256",
        "coverage_metrics",
        "coverage_metrics_sha256",
        "brier_metrics",
        "brier_metrics_sha256",
        "stage_metrics_sha256",
    }
    if set(value) != expected_keys:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage metrics keys changed"
        )
    if (
        value["schema_version"] != STAGE_METRICS_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"] != CONTRACT_SHA256
        or value["stage"] not in STAGES
        or value["valuation_methods"] != list(VALUATION_METHODS)
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage metrics identity changed"
        )
    for field in (
        "stage_metrics_input_sha256",
        "window_metrics_sha256",
        "frozen_window_metrics_sha256",
        "calendar_diagnostics_sha256",
        "semantic_vs_no_filing_meaning_action_differences_sha256",
        "semantic_vs_no_gemma_channel_action_differences_sha256",
        "online_vs_frozen_action_differences_sha256",
        "semantic_vs_no_filing_meaning_xor_attribution_10bps_sha256",
        "semantic_vs_no_gemma_channel_xor_attribution_10bps_sha256",
        "semantic_overlay_episode_attribution_10bps_sha256",
        "coverage_metrics_sha256",
        "brier_metrics_sha256",
    ):
        _sha256(value[field], f"stage metrics.{field}")
    if value["stage"] == "development":
        _sha256(
            value["development_block_diagnostics_sha256"],
            "stage metrics.development_block_diagnostics_sha256",
        )
    elif (
        value["development_block_diagnostics"] is not None
        or value["development_block_diagnostics_sha256"] is not None
    ):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "non-development metrics expose block diagnostics"
        )
    direct_hash_fields = {
        "window_metrics": "window_metrics_sha256",
        "frozen_window_metrics": "frozen_window_metrics_sha256",
    }
    for payload, hash_field in direct_hash_fields.items():
        if value[hash_field] != canonical_sha256(value[payload]):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"stage metrics {payload} hash changed"
            )
    nested_bindings = {
        "semantic_vs_no_filing_meaning_action_differences": (
            "semantic_vs_no_filing_meaning_action_differences_sha256",
            "action_differences_sha256",
        ),
        "online_vs_frozen_action_differences": (
            "online_vs_frozen_action_differences_sha256",
            "action_differences_sha256",
        ),
        "semantic_vs_no_gemma_channel_action_differences": (
            "semantic_vs_no_gemma_channel_action_differences_sha256",
            "action_differences_sha256",
        ),
        "semantic_vs_no_filing_meaning_xor_attribution_10bps": (
            "semantic_vs_no_filing_meaning_xor_attribution_10bps_sha256",
            "xor_attribution_sha256",
        ),
        "semantic_overlay_episode_attribution_10bps": (
            "semantic_overlay_episode_attribution_10bps_sha256",
            "episode_attribution_sha256",
        ),
        "semantic_vs_no_gemma_channel_xor_attribution_10bps": (
            "semantic_vs_no_gemma_channel_xor_attribution_10bps_sha256",
            "xor_attribution_sha256",
        ),
        "coverage_metrics": (
            "coverage_metrics_sha256",
            "coverage_metrics_sha256",
        ),
        "brier_metrics": (
            "brier_metrics_sha256",
            "brier_metrics_sha256",
        ),
        "calendar_diagnostics": (
            "calendar_diagnostics_sha256",
            "calendar_diagnostics_sha256",
        ),
    }
    for payload, (outer_field, inner_field) in nested_bindings.items():
        inner = _mapping(value[payload], f"stage metrics.{payload}")
        if value[outer_field] != inner.get(inner_field):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                f"stage metrics {payload} binding changed"
            )
        _require_self_hash(
            inner,
            hash_field=inner_field,
            location=f"stage metrics.{payload}",
        )
    if value["stage"] == "development":
        block = _mapping(
            value["development_block_diagnostics"],
            "stage metrics.development_block_diagnostics",
        )
        if (
            value["development_block_diagnostics_sha256"]
            != block.get("calendar_diagnostics_sha256")
        ):
            raise SecGemmaOnlineRiskOverlayMetricsError(
                "development block diagnostic binding changed"
            )
        _require_self_hash(
            block,
            hash_field="calendar_diagnostics_sha256",
            location="stage metrics.development_block_diagnostics",
        )
    observed = _require_self_hash(
        value,
        hash_field="stage_metrics_sha256",
        location="stage metrics",
    )
    expected = _sha256(
        expected_stage_metrics_sha256, "expected stage metrics hash"
    )
    if not hmac.compare_digest(observed, expected):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage metrics are not externally pinned"
        )
    return value


def validate_stage_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_stage_metrics_sha256: str,
    metrics_input: Mapping[str, Any],
    expected_stage_metrics_input_sha256: str,
) -> str:
    """Recompute metrics from exact inputs and require byte identity."""

    expected = build_stage_metrics(
        metrics_input,
        expected_stage_metrics_input_sha256=(
            expected_stage_metrics_input_sha256
        ),
    )
    value = _validate_stage_metrics_structure(
        metrics,
        expected_stage_metrics_sha256=expected_stage_metrics_sha256,
    )
    if value != expected:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "stage metrics differ from deterministic reconstruction"
        )
    return value["stage_metrics_sha256"]


def validate_stage_gate_report(
    report: Mapping[str, Any],
    *,
    expected_gate_report_sha256: str,
    metrics: Mapping[str, Any],
    expected_stage_metrics_sha256: str,
) -> str:
    """Rebuild a gate report and reject missing, extra, or changed checks."""

    value = copy.deepcopy(dict(_mapping(report, "stage gate report")))
    expected_keys = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "stage",
        "stage_metrics_sha256",
        "gate_definitions",
        "gate_definitions_sha256",
        "checks",
        "checks_sha256",
        "passed",
        "failed_checks",
        "failed_checks_sha256",
        "gate_report_sha256",
    }
    if set(value) != expected_keys:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "gate report keys changed"
        )
    rebuilt = build_stage_gate_report(
        metrics,
        expected_stage_metrics_sha256=expected_stage_metrics_sha256,
    )
    if value != rebuilt:
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "gate report differs from deterministic reconstruction"
        )
    observed = _require_self_hash(
        value,
        hash_field="gate_report_sha256",
        location="stage gate report",
    )
    expected = _sha256(
        expected_gate_report_sha256, "expected gate report hash"
    )
    if not hmac.compare_digest(observed, expected):
        raise SecGemmaOnlineRiskOverlayMetricsError(
            "gate report is not externally pinned"
        )
    return observed


__all__ = [
    "ACTION_DIFFERENCE_SCHEMA_VERSION",
    "BRIER_METRICS_SCHEMA_VERSION",
    "CALENDAR_DIAGNOSTICS_SCHEMA_VERSION",
    "COVERAGE_METRICS_SCHEMA_VERSION",
    "EPISODE_ATTRIBUTION_SCHEMA_VERSION",
    "FROZEN_CONTROL_IDS",
    "GATE_REPORT_SCHEMA_VERSION",
    "METRICS_INPUT_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayMetricsError",
    "STAGE_METRICS_SCHEMA_VERSION",
    "VALUATION_METHODS",
    "WINDOW_COMPARISON_SCHEMA_VERSION",
    "XOR_ATTRIBUTION_SCHEMA_VERSION",
    "build_stage_gate_report",
    "build_stage_metrics",
    "build_stage_metrics_input",
    "validate_stage_gate_report",
    "validate_stage_metrics",
    "validate_stage_metrics_input",
]
