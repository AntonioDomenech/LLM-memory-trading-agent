"""Deterministic ledger replay and gates for the frozen SEC/Gemma approach.

The scorer consumes already validated and externally pinned prediction,
label-release, and market-stage artifacts.  It nevertheless rechecks every
canonical hash it relies on and independently rebuilds the LONG/CASH ledger
and the same-session AAPL buy-and-hold benchmark.  There is no filesystem,
network, model, clock, or protected-data I/O in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import hmac
import json
import math
import re
from statistics import median
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    ACTIVE_EDGE_TOLERANCE,
    BRIER_TARGET_COST_BPS,
    CANDIDATE_IDS,
    DEVELOPMENT_FOLD_SPECS,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    STAGE_ORDER,
    STAGE_WINDOWS,
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    MARKET_ROW_SCHEMA_VERSION,
    decode_float_hex as decode_market_float_hex,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
    MODEL_VARIANTS,
)


SCORE_RECEIPT_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-score-receipt-v1"
GATE_RECEIPT_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-gate-receipt-v1"
RANKING_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-ranking-v1"
)
INITIAL_CAPITAL: Final[float] = 1_000.0
TERMINAL_CONVENTIONS: Final[tuple[str, str]] = (
    "adjusted_open",
    "terminal_adjusted_close",
)
TRANSACTION_COST_BPS: Final[tuple[int, int]] = (5, 10)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT_HEX_RE = re.compile(
    r"-?0x(?:0(?:\.0+)?|[01](?:\.[0-9a-f]+)?)p[+-][0-9]+\Z"
)
_PERIODS_BY_STAGE: Final[dict[str, tuple[str, ...]]] = {
    "development": tuple(str(year) for year in range(2005, 2019)),
    "intermediate": tuple(str(year) for year in range(2019, 2024)),
    "final": ("2024", "2025", "2026_ytd"),
}


class SecFilingGemmaScoringError(SecFilingGemmaContractError):
    """Scoring evidence, ledger replay, or frozen gate logic failed closed."""


def _snapshot(value: Any, location: str) -> Any:
    def detach(item: Any, item_location: str) -> Any:
        if isinstance(item, Mapping):
            try:
                pairs = list(item.items())
            except Exception as exc:
                raise SecFilingGemmaScoringError(
                    f"{item_location} could not be detached"
                ) from exc
            result: dict[str, Any] = {}
            for key, child in pairs:
                if not isinstance(key, str) or key in result:
                    raise SecFilingGemmaScoringError(
                        f"{item_location} must have unique string keys"
                    )
                result[key] = detach(child, f"{item_location}.{key}")
            return result
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            try:
                values = list(item)
            except Exception as exc:
                raise SecFilingGemmaScoringError(
                    f"{item_location} could not be detached"
                ) from exc
            return [
                detach(child, f"{item_location}[{index}]")
                for index, child in enumerate(values)
            ]
        if item is None or type(item) in {str, bool, int, float}:
            return item
        raise SecFilingGemmaScoringError(
            f"{item_location} contains a non-JSON value"
        )

    detached = detach(value, location)
    try:
        encoded = json.dumps(
            detached,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaScoringError(
            f"{location} must be finite canonical JSON"
        ) from exc


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SecFilingGemmaScoringError(
            f"{location} must be a string-keyed mapping"
        )
    return value


def _keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaScoringError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaScoringError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_int(
    value: Any,
    location: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaScoringError(
            f"{location} must be an integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise SecFilingGemmaScoringError(f"{location} exceeds its maximum")
    return value


def _iso(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecFilingGemmaScoringError(f"{location} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecFilingGemmaScoringError(
            f"{location} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecFilingGemmaScoringError(
            f"{location} must use canonical YYYY-MM-DD form"
        )
    return value


def _finite(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SecFilingGemmaScoringError(f"{location} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise SecFilingGemmaScoringError(f"{location} must be finite")
    return number


def _float_hex(value: float) -> str:
    number = _finite(value, "floating-point output")
    if number == 0.0:
        number = 0.0
    return number.hex()


def decode_score_float_hex(value: Any, location: str = "float_hex") -> float:
    if not isinstance(value, str) or _FLOAT_HEX_RE.fullmatch(value) is None:
        raise SecFilingGemmaScoringError(
            f"{location} must be canonical finite float.hex"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecFilingGemmaScoringError(
            f"{location} must be canonical finite float.hex"
        ) from exc
    if not math.isfinite(number) or _float_hex(number) != value:
        raise SecFilingGemmaScoringError(
            f"{location} must be canonical finite float.hex"
        )
    return number


def _optional_hex(value: float | None) -> str | None:
    return None if value is None else _float_hex(value)


def _self_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    expected_hash: str,
    location: str,
) -> str:
    observed = _sha256(value.get(hash_field), f"{location}.{hash_field}")
    body = {key: value[key] for key in value if key != hash_field}
    calculated = canonical_sha256(body)
    if (
        not hmac.compare_digest(observed, calculated)
        or not hmac.compare_digest(
            observed, _sha256(expected_hash, f"expected {location} hash")
        )
    ):
        raise SecFilingGemmaScoringError(
            f"{location} is noncanonical or not externally pinned"
        )
    return observed


def _period_for_session(stage: str, session: str) -> str:
    year = session[:4]
    if stage == "final" and year == "2026":
        return "2026_ytd"
    return year


def _stage_start(stage: str) -> str:
    if stage == "development":
        return DEVELOPMENT_FOLD_SPECS[0][2]
    return STAGE_WINDOWS[stage][0]


def _cost_rate(cost_bps: int) -> float:
    if isinstance(cost_bps, bool) or cost_bps not in TRANSACTION_COST_BPS:
        raise SecFilingGemmaScoringError(
            "cost_bps must be the exact integer 5 or 10"
        )
    return cost_bps / 10_000.0


def _cash_round_trip_edge(entry: float, exit_value: float, cost_bps: int) -> float:
    rate = _cost_rate(cost_bps)
    return (
        math.log1p(-rate)
        - math.log1p(rate)
        - math.log(exit_value / entry)
    )


def _validated_market_stage(
    value: Mapping[str, Any],
    *,
    expected_hash: str,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[float, float]]]:
    market = _mapping(_snapshot(value, "market stage"), "market stage")
    market_hash = _self_hash(
        market,
        hash_field="market_stage_manifest_sha256",
        expected_hash=expected_hash,
        location="market stage",
    )
    if market.get("artifact_stage") != stage:
        raise SecFilingGemmaScoringError("Market rows belong to another stage")
    rows_value = market.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise SecFilingGemmaScoringError("Market stage rows cannot be empty")
    genesis = _sha256(
        market.get("row_chain_genesis_sha256"), "market row-chain genesis"
    )
    previous = genesis
    sessions: list[str] = []
    prices: dict[str, tuple[float, float]] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows_value):
        row = _mapping(raw, f"market rows[{index}]")
        if row.get("schema_version") != MARKET_ROW_SCHEMA_VERSION:
            raise SecFilingGemmaScoringError("Market row schema changed")
        if _strict_int(row.get("row_index"), "market row index") != index:
            raise SecFilingGemmaScoringError(
                "Market rows are missing, duplicated, or reordered"
            )
        session = _iso(row.get("session"), "market session")
        if sessions and session <= sessions[-1]:
            raise SecFilingGemmaScoringError(
                "Market sessions must be unique and strictly increasing"
            )
        if row.get("previous_row_sha256") != previous:
            raise SecFilingGemmaScoringError("Market row chain is broken")
        row_hash = _sha256(row.get("row_sha256"), "market row hash")
        row_body = {key: row[key] for key in row if key != "row_sha256"}
        if not hmac.compare_digest(row_hash, canonical_sha256(row_body)):
            raise SecFilingGemmaScoringError("Market row self-hash changed")
        observations = _mapping(row.get("observations"), "market observations")
        aapl = _mapping(observations.get("AAPL"), "AAPL observation")
        if aapl.get("available") is not True:
            raise SecFilingGemmaScoringError("AAPL must exist on every scoring session")
        adjusted_open = decode_market_float_hex(
            aapl.get("adjusted_open_hex"), f"{session} adjusted open"
        )
        adjusted_close = decode_market_float_hex(
            aapl.get("adjusted_close_hex"), f"{session} adjusted close"
        )
        if adjusted_open <= 0.0 or adjusted_close <= 0.0:
            raise SecFilingGemmaScoringError("AAPL adjusted prices must be positive")
        sessions.append(session)
        prices[session] = (adjusted_open, adjusted_close)
        normalized.append(dict(row))
        previous = row_hash
    if market.get("row_count") is not None and market["row_count"] != len(normalized):
        raise SecFilingGemmaScoringError("Market row count changed")
    if market.get("row_chain_tip_sha256") is not None and market[
        "row_chain_tip_sha256"
    ] != previous:
        raise SecFilingGemmaScoringError("Market row-chain tip changed")
    return (
        {
            "market_stage_manifest_sha256": market_hash,
            "row_chain_tip_sha256": previous,
            "row_count": len(normalized),
            "rows_sha256": canonical_sha256(normalized),
        },
        normalized,
        prices,
    )


def _policy_path_item(
    row: Mapping[str, Any], candidate_id: str, variant: str, field: str
) -> Mapping[str, Any] | str:
    container = _mapping(row.get(field), f"prediction {field}")
    candidates = _mapping(container.get(candidate_id), f"{field}.{candidate_id}")
    if variant not in candidates:
        raise SecFilingGemmaScoringError(
            f"Prediction omits {candidate_id}/{variant} from {field}"
        )
    return candidates[variant]


def _validated_prediction_prefix(
    value: Mapping[str, Any],
    *,
    expected_hash: str,
    candidate_id: str,
    variant: str,
    market_sessions: Sequence[str],
    score_cutoff_session: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    if candidate_id not in CANDIDATE_IDS:
        raise SecFilingGemmaScoringError("Unknown frozen candidate id")
    if variant not in MODEL_VARIANTS:
        raise SecFilingGemmaScoringError("Unknown semantic/ablation variant")
    prefix = _mapping(_snapshot(value, "prediction prefix"), "prediction prefix")
    prefix_hash = _self_hash(
        prefix,
        hash_field="prediction_prefix_sha256",
        expected_hash=expected_hash,
        location="prediction prefix",
    )
    rows_value = prefix.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise SecFilingGemmaScoringError("Prediction rows cannot be empty")
    if prefix.get("row_count") != len(rows_value):
        raise SecFilingGemmaScoringError("Prediction row count changed")
    if prefix.get("rows_sha256") != canonical_sha256(rows_value):
        raise SecFilingGemmaScoringError("Prediction rows hash changed")
    session_index = {session: index for index, session in enumerate(market_sessions)}
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, str]] = []
    prior_decision: str | None = None
    active: dict[str, str] | None = None
    for index, raw in enumerate(rows_value):
        row = _mapping(raw, f"prediction rows[{index}]")
        row_hash = _sha256(row.get("prediction_row_sha256"), "prediction row hash")
        row_body = {
            key: row[key] for key in row if key != "prediction_row_sha256"
        }
        if not hmac.compare_digest(row_hash, canonical_sha256(row_body)):
            raise SecFilingGemmaScoringError("Prediction row self-hash changed")
        if _strict_int(
            row.get("sequence_number"), "prediction sequence number", minimum=1
        ) != index + 1:
            raise SecFilingGemmaScoringError(
                "Prediction sequence is missing, duplicated, or reordered"
            )
        decision = _iso(row.get("decision_session"), "prediction decision session")
        fill = _iso(row.get("fill_session"), "prediction fill session")
        exit_session = _iso(row.get("cash_exit_session"), "prediction exit session")
        maturity = _iso(
            row.get("label_maturity_session"), "prediction label maturity"
        )
        if exit_session != maturity:
            raise SecFilingGemmaScoringError(
                "Cash exit and label maturity must be identical t+21"
            )
        if row.get("horizon_sessions") != HORIZON_SESSIONS:
            raise SecFilingGemmaScoringError("Prediction horizon changed from 20 sessions")
        if decision > score_cutoff_session:
            raise SecFilingGemmaScoringError(
                "Prediction prefix contains a decision after the score cutoff"
            )
        if prior_decision is not None and decision <= prior_decision:
            raise SecFilingGemmaScoringError(
                "Prediction decisions must be unique and chronological"
            )
        prior_decision = decision
        if decision not in session_index:
            raise SecFilingGemmaScoringError(
                "Prediction decision is absent from exact market sessions"
            )
        position = session_index[decision]
        if position + 1 < len(market_sessions):
            if fill != market_sessions[position + 1]:
                raise SecFilingGemmaScoringError("Prediction fill is not exactly t+1")
        elif not (fill > score_cutoff_session and decision == score_cutoff_session):
            raise SecFilingGemmaScoringError("Prediction fill is outside t+1 evidence")
        if position + LABEL_MATURITY_OFFSET < len(market_sessions):
            if exit_session != market_sessions[position + LABEL_MATURITY_OFFSET]:
                raise SecFilingGemmaScoringError(
                    "Prediction cash exit is not exactly t+21"
                )
        elif exit_session <= score_cutoff_session:
            raise SecFilingGemmaScoringError(
                "Prediction claims a mature t+21 outcome outside market rows"
            )
        if active is not None and decision >= active["exit_session"]:
            active = None
        action = _policy_path_item(
            row, candidate_id, variant, "effective_episode_actions"
        )
        if not isinstance(action, str):
            raise SecFilingGemmaScoringError("Effective episode action must be text")
        input_state = _policy_path_item(
            row, candidate_id, variant, "candidate_policy_input_states"
        )
        output_state = _policy_path_item(
            row, candidate_id, variant, "candidate_policy_output_states"
        )
        if not isinstance(input_state, Mapping) or not isinstance(output_state, Mapping):
            raise SecFilingGemmaScoringError("Policy states must be mappings")
        expected_input_position = "CASH" if active is not None else "LONG"
        if input_state.get("position_at_decision_close") != expected_input_position:
            raise SecFilingGemmaScoringError(
                "Policy input state does not match the continuous episode path"
            )
        if action == "START_CASH_EPISODE":
            if active is not None:
                raise SecFilingGemmaScoringError("An active episode was extended")
            active = {
                "origin_decision_session": decision,
                "fill_session": fill,
                "exit_session": exit_session,
                "prediction_row_sha256": row_hash,
            }
            episodes.append(dict(active))
        elif action == "HOLD_EXISTING_CASH_EPISODE":
            if active is None:
                raise SecFilingGemmaScoringError("Cash hold lacks an active episode")
        elif action == "STAY_LONG":
            if active is not None:
                raise SecFilingGemmaScoringError("LONG action truncates an active episode")
        else:
            raise SecFilingGemmaScoringError("Unknown effective episode action")
        expected_output_position = "CASH" if active is not None else "LONG"
        if output_state.get("position_at_decision_close") != expected_output_position:
            raise SecFilingGemmaScoringError(
                "Policy output state does not match the continuous episode path"
            )
        if active is not None:
            if (
                output_state.get("episode_origin_decision_session")
                != active["origin_decision_session"]
                or output_state.get("episode_fill_session") != active["fill_session"]
                or output_state.get("episode_exit_session") != active["exit_session"]
            ):
                raise SecFilingGemmaScoringError(
                    "Policy output state changed an episode's frozen dates"
                )
        rows.append(dict(row))
    if prefix.get("tip_sha256") is not None and prefix["tip_sha256"] != rows[-1][
        "prediction_row_sha256"
    ]:
        raise SecFilingGemmaScoringError("Prediction prefix tip changed")
    return (
        {
            "prediction_prefix_sha256": prefix_hash,
            "prediction_tip_sha256": rows[-1]["prediction_row_sha256"],
            "prediction_row_count": len(rows),
            "prediction_rows_sha256": canonical_sha256(rows),
        },
        rows,
        episodes,
    )


def _validated_label_release(
    value: Mapping[str, Any],
    *,
    expected_hash: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    prices: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ledger = _mapping(_snapshot(value, "label release ledger"), "label release ledger")
    ledger_hash = _self_hash(
        ledger,
        hash_field="label_release_ledger_sha256",
        expected_hash=expected_hash,
        location="label release ledger",
    )
    entries_value = ledger.get("entries")
    if not isinstance(entries_value, list):
        raise SecFilingGemmaScoringError("Label release entries must be a list")
    if ledger.get("release_count") != len(entries_value):
        raise SecFilingGemmaScoringError("Label release count changed")
    rows_by_hash = {
        row["prediction_row_sha256"]: row for row in prediction_rows
    }
    entries: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(entries_value):
        entry = _mapping(raw, f"label release entries[{index}]")
        release_hash = _sha256(entry.get("release_sha256"), "label release hash")
        body = {key: entry[key] for key in entry if key != "release_sha256"}
        if not hmac.compare_digest(release_hash, canonical_sha256(body)):
            raise SecFilingGemmaScoringError("Label release entry hash changed")
        row_hash = _sha256(
            entry.get("prediction_row_sha256"), "label prediction row hash"
        )
        if row_hash in entries or row_hash not in rows_by_hash:
            raise SecFilingGemmaScoringError(
                "Label release duplicates or references an unknown prediction"
            )
        row = rows_by_hash[row_hash]
        if (
            entry.get("decision_session") != row["decision_session"]
            or entry.get("label_maturity_session") != row["label_maturity_session"]
            or entry.get("cost_bps") != BRIER_TARGET_COST_BPS
        ):
            raise SecFilingGemmaScoringError(
                "Label release changed its decision, maturity, or 10-bps target"
            )
        edge = decode_score_float_hex(
            entry.get("cash_active_log_edge_10bps_hex"),
            "cash_active_log_edge_10bps_hex",
        )
        fill = row["fill_session"]
        exit_session = row["cash_exit_session"]
        if fill not in prices or exit_session not in prices:
            raise SecFilingGemmaScoringError(
                "A released label lacks its exact t+1/t+21 market opens"
            )
        expected_edge = _cash_round_trip_edge(
            prices[fill][0], prices[exit_session][0], BRIER_TARGET_COST_BPS
        )
        if _float_hex(edge) != _float_hex(expected_edge):
            raise SecFilingGemmaScoringError(
                "Released label does not reconcile to exact market opens and two costs"
            )
        expected_target = expected_edge > ACTIVE_EDGE_TOLERANCE
        if type(entry.get("cash_beats_long_10bps")) is not bool or entry[
            "cash_beats_long_10bps"
        ] is not expected_target:
            raise SecFilingGemmaScoringError(
                "Released binary label changed its strict 1e-12 boundary"
            )
        entries[row_hash] = dict(entry)
    return (
        {
            "label_release_ledger_sha256": ledger_hash,
            "label_release_count": len(entries),
            "label_release_entries_sha256": canonical_sha256(entries_value),
        },
        entries,
    )


def _target_by_session(
    sessions: Sequence[str], episodes: Sequence[Mapping[str, str]]
) -> dict[str, int]:
    targets = {session: 1 for session in sessions}
    for episode in episodes:
        fill = episode["fill_session"]
        exit_session = episode["exit_session"]
        if exit_session <= fill:
            raise SecFilingGemmaScoringError("Cash episode has a nonpositive horizon")
        for session in sessions:
            if fill <= session < exit_session:
                if targets[session] == 0:
                    raise SecFilingGemmaScoringError("Cash episodes overlap")
                targets[session] = 0
    if any(type(value) is not int or value not in {0, 1} for value in targets.values()):
        raise SecFilingGemmaScoringError("Target exposure is not exactly binary")
    return targets


def _stage_boundary_exposures(
    *,
    stage: str,
    score_start: str,
    episodes: Sequence[Mapping[str, str]],
) -> tuple[int, int]:
    """Return positions immediately before the first scored opening fill.

    Development is the one experiment genesis, so both ledgers begin in cash
    and establish their positions at the first scored open.  Later stages are
    slices of the same cumulative policy path: buy-and-hold is already LONG,
    while the strategy can be CASH only when a previously opened episode is
    still active immediately before the boundary open.  A fill exactly on the
    boundary is therefore charged inside the new stage rather than silently
    treated as an inherited position.
    """

    if stage == "development":
        return 0, 0
    if stage not in STAGE_ORDER:
        raise SecFilingGemmaScoringError("Unknown stage-boundary exposure stage")
    active_at_boundary = [
        episode
        for episode in episodes
        if episode["fill_session"] < score_start <= episode["exit_session"]
    ]
    if len(active_at_boundary) > 1:
        raise SecFilingGemmaScoringError(
            "Multiple cash episodes overlap the stage boundary"
        )
    return (0 if active_at_boundary else 1), 1


def _drawdown(values: Sequence[float]) -> float:
    peak = -math.inf
    maximum = 0.0
    for value in values:
        number = _finite(value, "wealth curve")
        if number <= 0.0:
            raise SecFilingGemmaScoringError("Wealth must remain strictly positive")
        peak = max(peak, number)
        maximum = max(maximum, 1.0 - number / peak)
    return maximum


def _rolling_month_end(
    sessions: Sequence[str], active: Sequence[float], window: int
) -> tuple[float | None, int]:
    if len(sessions) != len(active):
        raise SecFilingGemmaScoringError("Rolling inputs are not aligned")
    final_by_month: dict[str, float] = {}
    for index in range(window - 1, len(active)):
        final_by_month[sessions[index][:7]] = math.fsum(
            active[index - window + 1 : index + 1]
        )
    if not final_by_month:
        return None, 0
    wins = sum(value > ACTIVE_EDGE_TOLERANCE for value in final_by_month.values())
    return wins / len(final_by_month), len(final_by_month)


def _fold_edges(stage: str, sessions: Sequence[str], active: Sequence[float]) -> dict[str, float]:
    if stage != "development":
        return {}
    result: dict[str, float] = {}
    for fold_id, _, first, last in DEVELOPMENT_FOLD_SPECS:
        values = [
            edge
            for session, edge in zip(sessions, active, strict=True)
            if first <= session <= last
        ]
        result[fold_id] = math.fsum(values)
    return result


def _simulate_ledgers(
    *,
    sessions: Sequence[str],
    prices: Mapping[str, tuple[float, float]],
    targets: Mapping[str, int],
    cost_bps: int,
    terminal_convention: str,
    initial_strategy_exposure: int,
    initial_benchmark_exposure: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if terminal_convention not in TERMINAL_CONVENTIONS:
        raise SecFilingGemmaScoringError("Unknown terminal valuation convention")
    rate = _cost_rate(cost_bps)
    strategy_wealth = INITIAL_CAPITAL
    benchmark_wealth = INITIAL_CAPITAL
    if type(initial_strategy_exposure) is not int or initial_strategy_exposure not in {
        0,
        1,
    }:
        raise SecFilingGemmaScoringError(
            "Initial strategy exposure must be exactly 0 or 1"
        )
    if type(initial_benchmark_exposure) is not int or initial_benchmark_exposure not in {
        0,
        1,
    }:
        raise SecFilingGemmaScoringError(
            "Initial benchmark exposure must be exactly 0 or 1"
        )
    strategy_exposure = initial_strategy_exposure
    benchmark_exposure = initial_benchmark_exposure
    previous_open: float | None = None
    previous_strategy_wealth = INITIAL_CAPITAL
    previous_benchmark_wealth = INITIAL_CAPITAL
    genesis = canonical_sha256(
        {
            "domain": "aapl-sec-gemma-continuous-ledger-genesis-v1",
            "initial_capital_hex": _float_hex(INITIAL_CAPITAL),
            "cost_bps": cost_bps,
            "terminal_convention": terminal_convention,
            "initial_strategy_exposure": initial_strategy_exposure,
            "initial_benchmark_exposure": initial_benchmark_exposure,
        }
    )
    parent = genesis
    ledger: list[dict[str, Any]] = []
    strategy_curve = [INITIAL_CAPITAL]
    benchmark_curve = [INITIAL_CAPITAL]
    for index, session in enumerate(sessions):
        adjusted_open = prices[session][0]
        if previous_open is not None:
            if strategy_exposure == 1:
                strategy_wealth *= adjusted_open / previous_open
            if benchmark_exposure == 1:
                benchmark_wealth *= adjusted_open / previous_open
        holding_exposure = strategy_exposure
        target = targets[session]
        strategy_trade = target != strategy_exposure
        if strategy_trade:
            strategy_wealth *= 1.0 / (1.0 + rate) if target == 1 else 1.0 - rate
            strategy_exposure = target
        benchmark_trade = benchmark_exposure == 0
        if benchmark_trade:
            benchmark_wealth *= 1.0 / (1.0 + rate)
            benchmark_exposure = 1
        if strategy_exposure not in {0, 1} or benchmark_exposure != 1:
            raise SecFilingGemmaScoringError("Exposure left the exact set {0,1}")
        if strategy_wealth <= 0.0 or benchmark_wealth <= 0.0:
            raise SecFilingGemmaScoringError("Transaction costs exhausted wealth")
        strategy_cash = strategy_wealth if strategy_exposure == 0 else 0.0
        strategy_shares = (
            0.0 if strategy_exposure == 0 else strategy_wealth / adjusted_open
        )
        benchmark_cash = 0.0
        benchmark_shares = benchmark_wealth / adjusted_open
        if (
            strategy_cash < 0.0
            or benchmark_cash < 0.0
            or (strategy_exposure == 0 and strategy_shares != 0.0)
            or (strategy_exposure == 1 and strategy_cash != 0.0)
        ):
            raise SecFilingGemmaScoringError(
                "Cash/share invariants failed with zero tolerance"
            )
        strategy_log = math.log(strategy_wealth / previous_strategy_wealth)
        benchmark_log = math.log(benchmark_wealth / previous_benchmark_wealth)
        active_log = strategy_log - benchmark_log
        row_body = {
            "schema_version": "aapl-sec-gemma-ledger-row-v1",
            "row_index": index,
            "session": session,
            "period": session[:4],
            "adjusted_open_hex": _float_hex(adjusted_open),
            "holding_exposure_for_return": holding_exposure,
            "target_exposure": target,
            "strategy_position_changed": strategy_trade,
            "strategy_wealth_hex": _float_hex(strategy_wealth),
            "strategy_cash_hex": _float_hex(strategy_cash),
            "strategy_shares_hex": _float_hex(strategy_shares),
            "strategy_log_increment_hex": _float_hex(strategy_log),
            "benchmark_target_exposure": 1,
            "benchmark_position_changed": benchmark_trade,
            "benchmark_wealth_hex": _float_hex(benchmark_wealth),
            "benchmark_cash_hex": _float_hex(benchmark_cash),
            "benchmark_shares_hex": _float_hex(benchmark_shares),
            "benchmark_log_increment_hex": _float_hex(benchmark_log),
            "active_log_increment_hex": _float_hex(active_log),
            "margin_debt_hex": _float_hex(0.0),
            "previous_ledger_row_sha256": parent,
        }
        row_hash = canonical_sha256(row_body)
        ledger.append({**row_body, "ledger_row_sha256": row_hash})
        parent = row_hash
        strategy_curve.append(strategy_wealth)
        benchmark_curve.append(benchmark_wealth)
        previous_open = adjusted_open
        previous_strategy_wealth = strategy_wealth
        previous_benchmark_wealth = benchmark_wealth
    terminal_session = sessions[-1]
    terminal_open, terminal_close = prices[terminal_session]
    terminal_strategy_wealth = strategy_wealth
    terminal_benchmark_wealth = benchmark_wealth
    if terminal_convention == "terminal_adjusted_close":
        if strategy_exposure == 1:
            terminal_strategy_wealth *= terminal_close / terminal_open
        terminal_benchmark_wealth *= terminal_close / terminal_open
    terminal_strategy_log = math.log(terminal_strategy_wealth / strategy_wealth)
    terminal_benchmark_log = math.log(terminal_benchmark_wealth / benchmark_wealth)
    terminal_active_log = terminal_strategy_log - terminal_benchmark_log
    if terminal_convention == "terminal_adjusted_close":
        strategy_curve.append(terminal_strategy_wealth)
        benchmark_curve.append(terminal_benchmark_wealth)
    terminal = {
        "session": terminal_session,
        "convention": terminal_convention,
        "valuation_price_hex": _float_hex(
            terminal_open
            if terminal_convention == "adjusted_open"
            else terminal_close
        ),
        "strategy_wealth_hex": _float_hex(terminal_strategy_wealth),
        "benchmark_wealth_hex": _float_hex(terminal_benchmark_wealth),
        "strategy_log_increment_hex": _float_hex(terminal_strategy_log),
        "benchmark_log_increment_hex": _float_hex(terminal_benchmark_log),
        "active_log_increment_hex": _float_hex(terminal_active_log),
        "strategy_max_drawdown_hex": _float_hex(_drawdown(strategy_curve)),
        "benchmark_max_drawdown_hex": _float_hex(_drawdown(benchmark_curve)),
        "terminal_mark_sha256": "",
    }
    terminal_body = {
        key: terminal[key] for key in terminal if key != "terminal_mark_sha256"
    }
    terminal["terminal_mark_sha256"] = canonical_sha256(terminal_body)
    return ledger, {
        "ledger_genesis_sha256": genesis,
        "ledger_tip_sha256": parent,
        "terminal": terminal,
    }


def _probability(row: Mapping[str, Any], variant: str) -> float:
    value = decode_score_float_hex(
        row.get(f"{variant}_cash_probability_hex"),
        f"{variant}_cash_probability_hex",
    )
    if not 0.0 <= value <= 1.0:
        raise SecFilingGemmaScoringError("Cash probability lies outside [0,1]")
    return value


def _brier_metrics(
    *,
    stage: str,
    score_start: str,
    score_cutoff: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    selected_variant: str,
) -> dict[str, Any]:
    rows_by_hash = {
        row["prediction_row_sha256"]: row for row in prediction_rows
    }
    eligible = [
        row
        for row in prediction_rows
        if score_start <= row["decision_session"] <= score_cutoff
        and row.get("prediction_status") == AVAILABLE_PREDICTION_STATUS
        and row["label_maturity_session"] <= score_cutoff
        and row["prediction_row_sha256"] in labels
    ]
    model_errors: list[float] = []
    ablation_errors: list[float] = []
    climatology_errors: list[float] = []
    records: list[dict[str, Any]] = []
    for row in eligible:
        fold = _mapping(row.get("fold_context"), "prediction fold context")
        train_cutoff = _iso(
            fold.get("fold_train_cutoff_session"), "fold train cutoff"
        )
        training_labels = [
            labels[training_hash]["cash_beats_long_10bps"]
            for training_hash, training_row in rows_by_hash.items()
            if training_hash in labels
            and training_row.get("prediction_status")
            == AVAILABLE_PREDICTION_STATUS
            and training_row["label_maturity_session"] <= train_cutoff
        ]
        positives = sum(training_labels)
        climatology = (positives + 1.0) / (len(training_labels) + 2.0)
        target = 1.0 if labels[row["prediction_row_sha256"]][
            "cash_beats_long_10bps"
        ] else 0.0
        model_probability = _probability(row, selected_variant)
        ablation_probability = _probability(row, "ablation")
        model_errors.append((model_probability - target) ** 2)
        ablation_errors.append((ablation_probability - target) ** 2)
        climatology_errors.append((climatology - target) ** 2)
        records.append(
            {
                "prediction_row_sha256": row["prediction_row_sha256"],
                "decision_session": row["decision_session"],
                "label_maturity_session": row["label_maturity_session"],
                "fold_id": row.get("fold_id"),
                "target": int(target),
                "model_probability_hex": _float_hex(model_probability),
                "ablation_probability_hex": _float_hex(ablation_probability),
                "causal_climatology_probability_hex": _float_hex(climatology),
            }
        )
    if not eligible:
        return {
            "row_count": 0,
            "brier_score_hex": None,
            "causal_climatology_brier_score_hex": None,
            "ablation_brier_score_hex": None,
            "relative_improvement_vs_climatology_hex": None,
            "relative_improvement_vs_ablation_hex": None,
            "records": [],
            "records_sha256": canonical_sha256([]),
        }
    model_brier = math.fsum(model_errors) / len(model_errors)
    ablation_brier = math.fsum(ablation_errors) / len(ablation_errors)
    climatology_brier = math.fsum(climatology_errors) / len(climatology_errors)
    relative_climatology = (
        (climatology_brier - model_brier) / climatology_brier
        if climatology_brier > 0.0
        else None
    )
    relative_ablation = (
        (ablation_brier - model_brier) / ablation_brier
        if ablation_brier > 0.0
        else None
    )
    return {
        "row_count": len(records),
        "brier_score_hex": _float_hex(model_brier),
        "causal_climatology_brier_score_hex": _float_hex(climatology_brier),
        "ablation_brier_score_hex": _float_hex(ablation_brier),
        "relative_improvement_vs_climatology_hex": _optional_hex(
            relative_climatology
        ),
        "relative_improvement_vs_ablation_hex": _optional_hex(
            relative_ablation
        ),
        "records": records,
        "records_sha256": canonical_sha256(records),
    }


def _score_metrics(
    *,
    stage: str,
    score_start: str,
    score_cutoff: str,
    cost_bps: int,
    terminal_convention: str,
    ledger: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any],
    episodes: Sequence[Mapping[str, str]],
    prices: Mapping[str, tuple[float, float]],
    labels: Mapping[str, Mapping[str, Any]],
    brier: Mapping[str, Any],
) -> dict[str, Any]:
    sessions = [row["session"] for row in ledger]
    active = [
        decode_score_float_hex(row["active_log_increment_hex"], "active increment")
        for row in ledger
    ]
    benchmark = [
        decode_score_float_hex(
            row["benchmark_log_increment_hex"], "benchmark increment"
        )
        for row in ledger
    ]
    terminal_active = decode_score_float_hex(
        terminal["active_log_increment_hex"], "terminal active increment"
    )
    terminal_benchmark = decode_score_float_hex(
        terminal["benchmark_log_increment_hex"], "terminal benchmark increment"
    )
    active_with_terminal = list(active)
    benchmark_with_terminal = list(benchmark)
    active_with_terminal[-1] += terminal_active
    benchmark_with_terminal[-1] += terminal_benchmark
    active_by_session = dict(
        zip(sessions, active_with_terminal, strict=True)
    )
    periods = _PERIODS_BY_STAGE[stage]
    annual_active = {
        period: math.fsum(
            edge
            for session, edge in zip(sessions, active_with_terminal, strict=True)
            if _period_for_session(stage, session) == period
        )
        for period in periods
    }
    annual_benchmark = {
        period: math.fsum(
            edge
            for session, edge in zip(sessions, benchmark_with_terminal, strict=True)
            if _period_for_session(stage, session) == period
        )
        for period in periods
    }
    total_active = math.fsum(active_with_terminal)
    annual_values = list(annual_active.values())
    best_year = max(annual_values) if annual_values else 0.0
    positive_years = [value for value in annual_values if value > ACTIVE_EDGE_TOLERANCE]
    positive_sum = math.fsum(positive_years)
    largest_year_share = (
        max(positive_years) / positive_sum if positive_sum > 0.0 else None
    )
    rolling_252, rolling_252_n = _rolling_month_end(sessions, active, 252)
    rolling_756, rolling_756_n = _rolling_month_end(sessions, active, 756)
    targets = [row["target_exposure"] for row in ledger]
    if any(type(value) is not int or value not in {0, 1} for value in targets):
        raise SecFilingGemmaScoringError("Ledger target exposure is not exact {0,1}")
    cash_days = sum(value == 0 for value in targets)
    origin_episodes = [
        dict(episode)
        for episode in episodes
        if score_start <= episode["fill_session"] <= score_cutoff
    ]
    contributing_episodes = [
        dict(episode)
        for episode in episodes
        if episode["fill_session"] <= score_cutoff
        and episode["exit_session"] >= score_start
    ]
    episode_records: list[dict[str, Any]] = []
    completed_origin_edges: list[float] = []
    stage_contribution_edges: list[float] = []
    for episode in contributing_episodes:
        origin_in_score_window = score_start <= episode["fill_session"] <= score_cutoff
        completed = episode["exit_session"] <= score_cutoff
        contribution_start = max(score_start, episode["fill_session"])
        contribution_end = min(score_cutoff, episode["exit_session"])
        stage_edge = math.fsum(
            active_by_session[session]
            for session in sessions
            if contribution_start <= session <= contribution_end
        )
        stage_contribution_edges.append(stage_edge)
        full_horizon_edge: float | None = None
        won: bool | None = None
        if completed:
            if episode["fill_session"] not in prices or episode["exit_session"] not in prices:
                raise SecFilingGemmaScoringError(
                    "Completed episode lacks exact entry/exit market opens"
                )
            full_horizon_edge = _cash_round_trip_edge(
                prices[episode["fill_session"]][0],
                prices[episode["exit_session"]][0],
                cost_bps,
            )
            if origin_in_score_window:
                label = labels.get(episode["prediction_row_sha256"])
                if label is None:
                    raise SecFilingGemmaScoringError(
                        "A completed in-window policy episode lacks released label evidence"
                    )
                completed_origin_edges.append(full_horizon_edge)
                won = bool(label["cash_beats_long_10bps"])
        episode_records.append(
            {
                **episode,
                "period": _period_for_session(stage, episode["fill_session"]),
                "origin_in_score_window": origin_in_score_window,
                "status": "completed" if completed else "open_at_cutoff",
                "stage_contribution_start_session": contribution_start,
                "stage_contribution_end_session": contribution_end,
                "active_log_edge_hex": _float_hex(stage_edge),
                "stage_active_log_edge_hex": _float_hex(stage_edge),
                "full_horizon_active_log_edge_hex": _optional_hex(
                    full_horizon_edge
                ),
                "eligible_for_mature_episode_statistics": (
                    origin_in_score_window and completed
                ),
                "win": won,
            }
        )
    positive_episode_edges = [
        value
        for value in stage_contribution_edges
        if value > ACTIVE_EDGE_TOLERANCE
    ]
    positive_episode_sum = math.fsum(positive_episode_edges)
    largest_episode_share = (
        max(positive_episode_edges) / positive_episode_sum
        if positive_episode_sum > 0.0
        else None
    )
    attributed_episode_edge = math.fsum(stage_contribution_edges)
    unattributed_active_edge = total_active - attributed_episode_edge
    if abs(unattributed_active_edge) > ACTIVE_EDGE_TOLERANCE:
        raise SecFilingGemmaScoringError(
            "Scored active edge is not fully attributable to exact cash episodes"
        )
    negative_periods = [
        period
        for period in periods
        if annual_benchmark[period] < -ACTIVE_EDGE_TOLERANCE
    ]
    negative_edges = {period: annual_active[period] for period in negative_periods}
    negative_win_rate = (
        sum(value > ACTIVE_EDGE_TOLERANCE for value in negative_edges.values())
        / len(negative_edges)
        if negative_edges
        else None
    )
    fold_edges = _fold_edges(stage, sessions, active_with_terminal)
    strategy_drawdown = decode_score_float_hex(
        terminal["strategy_max_drawdown_hex"], "strategy drawdown"
    )
    benchmark_drawdown = decode_score_float_hex(
        terminal["benchmark_max_drawdown_hex"], "benchmark drawdown"
    )
    period_episode_counts = {
        period: sum(
            record["origin_in_score_window"] and record["period"] == period
            for record in episode_records
        )
        for period in periods
    }
    return {
        "comparison_tolerance_hex": _float_hex(ACTIVE_EDGE_TOLERANCE),
        "session_count": len(sessions),
        "total_active_log_edge_hex": _float_hex(total_active),
        "relative_wealth_vs_buy_hold_hex": _float_hex(math.expm1(total_active)),
        "period_active_log_edges_hex": {
            key: _float_hex(value) for key, value in annual_active.items()
        },
        "period_buy_hold_log_returns_hex": {
            key: _float_hex(value) for key, value in annual_benchmark.items()
        },
        "annual_win_rate_hex": _float_hex(
            sum(value > ACTIVE_EDGE_TOLERANCE for value in annual_values)
            / len(annual_values)
        ),
        "winning_year_count": sum(
            value > ACTIVE_EDGE_TOLERANCE for value in annual_values
        ),
        "median_annual_active_log_edge_hex": _float_hex(median(annual_values)),
        "best_annual_active_log_edge_hex": _float_hex(best_year),
        "active_log_edge_without_best_year_hex": _float_hex(
            total_active - best_year
        ),
        "largest_positive_year_share_hex": _optional_hex(largest_year_share),
        "fold_active_log_edges_hex": {
            key: _float_hex(value) for key, value in fold_edges.items()
        },
        "positive_fold_count": sum(
            value > ACTIVE_EDGE_TOLERANCE for value in fold_edges.values()
        ),
        "rolling_252_session_month_end_win_rate_hex": _optional_hex(rolling_252),
        "rolling_252_session_month_end_observations": rolling_252_n,
        "rolling_756_session_month_end_win_rate_hex": _optional_hex(rolling_756),
        "rolling_756_session_month_end_observations": rolling_756_n,
        "cash_days": cash_days,
        "cash_episodes": len(origin_episodes),
        "contributing_cash_episodes": len(episode_records),
        "cash_day_rate_hex": _float_hex(cash_days / len(targets)),
        "period_cash_episode_counts": period_episode_counts,
        "completed_cash_episodes": len(completed_origin_edges),
        "open_cash_episodes_at_cutoff": sum(
            record["origin_in_score_window"]
            and record["status"] == "open_at_cutoff"
            for record in episode_records
        ),
        "open_contributing_cash_episodes_at_cutoff": sum(
            record["status"] == "open_at_cutoff" for record in episode_records
        ),
        "episode_win_rate_hex": _optional_hex(
            sum(value > ACTIVE_EDGE_TOLERANCE for value in completed_origin_edges)
            / len(completed_origin_edges)
            if completed_origin_edges
            else None
        ),
        "mean_episode_active_log_edge_hex": _optional_hex(
            math.fsum(completed_origin_edges) / len(completed_origin_edges)
            if completed_origin_edges
            else None
        ),
        "maximum_single_episode_positive_edge_share_hex": _optional_hex(
            largest_episode_share
        ),
        "episode_records": episode_records,
        "episode_records_sha256": canonical_sha256(episode_records),
        "attributed_episode_active_log_edge_hex": _float_hex(
            attributed_episode_edge
        ),
        "unattributed_active_log_edge_hex": _float_hex(
            unattributed_active_edge
        ),
        "negative_buy_hold_periods": negative_periods,
        "negative_buy_hold_period_active_log_edges_hex": {
            key: _float_hex(value) for key, value in negative_edges.items()
        },
        "aggregate_active_log_edge_in_negative_buy_hold_periods_hex": _float_hex(
            math.fsum(negative_edges.values())
        ),
        "negative_buy_hold_period_win_rate_hex": _optional_hex(negative_win_rate),
        "strategy_max_drawdown_hex": _float_hex(strategy_drawdown),
        "benchmark_max_drawdown_hex": _float_hex(benchmark_drawdown),
        "drawdown_disadvantage_percentage_points_hex": _float_hex(
            (strategy_drawdown - benchmark_drawdown) * 100.0
        ),
        "brier": dict(brier),
        "strict_invariants": {
            "permitted_target_exposures": [0, 1],
            "proof_tolerance_hex": _float_hex(0.0),
            "fractional_exposure_observed": False,
            "short_exposure_observed": False,
            "leverage_observed": False,
            "negative_cash_observed": False,
            "margin_debt_observed": False,
            "same_market_rows_as_benchmark": True,
            "same_adjusted_open_prices_as_benchmark": True,
        },
    }


def build_score_receipt(
    *,
    prediction_prefix: Mapping[str, Any],
    expected_prediction_prefix_sha256: str,
    label_release_evidence: Mapping[str, Any],
    expected_label_release_ledger_sha256: str,
    market_stage: Mapping[str, Any],
    expected_market_stage_manifest_sha256: str,
    selected_candidate_id: str,
    selected_variant: str,
    cost_bps: int,
    stage: str,
    score_cutoff_session: str,
    terminal_convention: str,
) -> dict[str, Any]:
    """Rebuild one candidate/variant/cost/terminal ledger and score receipt."""

    if stage not in STAGE_ORDER:
        raise SecFilingGemmaScoringError("Unknown scoring stage")
    cutoff = _iso(score_cutoff_session, "score cutoff session")
    if not (STAGE_WINDOWS[stage][0] <= cutoff <= STAGE_WINDOWS[stage][1]):
        raise SecFilingGemmaScoringError("Score cutoff lies outside its stage")
    if terminal_convention not in TERMINAL_CONVENTIONS:
        raise SecFilingGemmaScoringError("Unknown terminal convention")
    _cost_rate(cost_bps)
    market_binding, market_rows, prices = _validated_market_stage(
        market_stage,
        expected_hash=expected_market_stage_manifest_sha256,
        stage=stage,
    )
    all_sessions = [row["session"] for row in market_rows]
    if cutoff not in prices:
        raise SecFilingGemmaScoringError("Score cutoff is not an exact market session")
    if all_sessions[-1] != cutoff:
        raise SecFilingGemmaScoringError(
            "Market stage contains a session after the score cutoff"
        )
    score_start = _stage_start(stage)
    score_sessions = [
        session for session in all_sessions if score_start <= session <= cutoff
    ]
    if not score_sessions:
        raise SecFilingGemmaScoringError("No exact sessions exist in the score window")
    prediction_binding, prediction_rows, episodes = _validated_prediction_prefix(
        prediction_prefix,
        expected_hash=expected_prediction_prefix_sha256,
        candidate_id=selected_candidate_id,
        variant=selected_variant,
        market_sessions=all_sessions,
        score_cutoff_session=cutoff,
    )
    label_binding, labels = _validated_label_release(
        label_release_evidence,
        expected_hash=expected_label_release_ledger_sha256,
        prediction_rows=prediction_rows,
        prices=prices,
    )
    targets = _target_by_session(score_sessions, episodes)
    initial_strategy_exposure, initial_benchmark_exposure = (
        _stage_boundary_exposures(
            stage=stage,
            score_start=score_sessions[0],
            episodes=episodes,
        )
    )
    ledger, ledger_summary = _simulate_ledgers(
        sessions=score_sessions,
        prices=prices,
        targets=targets,
        cost_bps=cost_bps,
        terminal_convention=terminal_convention,
        initial_strategy_exposure=initial_strategy_exposure,
        initial_benchmark_exposure=initial_benchmark_exposure,
    )
    brier = _brier_metrics(
        stage=stage,
        score_start=score_start,
        score_cutoff=cutoff,
        prediction_rows=prediction_rows,
        labels=labels,
        selected_variant=selected_variant,
    )
    metrics = _score_metrics(
        stage=stage,
        score_start=score_start,
        score_cutoff=cutoff,
        cost_bps=cost_bps,
        terminal_convention=terminal_convention,
        ledger=ledger,
        terminal=ledger_summary["terminal"],
        episodes=episodes,
        prices=prices,
        labels=labels,
        brier=brier,
    )
    body = {
        "schema_version": SCORE_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "inputs": {
            **prediction_binding,
            **label_binding,
            **market_binding,
        },
        "configuration": {
            "stage": stage,
            "score_start_session": score_sessions[0],
            "score_cutoff_session": cutoff,
            "terminal_convention": terminal_convention,
            "selected_candidate_id": selected_candidate_id,
            "selected_variant": selected_variant,
            "cost_bps": cost_bps,
            "initial_capital_hex": _float_hex(INITIAL_CAPITAL),
            "initial_strategy_exposure": initial_strategy_exposure,
            "initial_benchmark_exposure": initial_benchmark_exposure,
            "stage_boundary_position_semantics": (
                "development_genesis_then_cumulative_position_state"
            ),
            "fill_timing": "decision_t_close_fill_t_plus_1_adjusted_open",
            "cash_interval": "fill_inclusive_exit_exclusive_t_plus_1_to_t_plus_21",
            "cost_application": "each_position_changing_fill",
        },
        "ledger_genesis_sha256": ledger_summary["ledger_genesis_sha256"],
        "ledger_tip_sha256": ledger_summary["ledger_tip_sha256"],
        "ledger_row_count": len(ledger),
        "ledger_rows_sha256": canonical_sha256(ledger),
        "ledger": ledger,
        "terminal": ledger_summary["terminal"],
        "metrics": metrics,
    }
    return {**body, "score_receipt_sha256": canonical_sha256(body)}


def validate_score_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_score_receipt_sha256: str,
    prediction_prefix: Mapping[str, Any],
    expected_prediction_prefix_sha256: str,
    label_release_evidence: Mapping[str, Any],
    expected_label_release_ledger_sha256: str,
    market_stage: Mapping[str, Any],
    expected_market_stage_manifest_sha256: str,
    selected_candidate_id: str,
    selected_variant: str,
    cost_bps: int,
    stage: str,
    score_cutoff_session: str,
    terminal_convention: str,
) -> str:
    observed = _mapping(_snapshot(receipt, "score receipt"), "score receipt")
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "inputs",
        "configuration",
        "ledger_genesis_sha256",
        "ledger_tip_sha256",
        "ledger_row_count",
        "ledger_rows_sha256",
        "ledger",
        "terminal",
        "metrics",
        "score_receipt_sha256",
    }
    _keys(observed, expected_keys, "score receipt")
    observed_hash = _self_hash(
        observed,
        hash_field="score_receipt_sha256",
        expected_hash=expected_score_receipt_sha256,
        location="score receipt",
    )
    expected = build_score_receipt(
        prediction_prefix=prediction_prefix,
        expected_prediction_prefix_sha256=expected_prediction_prefix_sha256,
        label_release_evidence=label_release_evidence,
        expected_label_release_ledger_sha256=(
            expected_label_release_ledger_sha256
        ),
        market_stage=market_stage,
        expected_market_stage_manifest_sha256=(
            expected_market_stage_manifest_sha256
        ),
        selected_candidate_id=selected_candidate_id,
        selected_variant=selected_variant,
        cost_bps=cost_bps,
        stage=stage,
        score_cutoff_session=score_cutoff_session,
        terminal_convention=terminal_convention,
    )
    if observed != expected:
        raise SecFilingGemmaScoringError(
            "Score receipt differs from deterministic ledger replay"
        )
    return observed_hash


def _embedded_score(receipt: Any, expected_hash: str) -> dict[str, Any]:
    value = _mapping(_snapshot(receipt, "embedded score receipt"), "embedded score receipt")
    _keys(
        value,
        {
            "schema_version",
            "contract_sha256",
            "inputs",
            "configuration",
            "ledger_genesis_sha256",
            "ledger_tip_sha256",
            "ledger_row_count",
            "ledger_rows_sha256",
            "ledger",
            "terminal",
            "metrics",
            "score_receipt_sha256",
        },
        "embedded score receipt",
    )
    if value.get("schema_version") != SCORE_RECEIPT_SCHEMA_VERSION:
        raise SecFilingGemmaScoringError("Embedded score schema changed")
    _self_hash(
        value,
        hash_field="score_receipt_sha256",
        expected_hash=expected_hash,
        location="embedded score receipt",
    )
    return dict(value)


def _metric_hex(metrics: Mapping[str, Any], key: str) -> float:
    return decode_score_float_hex(metrics.get(key), key)


def _optional_metric_hex(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else decode_score_float_hex(value, key)


def _scenario_index(
    receipts: Sequence[Mapping[str, Any]],
    expected_hashes: Sequence[str],
    *,
    stage: str,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    if (
        isinstance(receipts, (str, bytes))
        or not isinstance(receipts, Sequence)
        or isinstance(expected_hashes, (str, bytes))
        or not isinstance(expected_hashes, Sequence)
        or len(receipts) != len(expected_hashes)
    ):
        raise SecFilingGemmaScoringError(
            "Score receipts and external hashes must be aligned sequences"
        )
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    common: tuple[Any, ...] | None = None
    for raw, expected_hash in zip(receipts, expected_hashes, strict=True):
        receipt = _embedded_score(raw, expected_hash)
        config = _mapping(receipt.get("configuration"), "score configuration")
        if config.get("stage") != stage:
            raise SecFilingGemmaScoringError("Gate input belongs to another stage")
        key = (
            config.get("selected_variant"),
            config.get("cost_bps"),
            config.get("terminal_convention"),
        )
        if (
            key[0] not in MODEL_VARIANTS
            or key[1] not in TRANSACTION_COST_BPS
            or key[2] not in TERMINAL_CONVENTIONS
            or key in result
        ):
            raise SecFilingGemmaScoringError(
                "Gate scenarios are unknown, duplicate, or noncanonical"
            )
        inputs = _mapping(receipt.get("inputs"), "score inputs")
        identity = (
            config.get("selected_candidate_id"),
            config.get("stage"),
            config.get("score_start_session"),
            config.get("score_cutoff_session"),
            inputs.get("prediction_prefix_sha256"),
            inputs.get("label_release_ledger_sha256"),
            inputs.get("market_stage_manifest_sha256"),
        )
        if common is None:
            common = identity
        elif identity != common:
            raise SecFilingGemmaScoringError(
                "Gate scenarios do not share one candidate, stage, cutoff, and inputs"
            )
        result[key] = receipt
    terminals = TERMINAL_CONVENTIONS if stage == "final" else ("adjusted_open",)
    expected_keys = {
        (variant, cost, terminal)
        for variant in MODEL_VARIANTS
        for cost in TRANSACTION_COST_BPS
        for terminal in terminals
    }
    if set(result) != expected_keys:
        raise SecFilingGemmaScoringError(
            "Gate inputs omit or add a required variant/cost/terminal scenario"
        )
    return result


def _development_checks(
    scenarios: Mapping[tuple[str, int, str], Mapping[str, Any]]
) -> tuple[dict[str, bool], dict[str, Any]]:
    contract = build_contract_manifest()["gates"]["development_both_5bps_and_10bps"]
    checks: dict[str, bool] = {}
    for cost in TRANSACTION_COST_BPS:
        metrics = _mapping(
            scenarios[("semantic", cost, "adjusted_open")]["metrics"],
            "development metrics",
        )
        suffix = f"_{cost}bps"
        rolling_252 = _optional_metric_hex(
            metrics, "rolling_252_session_month_end_win_rate_hex"
        )
        rolling_756 = _optional_metric_hex(
            metrics, "rolling_756_session_month_end_win_rate_hex"
        )
        largest = _optional_metric_hex(metrics, "largest_positive_year_share_hex")
        negative_rate = _optional_metric_hex(
            metrics, "negative_buy_hold_period_win_rate_hex"
        )
        brier = _mapping(metrics.get("brier"), "development brier")
        relative_climatology = _optional_metric_hex(
            brier, "relative_improvement_vs_climatology_hex"
        )
        checks.update(
            {
                f"total_active_log_edge{suffix}": _metric_hex(
                    metrics, "total_active_log_edge_hex"
                )
                >= contract["minimum_total_active_log_edge"],
                f"median_annual_active_log_edge{suffix}": _metric_hex(
                    metrics, "median_annual_active_log_edge_hex"
                )
                >= contract["minimum_median_annual_active_log_edge"],
                f"edge_without_best_year{suffix}": _metric_hex(
                    metrics, "active_log_edge_without_best_year_hex"
                )
                >= contract["minimum_edge_without_best_year"],
                f"rolling_252_win_rate{suffix}": rolling_252 is not None
                and rolling_252 >= contract["minimum_252_session_month_end_win_rate"],
                f"rolling_756_win_rate{suffix}": rolling_756 is not None
                and rolling_756 >= contract["minimum_756_session_month_end_win_rate"],
                f"annual_win_rate{suffix}": _metric_hex(
                    metrics, "annual_win_rate_hex"
                )
                >= contract["minimum_annual_win_rate"],
                f"positive_folds{suffix}": metrics.get("positive_fold_count")
                >= contract["minimum_positive_folds"],
                f"cash_days{suffix}": metrics.get("cash_days")
                >= contract["minimum_cash_days"],
                f"cash_episodes{suffix}": metrics.get("cash_episodes")
                >= contract["minimum_cash_episodes"],
                f"cash_rate{suffix}": _metric_hex(metrics, "cash_day_rate_hex")
                <= contract["maximum_cash_rate"],
                f"largest_positive_year_share{suffix}": largest is not None
                and largest <= contract["maximum_largest_positive_year_share"],
                f"negative_buy_hold_edge{suffix}": _metric_hex(
                    metrics,
                    "aggregate_active_log_edge_in_negative_buy_hold_periods_hex",
                )
                >= contract["minimum_negative_buy_hold_year_edge"],
                f"negative_buy_hold_win_rate{suffix}": negative_rate is not None
                and negative_rate
                >= contract["minimum_negative_buy_hold_year_win_rate"],
                f"brier_vs_causal_climatology{suffix}": (
                    relative_climatology is not None
                    and relative_climatology
                    >= contract[
                        "minimum_relative_brier_improvement_vs_causal_climatology"
                    ]
                ),
            }
        )
    semantic_10 = _mapping(
        scenarios[("semantic", 10, "adjusted_open")]["metrics"],
        "semantic 10-bps metrics",
    )
    ablation_10 = _mapping(
        scenarios[("ablation", 10, "adjusted_open")]["metrics"],
        "ablation 10-bps metrics",
    )
    semantic_brier = _mapping(semantic_10["brier"], "semantic brier")
    relative_ablation = _optional_metric_hex(
        semantic_brier, "relative_improvement_vs_ablation_hex"
    )
    fold_differences_by_cost: dict[int, dict[str, float]] = {}
    for cost in TRANSACTION_COST_BPS:
        semantic_folds = _mapping(
            scenarios[("semantic", cost, "adjusted_open")]["metrics"][
                "fold_active_log_edges_hex"
            ],
            f"semantic {cost}-bps fold edges",
        )
        ablation_folds = _mapping(
            scenarios[("ablation", cost, "adjusted_open")]["metrics"][
                "fold_active_log_edges_hex"
            ],
            f"ablation {cost}-bps fold edges",
        )
        if set(semantic_folds) != set(ablation_folds):
            raise SecFilingGemmaScoringError("Semantic/ablation folds differ")
        fold_differences_by_cost[cost] = {
            fold: decode_score_float_hex(semantic_folds[fold], fold)
            - decode_score_float_hex(ablation_folds[fold], fold)
            for fold in semantic_folds
        }
    active_advantage = _metric_hex(semantic_10, "total_active_log_edge_hex") - _metric_hex(
        ablation_10, "total_active_log_edge_hex"
    )
    checks.update(
        {
            "10bps_edge_advantage_vs_ablation": active_advantage
            >= contract["minimum_10bps_edge_advantage_vs_ablation"],
            "relative_brier_improvement_vs_ablation": relative_ablation is not None
            and relative_ablation
            >= contract["minimum_relative_brier_improvement_vs_ablation"],
            **{
                f"each_fold_edge_vs_ablation_{cost}bps": bool(
                    fold_differences_by_cost[cost]
                )
                and min(fold_differences_by_cost[cost].values())
                >= contract["minimum_fold_edge_difference_vs_ablation"]
                for cost in TRANSACTION_COST_BPS
            },
        }
    )
    return checks, {
        "10bps_active_edge_advantage_vs_ablation_hex": _float_hex(active_advantage),
        "fold_edge_differences_vs_ablation_hex_by_cost": {
            str(cost): {
                key: _float_hex(value)
                for key, value in fold_differences_by_cost[cost].items()
            }
            for cost in TRANSACTION_COST_BPS
        },
        "selection_brier_score_hex": semantic_brier["brier_score_hex"],
        "selection_10bps_active_log_edge_hex": semantic_10[
            "total_active_log_edge_hex"
        ],
    }


def _intermediate_checks(
    scenarios: Mapping[tuple[str, int, str], Mapping[str, Any]]
) -> tuple[dict[str, bool], dict[str, Any]]:
    gates = build_contract_manifest()["gates"]
    five = gates["intermediate_5bps"]
    ten = gates["intermediate_10bps"]
    semantic_5 = _mapping(
        scenarios[("semantic", 5, "adjusted_open")]["metrics"], "semantic 5"
    )
    semantic_10 = _mapping(
        scenarios[("semantic", 10, "adjusted_open")]["metrics"], "semantic 10"
    )
    ablation_5 = _mapping(
        scenarios[("ablation", 5, "adjusted_open")]["metrics"], "ablation 5"
    )
    ablation_10 = _mapping(
        scenarios[("ablation", 10, "adjusted_open")]["metrics"], "ablation 10"
    )
    largest = _optional_metric_hex(semantic_5, "largest_positive_year_share_hex")
    negative_rate = _optional_metric_hex(
        semantic_5, "negative_buy_hold_period_win_rate_hex"
    )
    relative_climatology = _optional_metric_hex(
        _mapping(semantic_5["brier"], "intermediate brier"),
        "relative_improvement_vs_climatology_hex",
    )
    advantage_5 = _metric_hex(semantic_5, "total_active_log_edge_hex") - _metric_hex(
        ablation_5, "total_active_log_edge_hex"
    )
    advantage_10 = _metric_hex(
        semantic_10, "total_active_log_edge_hex"
    ) - _metric_hex(ablation_10, "total_active_log_edge_hex")
    mean_episode_10 = _optional_metric_hex(
        semantic_10, "mean_episode_active_log_edge_hex"
    )
    checks = {
        "5bps_total_active_log_edge": _metric_hex(
            semantic_5, "total_active_log_edge_hex"
        )
        >= five["minimum_total_active_log_edge"],
        "5bps_winning_years": semantic_5["winning_year_count"]
        >= five["minimum_winning_years"],
        "5bps_exact_year_count": len(semantic_5["period_active_log_edges_hex"])
        == five["year_count"],
        "5bps_median_annual_edge": _metric_hex(
            semantic_5, "median_annual_active_log_edge_hex"
        )
        >= five["minimum_median_annual_edge"],
        "5bps_edge_without_best_year_exclusive": _metric_hex(
            semantic_5, "active_log_edge_without_best_year_hex"
        )
        > five["minimum_edge_without_best_year_exclusive"],
        "5bps_rolling_252": (
            _optional_metric_hex(
                semantic_5, "rolling_252_session_month_end_win_rate_hex"
            )
            is not None
            and _optional_metric_hex(
                semantic_5, "rolling_252_session_month_end_win_rate_hex"
            )
            >= five["minimum_252_session_month_end_win_rate"]
        ),
        "5bps_cash_days": semantic_5["cash_days"] >= five["minimum_cash_days"],
        "5bps_cash_episodes": semantic_5["cash_episodes"]
        >= five["minimum_cash_episodes"],
        "5bps_cash_rate": _metric_hex(semantic_5, "cash_day_rate_hex")
        <= five["maximum_cash_rate"],
        "5bps_largest_positive_year_share": largest is not None
        and largest <= five["maximum_largest_positive_year_share"],
        "5bps_negative_buy_hold_edge_exclusive": _metric_hex(
            semantic_5,
            "aggregate_active_log_edge_in_negative_buy_hold_periods_hex",
        )
        > five["minimum_negative_buy_hold_year_edge_exclusive"],
        "5bps_negative_buy_hold_year_exists": negative_rate is not None,
        "5bps_brier_vs_climatology": relative_climatology is not None
        and relative_climatology
        >= five["minimum_relative_brier_improvement_vs_climatology"],
        "5bps_edge_advantage_vs_ablation": advantage_5
        >= five["minimum_edge_advantage_vs_ablation"],
        "10bps_total_active_log_edge_exclusive": _metric_hex(
            semantic_10, "total_active_log_edge_hex"
        )
        > ten["minimum_total_active_log_edge_exclusive"],
        "10bps_winning_years": semantic_10["winning_year_count"]
        >= ten["minimum_winning_years"],
        "10bps_edge_without_best_year": _metric_hex(
            semantic_10, "active_log_edge_without_best_year_hex"
        )
        >= ten["minimum_edge_without_best_year"],
        "10bps_mean_episode_edge_exclusive": mean_episode_10 is not None
        and mean_episode_10 > ten["minimum_mean_episode_edge_exclusive"],
        "10bps_must_beat_ablation": advantage_10 > ACTIVE_EDGE_TOLERANCE,
    }
    return checks, {
        "5bps_active_edge_advantage_vs_ablation_hex": _float_hex(advantage_5),
        "10bps_active_edge_advantage_vs_ablation_hex": _float_hex(advantage_10),
    }


def _final_checks(
    scenarios: Mapping[tuple[str, int, str], Mapping[str, Any]]
) -> tuple[dict[str, bool], dict[str, Any]]:
    contract = build_contract_manifest()["gates"]["final"]
    checks: dict[str, bool] = {}
    comparison: dict[str, Any] = {}
    for terminal in TERMINAL_CONVENTIONS:
        semantic_5 = _mapping(scenarios[("semantic", 5, terminal)]["metrics"], "final 5")
        semantic_10 = _mapping(
            scenarios[("semantic", 10, terminal)]["metrics"], "final 10"
        )
        ablation_10 = _mapping(
            scenarios[("ablation", 10, terminal)]["metrics"], "final ablation"
        )
        periods_5 = _mapping(semantic_5["period_active_log_edges_hex"], "final 5 periods")
        periods_10 = _mapping(
            semantic_10["period_active_log_edges_hex"], "final 10 periods"
        )
        ablation_periods = _mapping(
            ablation_10["period_active_log_edges_hex"], "final ablation periods"
        )
        if set(periods_5) != set(contract["periods"]) or set(periods_10) != set(
            contract["periods"]
        ) or set(ablation_periods) != set(contract["periods"]):
            raise SecFilingGemmaScoringError("Final periods are incomplete or changed")
        suffix = f"_{terminal}"
        episode_rate = _optional_metric_hex(semantic_10, "episode_win_rate_hex")
        episode_mean = _optional_metric_hex(
            semantic_10, "mean_episode_active_log_edge_hex"
        )
        concentration = _optional_metric_hex(
            semantic_10, "maximum_single_episode_positive_edge_share_hex"
        )
        differences = {
            period: decode_score_float_hex(periods_10[period], period)
            - decode_score_float_hex(ablation_periods[period], period)
            for period in contract["periods"]
        }
        continuous_advantage = _metric_hex(
            semantic_10, "total_active_log_edge_hex"
        ) - _metric_hex(ablation_10, "total_active_log_edge_hex")
        checks.update(
            {
                f"each_period_5bps{suffix}": all(
                    decode_score_float_hex(periods_5[period], period)
                    >= contract["minimum_each_period_5bps_active_log_edge"]
                    for period in contract["periods"]
                ),
                f"continuous_5bps{suffix}": _metric_hex(
                    semantic_5, "total_active_log_edge_hex"
                )
                >= contract["minimum_continuous_5bps_active_log_edge"],
                f"each_period_10bps_exclusive{suffix}": all(
                    decode_score_float_hex(periods_10[period], period)
                    > contract["minimum_each_period_10bps_active_log_edge_exclusive"]
                    for period in contract["periods"]
                ),
                f"continuous_10bps_exclusive{suffix}": _metric_hex(
                    semantic_10, "total_active_log_edge_hex"
                )
                > contract["minimum_continuous_10bps_active_log_edge_exclusive"],
                f"episode_count_each_period{suffix}": all(
                    semantic_10["period_cash_episode_counts"][period]
                    >= contract["minimum_episode_count_each_period"]
                    for period in contract["periods"]
                ),
                f"total_episode_count{suffix}": semantic_10["cash_episodes"]
                >= contract["minimum_total_episode_count"],
                f"episode_win_rate{suffix}": episode_rate is not None
                and episode_rate >= contract["minimum_episode_win_rate"],
                f"mean_episode_edge_10bps_exclusive{suffix}": episode_mean is not None
                and episode_mean
                > contract["minimum_mean_episode_edge_10bps_exclusive"],
                f"episode_positive_edge_concentration{suffix}": concentration is not None
                and concentration
                <= contract["maximum_single_episode_positive_edge_share"],
                f"drawdown_disadvantage{suffix}": _metric_hex(
                    semantic_10, "drawdown_disadvantage_percentage_points_hex"
                )
                <= contract["maximum_drawdown_disadvantage_percentage_points"],
                f"continuous_edge_vs_ablation_exclusive{suffix}": continuous_advantage
                > contract[
                    "minimum_continuous_edge_advantage_vs_ablation_exclusive"
                ],
                f"periods_beating_ablation{suffix}": sum(
                    value > ACTIVE_EDGE_TOLERANCE for value in differences.values()
                )
                >= contract["minimum_periods_beating_ablation"],
            }
        )
        comparison[terminal] = {
            "continuous_10bps_edge_advantage_vs_ablation_hex": _float_hex(
                continuous_advantage
            ),
            "period_10bps_edge_differences_vs_ablation_hex": {
                key: _float_hex(value) for key, value in differences.items()
            },
        }
    return checks, comparison


def build_stage_gate_receipt(
    score_receipts: Sequence[Mapping[str, Any]],
    *,
    expected_score_receipt_sha256s: Sequence[str],
    stage: str,
) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise SecFilingGemmaScoringError("Unknown gate stage")
    scenarios = _scenario_index(
        score_receipts, expected_score_receipt_sha256s, stage=stage
    )
    if stage == "development":
        checks, comparison = _development_checks(scenarios)
    elif stage == "intermediate":
        checks, comparison = _intermediate_checks(scenarios)
    else:
        checks, comparison = _final_checks(scenarios)
    ordered_keys = sorted(scenarios, key=lambda item: (item[0], item[1], item[2]))
    first = scenarios[ordered_keys[0]]
    body = {
        "schema_version": GATE_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "stage": stage,
        "selected_candidate_id": first["configuration"]["selected_candidate_id"],
        "score_cutoff_session": first["configuration"]["score_cutoff_session"],
        "score_receipt_sha256s": [
            scenarios[key]["score_receipt_sha256"] for key in ordered_keys
        ],
        "scenario_keys": [
            {"variant": key[0], "cost_bps": key[1], "terminal": key[2]}
            for key in ordered_keys
        ],
        "checks": checks,
        "comparison": comparison,
        "passed": all(checks.values()),
    }
    return {**body, "gate_receipt_sha256": canonical_sha256(body)}


def validate_stage_gate_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_gate_receipt_sha256: str,
    score_receipts: Sequence[Mapping[str, Any]],
    expected_score_receipt_sha256s: Sequence[str],
    stage: str,
) -> str:
    observed = _mapping(_snapshot(receipt, "gate receipt"), "gate receipt")
    observed_hash = _self_hash(
        observed,
        hash_field="gate_receipt_sha256",
        expected_hash=expected_gate_receipt_sha256,
        location="gate receipt",
    )
    expected = build_stage_gate_receipt(
        score_receipts,
        expected_score_receipt_sha256s=expected_score_receipt_sha256s,
        stage=stage,
    )
    if observed != expected:
        raise SecFilingGemmaScoringError("Gate receipt differs from exact replay")
    return observed_hash


def build_development_ranking_receipt(
    gate_receipts: Sequence[Mapping[str, Any]],
    *,
    expected_gate_receipt_sha256s: Sequence[str],
) -> dict[str, Any]:
    if (
        len(gate_receipts) != len(CANDIDATE_IDS)
        or len(expected_gate_receipt_sha256s) != len(CANDIDATE_IDS)
    ):
        raise SecFilingGemmaScoringError(
            "Development ranking requires exactly all four candidates"
        )
    by_candidate: dict[str, dict[str, Any]] = {}
    for raw, expected_hash in zip(
        gate_receipts, expected_gate_receipt_sha256s, strict=True
    ):
        gate = _mapping(_snapshot(raw, "development gate"), "development gate")
        _self_hash(
            gate,
            hash_field="gate_receipt_sha256",
            expected_hash=expected_hash,
            location="development gate",
        )
        candidate = gate.get("selected_candidate_id")
        if gate.get("stage") != "development" or candidate not in CANDIDATE_IDS:
            raise SecFilingGemmaScoringError("Ranking input is not a development gate")
        if candidate in by_candidate:
            raise SecFilingGemmaScoringError("Development candidate is duplicated")
        by_candidate[candidate] = dict(gate)
    if set(by_candidate) != set(CANDIDATE_IDS):
        raise SecFilingGemmaScoringError("Development candidate grid is incomplete")
    passing = [candidate for candidate in CANDIDATE_IDS if by_candidate[candidate]["passed"]]

    def rank_key(candidate: str) -> tuple[float, float, int]:
        comparison = _mapping(by_candidate[candidate]["comparison"], "ranking comparison")
        return (
            decode_score_float_hex(comparison["selection_brier_score_hex"], "rank brier"),
            -decode_score_float_hex(
                comparison["selection_10bps_active_log_edge_hex"], "rank edge"
            ),
            CANDIDATE_IDS.index(candidate),
        )

    ordered_passing = sorted(passing, key=rank_key)
    selected = ordered_passing[0] if ordered_passing else None
    body = {
        "schema_version": RANKING_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "candidate_order": list(CANDIDATE_IDS),
        "gate_receipt_sha256s_by_candidate": {
            candidate: by_candidate[candidate]["gate_receipt_sha256"]
            for candidate in CANDIDATE_IDS
        },
        "passing_candidates_in_rank_order": ordered_passing,
        "selected_candidate_id": selected,
        "ranking_policy": (
            "pass_all_then_lowest_10bps_brier_then_highest_10bps_active_edge_"
            "then_frozen_candidate_order"
        ),
    }
    return {**body, "ranking_receipt_sha256": canonical_sha256(body)}


def validate_development_ranking_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_ranking_receipt_sha256: str,
    gate_receipts: Sequence[Mapping[str, Any]],
    expected_gate_receipt_sha256s: Sequence[str],
) -> str:
    observed = _mapping(_snapshot(receipt, "ranking receipt"), "ranking receipt")
    observed_hash = _self_hash(
        observed,
        hash_field="ranking_receipt_sha256",
        expected_hash=expected_ranking_receipt_sha256,
        location="ranking receipt",
    )
    expected = build_development_ranking_receipt(
        gate_receipts,
        expected_gate_receipt_sha256s=expected_gate_receipt_sha256s,
    )
    if observed != expected:
        raise SecFilingGemmaScoringError("Ranking receipt differs from exact replay")
    return observed_hash


__all__ = [
    "GATE_RECEIPT_SCHEMA_VERSION",
    "INITIAL_CAPITAL",
    "RANKING_RECEIPT_SCHEMA_VERSION",
    "SCORE_RECEIPT_SCHEMA_VERSION",
    "TERMINAL_CONVENTIONS",
    "TRANSACTION_COST_BPS",
    "SecFilingGemmaScoringError",
    "build_development_ranking_receipt",
    "build_score_receipt",
    "build_stage_gate_receipt",
    "decode_score_float_hex",
    "validate_development_ranking_receipt",
    "validate_score_receipt",
    "validate_stage_gate_receipt",
]
