"""Pure chronological orchestration for the SEC/Gemma online risk overlay.

The replay owns no data acquisition, filesystem, clock, or model-runtime
effects.  It accepts externally pinned decision-time feature rows, exact
market rows, and exact baseline signals.  Outcome lessons can enter only by
replaying the fixed t+21 counterfactual ledger from those inputs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hmac
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_learner import (
    FEATURE_ARMS,
    build_online_overlay_fit_audit,
    build_online_overlay_lesson,
    build_online_overlay_prediction_from_fit,
    validate_online_overlay_feature_row,
    validate_online_overlay_fit_audit,
    validate_online_overlay_lesson,
    validate_online_overlay_prediction_from_fit,
)
from agent_benchmark.sec_gemma_online_risk_overlay_ledger import (
    build_combined_target_rows,
    build_mature_counterfactual_lesson,
    compare_ledgers,
    run_binary_ledger,
    validate_mature_counterfactual_lesson,
)
from agent_benchmark.sec_gemma_online_risk_overlay_policy import (
    replay_nonoverlapping_overlay_policy,
)


CHRONOLOGICAL_REPLAY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-chronological-replay-v1"
)
LABEL_INTENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-label-intent-v1"
)
FIT_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-fit-checkpoint-v1"
)
ALLOWED_REPLAY_COST_BPS: Final[tuple[int, ...]] = (5, 10)
PORTFOLIO_GENESIS_SESSION: Final[str] = "2000-01-03"


class SecGemmaOnlineRiskOverlayReplayError(ValueError):
    """Raised when an exact causal replay cannot be constructed."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecGemmaOnlineRiskOverlayReplayError(
            f"{location} must be a mapping"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayReplayError(
            f"{location} must be a canonical ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayReplayError(
            f"{location} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayReplayError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecGemmaOnlineRiskOverlayReplayError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _acceptance(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 14
        or not value.isascii()
        or not value.isdigit()
    ):
        raise SecGemmaOnlineRiskOverlayReplayError(
            f"{location} is invalid"
        )
    return value


def _market_sessions(
    market_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    if isinstance(market_rows, (str, bytes)) or not isinstance(
        market_rows, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "market rows must be a sequence"
        )
    sessions: list[str] = []
    previous: str | None = None
    for ordinal, raw in enumerate(market_rows, start=1):
        row = _mapping(raw, f"market row {ordinal}")
        session = _iso_date(
            row.get("session"), f"market row {ordinal}.session"
        )
        if previous is not None and session <= previous:
            raise SecGemmaOnlineRiskOverlayReplayError(
                "market sessions must be strictly increasing"
            )
        sessions.append(session)
        previous = session
    if not sessions:
        raise SecGemmaOnlineRiskOverlayReplayError(
            "market rows must not be empty"
        )
    if sessions[0] != PORTFOLIO_GENESIS_SESSION:
        raise SecGemmaOnlineRiskOverlayReplayError(
            "portfolio market rows must begin exactly on 2000-01-03"
        )
    return sessions


def _require_external_batch_hash(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sha256: str,
    location: str,
) -> str:
    expected = _sha256(expected_sha256, f"expected {location} hash")
    observed = canonical_sha256([dict(_mapping(row, location)) for row in rows])
    if not hmac.compare_digest(observed, expected):
        raise SecGemmaOnlineRiskOverlayReplayError(
            f"{location} differ from their external immutable pin"
        )
    return observed


def _prevalidate_feature_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    expected_feature_row_sha256s: Sequence[str],
    permitted_sessions: set[str],
) -> list[dict[str, Any]]:
    """Fail same-session ambiguity before constructing any replay artifact."""

    if isinstance(feature_rows, (str, bytes)) or not isinstance(
        feature_rows, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "feature rows must be a sequence"
        )
    if isinstance(
        expected_feature_row_sha256s, (str, bytes)
    ) or not isinstance(expected_feature_row_sha256s, Sequence):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "expected feature hashes must be a sequence"
        )
    pins = [
        _sha256(value, f"expected feature hash {ordinal}")
        for ordinal, value in enumerate(
            expected_feature_row_sha256s, start=1
        )
    ]
    if len(pins) != len(set(pins)) or len(pins) != len(feature_rows):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "expected feature hashes must uniquely pin every feature row"
        )

    raw_rows: list[dict[str, Any]] = []
    observed_hashes: set[str] = set()
    accessions: set[str] = set()
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ordinal, raw in enumerate(feature_rows, start=1):
        row = copy.deepcopy(dict(_mapping(raw, f"feature row {ordinal}")))
        observed = _sha256(
            row.get("feature_row_sha256"),
            f"feature row {ordinal}.feature_row_sha256",
        )
        if observed in observed_hashes:
            raise SecGemmaOnlineRiskOverlayReplayError(
                "feature row hash is duplicated"
            )
        observed_hashes.add(observed)
        accession = row.get("accession_number")
        if (
            not isinstance(accession, str)
            or not accession
            or accession in accessions
        ):
            raise SecGemmaOnlineRiskOverlayReplayError(
                "feature accession is invalid or duplicated"
            )
        accessions.add(accession)
        decision = _iso_date(
            row.get("decision_session"),
            f"feature row {ordinal}.decision_session",
        )
        if decision not in permitted_sessions:
            raise SecGemmaOnlineRiskOverlayReplayError(
                "feature decision lies outside the market prefix"
            )
        _acceptance(
            row.get("acceptance_datetime"),
            f"feature row {ordinal}.acceptance_datetime",
        )
        raw_rows.append(row)
        by_session[decision].append(row)
    if observed_hashes != set(pins):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "expected feature hashes do not match supplied feature rows"
        )
    for same_session in by_session.values():
        if len(same_session) > 1 and any(
            row["acceptance_datetime"] is None for row in same_session
        ):
            raise SecGemmaOnlineRiskOverlayReplayError(
                "same-session filing order is ambiguous without exact "
                "acceptance timestamps"
            )

    validated = [
        validate_online_overlay_feature_row(
            row,
            expected_feature_row_sha256=row["feature_row_sha256"],
        )
        for row in raw_rows
    ]
    validated.sort(
        key=lambda row: (
            row["decision_session"],
            (
                ""
                if row["acceptance_datetime"] is None
                else row["acceptance_datetime"]
            ),
            row["accession_number"],
            row["feature_row_sha256"],
        )
    )
    return validated


def _build_label_intent(
    *,
    feature_row: Mapping[str, Any],
    decision_position: int,
    market_genesis_session: str,
) -> dict[str, Any]:
    body = {
        "schema_version": LABEL_INTENT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "accession_number": feature_row["accession_number"],
        "decision_session": feature_row["decision_session"],
        "acceptance_datetime": feature_row["acceptance_datetime"],
        "feature_row_sha256": feature_row["feature_row_sha256"],
        "feature_fit_eligible": feature_row["fit_eligible"],
        "market_genesis_session": market_genesis_session,
        "decision_position": decision_position,
        "entry_position": decision_position + 1,
        "maturity_position": (
            decision_position + LABEL_MATURITY_OFFSET
        ),
        "entry_session_offset": 1,
        "label_maturity_offset": LABEL_MATURITY_OFFSET,
        "horizon_sessions": HORIZON_SESSIONS,
    }
    return {**body, "label_intent_sha256": canonical_sha256(body)}


def _materialize_lesson(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    baseline_signals: Sequence[Mapping[str, Any]],
    feature: Mapping[str, Any],
    intent: Mapping[str, Any],
    maturity_session: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one outcome lesson from only the now-visible market prefix."""

    counterfactual = build_mature_counterfactual_lesson(
        market_rows=market_rows,
        baseline_signals=baseline_signals,
        decision_session=feature["decision_session"],
        accession_number=feature["accession_number"],
        feature_row_sha256=feature["feature_row_sha256"],
        feature_fit_eligible=bool(feature["fit_eligible"]),
        as_of_session=maturity_session,
    )
    validate_mature_counterfactual_lesson(
        counterfactual,
        expected_counterfactual_lesson_sha256=counterfactual[
            "counterfactual_lesson_sha256"
        ],
        market_rows=market_rows,
        baseline_signals=baseline_signals,
        decision_session=feature["decision_session"],
        accession_number=feature["accession_number"],
        feature_row_sha256=feature["feature_row_sha256"],
        feature_fit_eligible=bool(feature["fit_eligible"]),
        as_of_session=maturity_session,
    )
    if (
        counterfactual["maturity_session"] != maturity_session
        or int(intent["maturity_position"]) != len(market_rows) - 1
        or counterfactual["feature_row_sha256"]
        != feature["feature_row_sha256"]
        or counterfactual["feature_fit_eligible"]
        != feature["fit_eligible"]
        or counterfactual["train_eligible"] != feature["fit_eligible"]
        or counterfactual["audit_only"] == feature["fit_eligible"]
    ):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "counterfactual evidence differs from its immutable "
            "decision-time feature row or maturity prefix"
        )
    edge = float.fromhex(
        counterfactual["incremental_log_edge_10bps_hex"]
    )
    lesson = build_online_overlay_lesson(
        feature_row=feature,
        expected_feature_row_sha256=feature["feature_row_sha256"],
        maturity_session=maturity_session,
        active_log_edge_10bps=edge,
        label_evidence_sha256=counterfactual[
            "counterfactual_lesson_sha256"
        ],
    )
    validate_online_overlay_lesson(
        lesson,
        expected_lesson_row_sha256=lesson["lesson_row_sha256"],
    )
    if (
        lesson["trainable"] != feature["fit_eligible"]
        or lesson["binary_cash_win_target"]
        != counterfactual["binary_overlay_win"]
        or lesson["active_log_edge_10bps_hex"]
        != counterfactual["incremental_log_edge_10bps_hex"]
        or lesson["label_evidence_sha256"]
        != counterfactual["counterfactual_lesson_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "learner lesson differs from validated counterfactual evidence"
        )
    return counterfactual, lesson


def _fit_checkpoint(
    fit_audit: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": FIT_CHECKPOINT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "as_of_session": fit_audit["as_of_session"],
        "fit_audit_sha256": fit_audit["fit_audit_sha256"],
        "fit_audit": copy.deepcopy(dict(fit_audit)),
    }
    return {**body, "fit_checkpoint_sha256": canonical_sha256(body)}


def _build_chronological_learning(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    baseline_signals: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    label_intents: Sequence[Mapping[str, Any]],
    arm: str,
    frozen_before_boundary: str | None,
    sessions: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any] | None,
    list[dict[str, Any]] | None,
]:
    counterfactuals: list[dict[str, Any]] = []
    lessons: list[dict[str, Any]] = []
    online_predictions: list[dict[str, Any]] = []
    fit_checkpoints: list[dict[str, Any]] = []
    frozen_fit: dict[str, Any] | None = None
    frozen_predictions: list[dict[str, Any]] | None = (
        [] if frozen_before_boundary is not None else None
    )
    boundary: str | None = None
    preceding_boundary_session: str | None = None
    if frozen_before_boundary is not None:
        boundary = _iso_date(
            frozen_before_boundary, "frozen-before boundary"
        )
        if boundary not in sessions:
            raise SecGemmaOnlineRiskOverlayReplayError(
                "frozen-before boundary must be an exact market session"
            )
        boundary_position = sessions.index(boundary)
        if boundary_position == 0:
            raise SecGemmaOnlineRiskOverlayReplayError(
                "frozen-before boundary requires a preceding market session"
            )
        preceding_boundary_session = sessions[boundary_position - 1]

    features_by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    features_by_hash: dict[str, Mapping[str, Any]] = {}
    for feature in features:
        features_by_session[feature["decision_session"]].append(feature)
        features_by_hash[feature["feature_row_sha256"]] = feature
    intents_by_maturity: dict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for intent in label_intents:
        maturity_position = int(intent["maturity_position"])
        if maturity_position >= len(sessions):
            continue
        feature = features_by_hash[intent["feature_row_sha256"]]
        intents_by_maturity[sessions[maturity_position]].append(
            (feature, intent)
        )

    def build_fit(as_of_session: str) -> dict[str, Any]:
        fit = build_online_overlay_fit_audit(
            matured_lessons=lessons,
            expected_lesson_row_sha256s=[
                lesson["lesson_row_sha256"] for lesson in lessons
            ],
            arm=arm,
            as_of_session=as_of_session,
        )
        validate_online_overlay_fit_audit(
            fit,
            expected_fit_audit_sha256=fit["fit_audit_sha256"],
        )
        return fit

    for position, decision_session in enumerate(sessions):
        if boundary is not None and decision_session == boundary:
            assert preceding_boundary_session is not None
            frozen_fit = build_fit(preceding_boundary_session)

        market_prefix = market_rows[: position + 1]
        baseline_prefix = baseline_signals[: position + 1]
        for feature, intent in intents_by_maturity.get(
            decision_session, []
        ):
            counterfactual, lesson = _materialize_lesson(
                market_rows=market_prefix,
                baseline_signals=baseline_prefix,
                feature=feature,
                intent=intent,
                maturity_session=decision_session,
            )
            counterfactuals.append(counterfactual)
            lessons.append(lesson)

        decision_features = features_by_session.get(decision_session, [])
        if not decision_features:
            continue
        fit = build_fit(decision_session)
        fit_checkpoints.append(_fit_checkpoint(fit))
        same_session_hash: str | None = None
        for feature in decision_features:
            prediction = build_online_overlay_prediction_from_fit(
                feature_row=feature,
                expected_feature_row_sha256=feature[
                    "feature_row_sha256"
                ],
                fit_audit=fit,
                expected_fit_audit_sha256=fit["fit_audit_sha256"],
            )
            validate_online_overlay_prediction_from_fit(
                prediction,
                expected_prediction_row_sha256=prediction[
                    "prediction_row_sha256"
                ],
                feature_row=feature,
                expected_feature_row_sha256=feature[
                    "feature_row_sha256"
                ],
                fit_audit=fit,
                expected_fit_audit_sha256=fit["fit_audit_sha256"],
            )
            if same_session_hash is None:
                same_session_hash = prediction["fit_audit_sha256"]
            elif prediction["fit_audit_sha256"] != same_session_hash:
                raise SecGemmaOnlineRiskOverlayReplayError(
                    "same-session filings did not share one exact fit"
                )
            online_predictions.append(prediction)
            if frozen_predictions is not None:
                if boundary is not None and decision_session < boundary:
                    frozen_predictions.append(copy.deepcopy(prediction))
                else:
                    if frozen_fit is None:
                        raise SecGemmaOnlineRiskOverlayReplayError(
                            "frozen fit was not sealed before boundary use"
                        )
                    frozen_prediction = (
                        build_online_overlay_prediction_from_fit(
                            feature_row=feature,
                            expected_feature_row_sha256=feature[
                                "feature_row_sha256"
                            ],
                            fit_audit=frozen_fit,
                            expected_fit_audit_sha256=frozen_fit[
                                "fit_audit_sha256"
                            ],
                        )
                    )
                    validate_online_overlay_prediction_from_fit(
                        frozen_prediction,
                        expected_prediction_row_sha256=frozen_prediction[
                            "prediction_row_sha256"
                        ],
                        feature_row=feature,
                        expected_feature_row_sha256=feature[
                            "feature_row_sha256"
                        ],
                        fit_audit=frozen_fit,
                        expected_fit_audit_sha256=frozen_fit[
                            "fit_audit_sha256"
                        ],
                    )
                    frozen_predictions.append(frozen_prediction)

    terminal_fit = build_fit(sessions[-1])
    return (
        counterfactuals,
        lessons,
        online_predictions,
        fit_checkpoints,
        terminal_fit,
        frozen_fit,
        frozen_predictions,
    )


def _aapl_target_rows(
    sessions: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        body = {"session": session, "target_exposure": 1}
        rows.append({**body, "target_row_sha256": canonical_sha256(body)})
    return rows


def _run_common_ledgers(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    combined_target_rows: Sequence[Mapping[str, Any]],
    baseline_target_rows: Sequence[Mapping[str, Any]],
    aapl_target_rows: Sequence[Mapping[str, Any]],
    combined_policy_id: str,
    baseline_policy_id: str,
    aapl_policy_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cost_bps in ALLOWED_REPLAY_COST_BPS:
        combined = run_binary_ledger(
            market_rows=market_rows,
            target_rows=combined_target_rows,
            policy_id=combined_policy_id,
            cost_bps=cost_bps,
        )
        baseline = run_binary_ledger(
            market_rows=market_rows,
            target_rows=baseline_target_rows,
            policy_id=baseline_policy_id,
            cost_bps=cost_bps,
        )
        aapl = run_binary_ledger(
            market_rows=market_rows,
            target_rows=aapl_target_rows,
            policy_id=aapl_policy_id,
            cost_bps=cost_bps,
        )
        result[f"cost_{cost_bps}bps"] = {
            "combined": combined,
            "baseline": baseline,
            "aapl_buy_and_hold": aapl,
            "combined_vs_baseline": compare_ledgers(
                combined, baseline
            ),
            "combined_vs_aapl": compare_ledgers(combined, aapl),
            "baseline_vs_aapl": compare_ledgers(baseline, aapl),
        }
    return result


def replay_sec_gemma_online_risk_overlay_chronology(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    expected_market_rows_sha256: str,
    baseline_signals: Sequence[Mapping[str, Any]],
    expected_baseline_signals_sha256: str,
    feature_rows: Sequence[Mapping[str, Any]],
    expected_feature_row_sha256s: Sequence[str],
    arm: str,
    replay_id: str,
    frozen_before_boundary: str | None = None,
) -> dict[str, Any]:
    """Replay one arm with continuous learning and an optional frozen control."""

    if arm not in FEATURE_ARMS:
        raise SecGemmaOnlineRiskOverlayReplayError(
            "unknown SEC/Gemma feature arm"
        )
    if not isinstance(replay_id, str) or not replay_id:
        raise SecGemmaOnlineRiskOverlayReplayError("replay_id is invalid")
    sessions = _market_sessions(market_rows)
    features = _prevalidate_feature_rows(
        feature_rows,
        expected_feature_row_sha256s=expected_feature_row_sha256s,
        permitted_sessions=set(sessions),
    )
    market_hash = _require_external_batch_hash(
        market_rows,
        expected_sha256=expected_market_rows_sha256,
        location="market rows",
    )
    baseline_hash = _require_external_batch_hash(
        baseline_signals,
        expected_sha256=expected_baseline_signals_sha256,
        location="baseline signals",
    )

    baseline_target_stream = build_combined_target_rows(
        market_rows=market_rows,
        baseline_signals=baseline_signals,
        scheduled_overlays=[],
    )
    positions = {
        session: position for position, session in enumerate(sessions)
    }
    label_intents = [
        _build_label_intent(
            feature_row=feature,
            decision_position=positions[feature["decision_session"]],
            market_genesis_session=sessions[0],
        )
        for feature in features
    ]
    (
        counterfactuals,
        lessons,
        predictions,
        fit_checkpoints,
        terminal_fit,
        frozen_fit,
        frozen_predictions,
    ) = _build_chronological_learning(
        market_rows=market_rows,
        baseline_signals=baseline_signals,
        features=features,
        label_intents=label_intents,
        arm=arm,
        frozen_before_boundary=frozen_before_boundary,
        sessions=sessions,
    )
    matured_intent_hashes = {
        lesson["feature_row_sha256"] for lesson in lessons
    }
    pending_intents = [
        copy.deepcopy(intent)
        for intent in label_intents
        if intent["feature_row_sha256"] not in matured_intent_hashes
    ]
    baseline_policy_id = f"{replay_id}:baseline"
    aapl_policy_id = f"{replay_id}:aapl-buy-and-hold"
    primary_policy_id = f"{replay_id}:{arm}"
    primary_policy = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=predictions,
        policy_id=primary_policy_id,
    )
    primary_targets = build_combined_target_rows(
        market_rows=market_rows,
        baseline_signals=baseline_signals,
        scheduled_overlays=primary_policy["scheduled_overlays"],
    )
    aapl_targets = _aapl_target_rows(sessions)
    primary_ledgers = _run_common_ledgers(
        market_rows=market_rows,
        combined_target_rows=primary_targets["target_rows"],
        baseline_target_rows=baseline_target_stream["target_rows"],
        aapl_target_rows=aapl_targets,
        combined_policy_id=primary_policy_id,
        baseline_policy_id=baseline_policy_id,
        aapl_policy_id=aapl_policy_id,
    )
    primary = {
        "learning_mode": "continuous_online",
        "predictions": predictions,
        "predictions_sha256": canonical_sha256(predictions),
        "fit_checkpoints": fit_checkpoints,
        "fit_checkpoints_sha256": canonical_sha256(fit_checkpoints),
        "terminal_fit_audit": terminal_fit,
        "terminal_fit_audit_sha256": terminal_fit["fit_audit_sha256"],
        "policy_replay": primary_policy,
        "policy_replay_sha256": primary_policy["policy_replay_sha256"],
        "target_stream": primary_targets,
        "target_stream_sha256": primary_targets["target_stream_sha256"],
        "ledgers": primary_ledgers,
        "ledgers_sha256": canonical_sha256(primary_ledgers),
    }

    frozen: dict[str, Any] | None = None
    if frozen_predictions is not None and frozen_fit is not None:
        assert frozen_before_boundary is not None
        frozen_policy_id = primary_policy_id
        frozen_policy = replay_nonoverlapping_overlay_policy(
            market_sessions=sessions,
            prediction_rows=frozen_predictions,
            policy_id=frozen_policy_id,
        )
        frozen_targets = build_combined_target_rows(
            market_rows=market_rows,
            baseline_signals=baseline_signals,
            scheduled_overlays=frozen_policy["scheduled_overlays"],
        )
        frozen_ledgers = _run_common_ledgers(
            market_rows=market_rows,
            combined_target_rows=frozen_targets["target_rows"],
            baseline_target_rows=baseline_target_stream["target_rows"],
            aapl_target_rows=aapl_targets,
            combined_policy_id=frozen_policy_id,
            baseline_policy_id=baseline_policy_id,
            aapl_policy_id=aapl_policy_id,
        )
        frozen = {
            "learning_mode": "frozen_before_boundary",
            "frozen_before_boundary": frozen_before_boundary,
            "frozen_fit_audit": frozen_fit,
            "frozen_fit_audit_sha256": frozen_fit["fit_audit_sha256"],
            "predictions": frozen_predictions,
            "predictions_sha256": canonical_sha256(
                frozen_predictions
            ),
            "policy_replay": frozen_policy,
            "policy_replay_sha256": frozen_policy[
                "policy_replay_sha256"
            ],
            "target_stream": frozen_targets,
            "target_stream_sha256": frozen_targets[
                "target_stream_sha256"
            ],
            "ledgers": frozen_ledgers,
            "ledgers_sha256": canonical_sha256(frozen_ledgers),
        }

    body = {
        "schema_version": CHRONOLOGICAL_REPLAY_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "replay_id": replay_id,
        "arm": arm,
        "market_first_session": sessions[0],
        "market_cutoff_session": sessions[-1],
        "market_row_count": len(sessions),
        "market_rows_sha256": market_hash,
        "baseline_signals_sha256": baseline_hash,
        "ordered_feature_row_sha256s": [
            feature["feature_row_sha256"] for feature in features
        ],
        "ordered_feature_rows_sha256": canonical_sha256(
            [feature["feature_row_sha256"] for feature in features]
        ),
        "label_intents": label_intents,
        "label_intents_sha256": canonical_sha256(label_intents),
        "counterfactual_lessons": counterfactuals,
        "counterfactual_lessons_sha256": canonical_sha256(
            counterfactuals
        ),
        "learner_lessons": lessons,
        "learner_lessons_sha256": canonical_sha256(lessons),
        "pending_label_intents": pending_intents,
        "pending_label_intents_sha256": canonical_sha256(
            pending_intents
        ),
        "baseline_target_stream": baseline_target_stream,
        "baseline_target_stream_sha256": baseline_target_stream[
            "target_stream_sha256"
        ],
        "aapl_target_rows": aapl_targets,
        "aapl_target_rows_sha256": canonical_sha256(aapl_targets),
        "primary": primary,
        "frozen_control": frozen,
    }
    return {**body, "chronological_replay_sha256": canonical_sha256(body)}


def validate_sec_gemma_online_risk_overlay_chronology(
    replay: Mapping[str, Any],
    *,
    expected_chronological_replay_sha256: str,
    market_rows: Sequence[Mapping[str, Any]],
    expected_market_rows_sha256: str,
    baseline_signals: Sequence[Mapping[str, Any]],
    expected_baseline_signals_sha256: str,
    feature_rows: Sequence[Mapping[str, Any]],
    expected_feature_row_sha256s: Sequence[str],
    arm: str,
    replay_id: str,
    frozen_before_boundary: str | None = None,
) -> str:
    """Rebuild one complete chronology and require byte-level identity."""

    if not isinstance(replay, Mapping):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "chronological replay must be a mapping"
        )
    rebuilt = replay_sec_gemma_online_risk_overlay_chronology(
        market_rows=market_rows,
        expected_market_rows_sha256=expected_market_rows_sha256,
        baseline_signals=baseline_signals,
        expected_baseline_signals_sha256=(
            expected_baseline_signals_sha256
        ),
        feature_rows=feature_rows,
        expected_feature_row_sha256s=expected_feature_row_sha256s,
        arm=arm,
        replay_id=replay_id,
        frozen_before_boundary=frozen_before_boundary,
    )
    if dict(replay) != rebuilt:
        raise SecGemmaOnlineRiskOverlayReplayError(
            "chronological replay differs from deterministic reconstruction"
        )
    observed = _sha256(
        replay.get("chronological_replay_sha256"),
        "chronological replay hash",
    )
    expected = _sha256(
        expected_chronological_replay_sha256,
        "expected chronological replay hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecGemmaOnlineRiskOverlayReplayError(
            "chronological replay is not externally pinned"
        )
    return observed


__all__ = [
    "ALLOWED_REPLAY_COST_BPS",
    "CHRONOLOGICAL_REPLAY_SCHEMA_VERSION",
    "FIT_CHECKPOINT_SCHEMA_VERSION",
    "LABEL_INTENT_SCHEMA_VERSION",
    "PORTFOLIO_GENESIS_SESSION",
    "SecGemmaOnlineRiskOverlayReplayError",
    "replay_sec_gemma_online_risk_overlay_chronology",
    "validate_sec_gemma_online_risk_overlay_chronology",
]
