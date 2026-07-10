"""Causal CFTC Commitments of Traders sentiment policy.

This module is intentionally small and dependency free.  It consumes normalized
weekly observations prepared elsewhere; it does not download, revise, or infer
CFTC data.  A decision uses the newest *complete* NASDAQ/SP500/VIX release known
on the decision date and standardizes it against releases that became available
strictly earlier.  Consequently, neither the current observation nor a later
release can leak into its own reference distribution.

The policy family is frozen to four variants: a 26- or 52-release lookback crossed
with an absolute z threshold of 0.75 or 1.25.  Two of these three risk-off signals
are required to move from the long baseline to cash:

* unusually low NASDAQ non-commercial net share;
* unusually low NASDAQ-minus-SP500 net-share divergence; and
* unusually high VIX non-commercial net share.

Outputs are binary target exposures (1.0 long or 0.0 cash).  Shorting, leverage,
partial sizing, fitting, and outcome-based updates do not exist in this module.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping


Market = Literal["NASDAQ", "SP500", "VIX"]
Stance = Literal["LONG", "CASH"]
DateLike = str | date | datetime

REQUIRED_MARKETS: tuple[Market, ...] = ("NASDAQ", "SP500", "VIX")
POLICY_METHOD = "cftc_legacy_causal_2_of_3_v1"
MAX_RELEASE_AGE_DAYS = 14


def _as_date(value: DateLike, *, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty")
        try:
            if len(text) == 10:
                return date.fromisoformat(text)
            # Weekly COT inputs are date-granular.  Accept a valid timestamp
            # for convenience but deliberately discard its time component.
            normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
            return datetime.fromisoformat(normalized).date()
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 date") from exc
    raise TypeError(f"{field_name} must be a string, date, or datetime")


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


@dataclass(frozen=True)
class COTWeeklyRecord:
    """One normalized market observation and the date it became usable.

    ``net_share`` is expected to be calculated upstream from a single CFTC
    vintage, for example ``(noncommercial_long - noncommercial_short) /
    open_interest``.  ``availability_date`` must represent the conservative
    point-in-time date chosen by the data contract, not merely the report date.
    """

    market: Market
    report_date: DateLike
    availability_date: DateLike
    net_share: float

    def __post_init__(self) -> None:
        market = str(self.market).strip().upper()
        if market not in REQUIRED_MARKETS:
            raise ValueError(f"market must be one of {list(REQUIRED_MARKETS)}")
        report_date = _as_date(self.report_date, field_name="report_date")
        availability_date = _as_date(self.availability_date, field_name="availability_date")
        if availability_date < report_date:
            raise ValueError("availability_date cannot be before report_date")
        net_share = _finite_float(self.net_share, field_name="net_share")

        object.__setattr__(self, "market", market)
        object.__setattr__(self, "report_date", report_date)
        object.__setattr__(self, "availability_date", availability_date)
        object.__setattr__(self, "net_share", net_share)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "COTWeeklyRecord":
        return cls(
            market=payload["market"],
            report_date=payload["report_date"],
            availability_date=payload["availability_date"],
            net_share=payload["net_share"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "report_date": self.report_date.isoformat(),
            "availability_date": self.availability_date.isoformat(),
            "net_share": self.net_share,
        }


@dataclass(frozen=True)
class COTPolicyVariant:
    variant_id: str
    lookback_releases: int
    z_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "lookback_releases": self.lookback_releases,
            "z_threshold": self.z_threshold,
        }


# This tuple is the complete policy search space.  Do not generate additional
# thresholds or lookbacks from evaluation results.
CFTC_COT_VARIANTS: tuple[COTPolicyVariant, ...] = (
    COTPolicyVariant("cot_26w_z075", 26, 0.75),
    COTPolicyVariant("cot_26w_z125", 26, 1.25),
    COTPolicyVariant("cot_52w_z075", 52, 0.75),
    COTPolicyVariant("cot_52w_z125", 52, 1.25),
)
CFTC_COT_VARIANTS_BY_ID: Mapping[str, COTPolicyVariant] = MappingProxyType(
    {variant.variant_id: variant for variant in CFTC_COT_VARIANTS}
)


@dataclass(frozen=True)
class COTSignalDiagnostic:
    name: str
    direction: Literal["low", "high"]
    current_value: float
    history_count: int
    history_mean: float | None
    history_stddev: float | None
    z_score: float | None
    threshold: float
    triggered: bool
    status: Literal["ready", "insufficient_history", "zero_variance"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "current_value": self.current_value,
            "history_count": self.history_count,
            "history_mean": self.history_mean,
            "history_stddev": self.history_stddev,
            "z_score": self.z_score,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "status": self.status,
        }


@dataclass(frozen=True)
class COTPolicyDecision:
    variant_id: str
    decision_date: date
    stance: Stance
    target_exposure: float
    cash_votes: int
    required_cash_votes: int
    status: Literal[
        "ready",
        "no_complete_release",
        "insufficient_history",
        "stale_release",
    ]
    current_report_date: date | None
    current_availability_date: date | None
    complete_releases_available: int
    prior_releases_used: int
    signals: tuple[COTSignalDiagnostic, ...]
    method: str = POLICY_METHOD

    @property
    def action(self) -> Literal["BUY_ALL", "CASH_ALL"]:
        return "CASH_ALL" if self.stance == "CASH" else "BUY_ALL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "variant_id": self.variant_id,
            "decision_date": self.decision_date.isoformat(),
            "stance": self.stance,
            "action": self.action,
            "target_exposure": self.target_exposure,
            "cash_votes": self.cash_votes,
            "required_cash_votes": self.required_cash_votes,
            "status": self.status,
            "current_report_date": (
                self.current_report_date.isoformat() if self.current_report_date else None
            ),
            "current_availability_date": (
                self.current_availability_date.isoformat()
                if self.current_availability_date
                else None
            ),
            "complete_releases_available": self.complete_releases_available,
            "prior_releases_used": self.prior_releases_used,
            "signals": [signal.to_dict() for signal in self.signals],
        }


@dataclass(frozen=True)
class _CompleteRelease:
    report_date: date
    availability_date: date
    nasdaq_net_share: float
    sp500_net_share: float
    vix_net_share: float

    @property
    def nasdaq_sp500_divergence(self) -> float:
        return self.nasdaq_net_share - self.sp500_net_share


def get_cftc_cot_variant(variant: str | COTPolicyVariant) -> COTPolicyVariant:
    """Resolve a variant while rejecting any undeclared search-space expansion."""

    variant_id = variant.variant_id if isinstance(variant, COTPolicyVariant) else str(variant)
    declared = CFTC_COT_VARIANTS_BY_ID.get(variant_id)
    if declared is None:
        raise ValueError(
            f"unknown CFTC COT variant {variant_id!r}; allowed variants are "
            f"{list(CFTC_COT_VARIANTS_BY_ID)}"
        )
    if isinstance(variant, COTPolicyVariant) and variant != declared:
        raise ValueError(f"variant {variant_id!r} does not match its frozen declaration")
    return declared


def _normalize_records(
    records: Iterable[COTWeeklyRecord | Mapping[str, Any]],
) -> tuple[COTWeeklyRecord, ...]:
    normalized = tuple(
        item if isinstance(item, COTWeeklyRecord) else COTWeeklyRecord.from_mapping(item)
        for item in records
    )
    seen: set[tuple[date, Market]] = set()
    for item in normalized:
        key = (item.report_date, item.market)
        if key in seen:
            raise ValueError(
                "duplicate normalized COT record for "
                f"report_date={item.report_date.isoformat()} market={item.market}"
            )
        seen.add(key)
    return normalized


def _complete_releases(
    records: tuple[COTWeeklyRecord, ...], decision_date: date
) -> tuple[_CompleteRelease, ...]:
    grouped: dict[date, dict[Market, COTWeeklyRecord]] = {}
    for item in records:
        # The value and even the report's existence are unusable until its
        # conservative availability date.  Future releases are simply ignored.
        if item.availability_date > decision_date:
            continue
        grouped.setdefault(item.report_date, {})[item.market] = item

    complete: list[_CompleteRelease] = []
    for report_date, market_records in grouped.items():
        if any(market not in market_records for market in REQUIRED_MARKETS):
            continue
        availability_date = max(
            market_records[market].availability_date for market in REQUIRED_MARKETS
        )
        complete.append(
            _CompleteRelease(
                report_date=report_date,
                availability_date=availability_date,
                nasdaq_net_share=market_records["NASDAQ"].net_share,
                sp500_net_share=market_records["SP500"].net_share,
                vix_net_share=market_records["VIX"].net_share,
            )
        )
    return tuple(sorted(complete, key=lambda item: (item.availability_date, item.report_date)))


def _signal(
    *,
    name: str,
    direction: Literal["low", "high"],
    current_value: float,
    history: tuple[float, ...],
    lookback: int,
    threshold: float,
) -> COTSignalDiagnostic:
    if len(history) < lookback:
        return COTSignalDiagnostic(
            name=name,
            direction=direction,
            current_value=current_value,
            history_count=len(history),
            history_mean=None,
            history_stddev=None,
            z_score=None,
            threshold=threshold,
            triggered=False,
            status="insufficient_history",
        )

    window = history[-lookback:]
    mean = statistics.fmean(window)
    stddev = statistics.pstdev(window)
    if stddev == 0.0:
        return COTSignalDiagnostic(
            name=name,
            direction=direction,
            current_value=current_value,
            history_count=len(window),
            history_mean=mean,
            history_stddev=stddev,
            z_score=None,
            threshold=threshold,
            triggered=False,
            status="zero_variance",
        )

    z_score = (current_value - mean) / stddev
    triggered = z_score <= -threshold if direction == "low" else z_score >= threshold
    return COTSignalDiagnostic(
        name=name,
        direction=direction,
        current_value=current_value,
        history_count=len(window),
        history_mean=mean,
        history_stddev=stddev,
        z_score=z_score,
        threshold=threshold,
        triggered=triggered,
        status="ready",
    )


def evaluate_cftc_cot_policy(
    records: Iterable[COTWeeklyRecord | Mapping[str, Any]],
    *,
    decision_date: DateLike,
    variant: str | COTPolicyVariant,
) -> COTPolicyDecision:
    """Evaluate one frozen COT variant at a point in time.

    The current complete release is the last one available by ``decision_date``.
    Reference observations must have an availability date *strictly before* that
    current release, which also handles conservative same-day ties without using
    simultaneously released information as history.
    """

    resolved_variant = get_cftc_cot_variant(variant)
    as_of = _as_date(decision_date, field_name="decision_date")
    complete = _complete_releases(_normalize_records(records), as_of)
    if not complete:
        return COTPolicyDecision(
            variant_id=resolved_variant.variant_id,
            decision_date=as_of,
            stance="LONG",
            target_exposure=1.0,
            cash_votes=0,
            required_cash_votes=2,
            status="no_complete_release",
            current_report_date=None,
            current_availability_date=None,
            complete_releases_available=0,
            prior_releases_used=0,
            signals=(),
        )

    current = complete[-1]
    release_age_days = (as_of - current.availability_date).days
    if release_age_days > MAX_RELEASE_AGE_DAYS:
        # Stale sentiment must never keep an old risk-off call alive.  The
        # policy fails back to its unleveraged long baseline and emits no
        # signal votes for downstream consumers to misinterpret.
        return COTPolicyDecision(
            variant_id=resolved_variant.variant_id,
            decision_date=as_of,
            stance="LONG",
            target_exposure=1.0,
            cash_votes=0,
            required_cash_votes=2,
            status="stale_release",
            current_report_date=current.report_date,
            current_availability_date=current.availability_date,
            complete_releases_available=len(complete),
            prior_releases_used=0,
            signals=(),
        )

    prior = tuple(
        item for item in complete if item.availability_date < current.availability_date
    )
    lookback = resolved_variant.lookback_releases
    threshold = resolved_variant.z_threshold
    signals = (
        _signal(
            name="nasdaq_net_share_low",
            direction="low",
            current_value=current.nasdaq_net_share,
            history=tuple(item.nasdaq_net_share for item in prior),
            lookback=lookback,
            threshold=threshold,
        ),
        _signal(
            name="nasdaq_sp500_divergence_low",
            direction="low",
            current_value=current.nasdaq_sp500_divergence,
            history=tuple(item.nasdaq_sp500_divergence for item in prior),
            lookback=lookback,
            threshold=threshold,
        ),
        _signal(
            name="vix_net_share_high",
            direction="high",
            current_value=current.vix_net_share,
            history=tuple(item.vix_net_share for item in prior),
            lookback=lookback,
            threshold=threshold,
        ),
    )
    cash_votes = sum(int(signal.triggered) for signal in signals)
    ready = len(prior) >= lookback
    stance: Stance = "CASH" if ready and cash_votes >= 2 else "LONG"
    return COTPolicyDecision(
        variant_id=resolved_variant.variant_id,
        decision_date=as_of,
        stance=stance,
        target_exposure=0.0 if stance == "CASH" else 1.0,
        cash_votes=cash_votes,
        required_cash_votes=2,
        status="ready" if ready else "insufficient_history",
        current_report_date=current.report_date,
        current_availability_date=current.availability_date,
        complete_releases_available=len(complete),
        prior_releases_used=min(len(prior), lookback),
        signals=signals,
    )


def evaluate_all_cftc_cot_variants(
    records: Iterable[COTWeeklyRecord | Mapping[str, Any]],
    *,
    decision_date: DateLike,
) -> tuple[COTPolicyDecision, ...]:
    """Evaluate the complete four-variant family in its frozen order."""

    materialized = tuple(records)
    return tuple(
        evaluate_cftc_cot_policy(
            materialized,
            decision_date=decision_date,
            variant=variant,
        )
        for variant in CFTC_COT_VARIANTS
    )
