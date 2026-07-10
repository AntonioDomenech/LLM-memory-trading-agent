"""Prompt blinding for historical LLM evaluation.

Modern local language models can contain facts from after a benchmark's
selection cutoff.  Merely filtering the warehouse therefore does not prevent
parametric look-ahead: a prompt containing an issuer and an exact historical
date can invite the model to recall what happened next.

This module produces a deliberately lossy prompt view for non-live historical
evaluation.  It removes issuer/context identities, converts calendar dates to
relative offsets, and strips raw amounts that can fingerprint a well-known
price path or financial statement.  Normalized returns, volatility, drawdown,
ratios, ranks, probabilities, and pre-cutoff outcomes remain available.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from .schemas import BenchmarkConfig


_DATE_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2}-\d{2}-\d{2})(?:[T ][0-9:.+\-Z]+)?")
_COMPACT_DATE_PATTERN = re.compile(r"(?<![\d.])(?:19|20)\d{6}(?!\d)")
_ISO_WEEK_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}-W\d{1,2}(?!\d)", re.IGNORECASE)
_SLASH_DATE_PATTERN = re.compile(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}(?!\d)")
_MONTH_DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(?:\d{1,2}(?:st|nd|rd|th)?(?:,|\s)+)?(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

HISTORICAL_BLINDING_CONTRACT = "identity_relative_time_scale_free_v2"

# Absolute values that are unnecessary for directional reasoning and can make
# a famous historical state identifiable.  Derived, scale-free fields such as
# returns, distances, z-scores, weights, ranks, and probabilities are retained.
_DROP_VALUE_KEYS = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "price",
    "entry_price",
    "exit_price",
    "volume",
    "cash",
    "equity",
    "initial_cash",
    "position_shares",
    "shares",
    "signed_delta",
    "requested_shares",
    "affordable_shares",
    "commission",
    "fees",
    "slippage_cost",
    "notional",
    "amount",
    "cost",
    "proceeds",
    "value",
}

_DROP_VALUE_SUFFIXES = (
    "_price",
    "_prices",
    "_shares",
    "_volume",
    "_notional",
    "_amount",
)

_FREE_TEXT_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
)

_AAPL_SEMANTIC_TERMS = (
    "AirPods",
    "App Store",
    "Apple Watch",
    "Cupertino",
    "iCloud",
    "iOS",
    "iPad",
    "iPhone",
    "MacBook",
    "macOS",
    "Siri",
    "Steve Jobs",
    "Tim Cook",
    "Vision Pro",
)

_HISTORICAL_TEXT_REDACTED = "HISTORICAL_TEXT_REDACTED"
_ALLOWED_OUTPUT_STRING_KEYS = {"action", "stage1_alignment", "stance", "symbol"}
_ALLOWED_OUTPUT_STRING_VALUES = {
    "bearish",
    "bullish",
    "BUY_ALL",
    "CASH_ALL",
    "follow",
    "HOLD",
    "neutral",
    "partial",
    "uncertain",
    "veto",
}

_ALLOWED_NUMERIC_EXACT = {
    "alpha",
    "beta",
    "cash_weight",
    "confidence",
    "effective_sample_size",
    "eligible_lesson_count",
    "horizon_days",
    "lower_bound",
    "memory_cases",
    "memory_horizon",
    "price_observations",
    "rank",
    "sample_size",
    "upper_bound",
}

_ALLOWED_NUMERIC_CONTAINERS = {
    "current_position_weights",
    "no_trade_target_weights",
    "recommended_exposure_band",
    "suggested_exposure_band",
    "target_weights",
    "valid_target_exposure_range",
}

_ALLOWED_NUMERIC_EXACT.update(
    {
        "action_hysteresis_confirmations",
        "cash_active_return",
        "cash_outperformance_probability",
        "current_exposure",
        "current_weight",
        "downside_rate",
        "estimated_slippage_cost_bps",
        "estimated_turnover",
        "event_drawdown_trigger",
        "event_volatility_trigger",
        "expected_active_return",
        "expected_holding_days",
        "expected_return_bps",
        "gross_exposure",
        "hit_rate",
        "loss_probability",
        "max_daily_turnover",
        "max_gross_exposure",
        "max_nonzero_positions",
        "mean_return",
        "memory_base_rate_return",
        "memory_confidence",
        "memory_downside_rate",
        "memory_hit_rate",
        "memory_mean_return",
        "minimum_holding_days",
        "net_exposure",
        "probability_lower_bound",
        "probability_upper_bound",
        "proposed_target_weight",
        "risk_off_probability",
        "score",
        "signal_rank",
        "signal_score",
        "slippage_bps",
        "target_exposure",
        "turnover_edge_multiplier",
        "turnover_prompt_buffer",
    }
)

_ALLOWED_OUTPUT_NUMERIC_KEYS = {
    "cash_weight",
    "confidence",
    "estimated_slippage_cost_bps",
    "estimated_turnover",
    "expected_holding_days",
    "expected_return_bps",
    "gross_exposure",
    "horizon_days",
    "net_exposure",
    "proposed_target_weight",
    "target_exposure",
}

_KNOWN_ALIAS_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:ASSET|MARKET|SENTIMENT|RATES)_\d+(?![A-Za-z])",
    re.IGNORECASE,
)

# The current prompt exposes raw SEC amounts.  Until a separately validated
# point-in-time ratio/surprise feature contract exists, omit those amounts from
# the LLM view instead of allowing issuer fingerprinting.
_DROP_SECTION_KEYS = {"fundamentals"}

_FIXED_ALIASES = {
    "SPY": "MARKET_1",
    "QQQ": "MARKET_2",
    "IWM": "MARKET_3",
    "DIA": "MARKET_4",
    "^VIX": "SENTIMENT_1",
    "VIX": "SENTIMENT_1",
    "^TNX": "RATES_1",
    "TNX": "RATES_1",
}


@dataclass(frozen=True)
class BlindedPrompt:
    system: str
    user: str
    aliases_to_real: Mapping[str, str]
    anchor_date: str
    removed_absolute_fields: tuple[str, ...]
    removed_sections: tuple[str, ...]

    @property
    def applied(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        prompt_hash = hashlib.sha256(f"{self.system}\n{self.user}".encode("utf-8")).hexdigest()
        alias_hash = hashlib.sha256(
            json.dumps(
                sorted((str(alias), str(real)) for alias, real in self.aliases_to_real.items()),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "applied": True,
            "mode": HISTORICAL_BLINDING_CONTRACT,
            "contract_version": HISTORICAL_BLINDING_CONTRACT,
            "sanitized_prompt_sha256": f"sha256:{prompt_hash}",
            "alias_map_sha256": f"sha256:{alias_hash}",
            "anchor_exposed": False,
            "asset_identity_exposed": False,
            "absolute_market_values_exposed": False,
            "raw_fundamental_amounts_exposed": False,
            "removed_absolute_fields": list(self.removed_absolute_fields),
            "removed_sections": list(self.removed_sections),
        }


def should_blind_historical_prompt(config: BenchmarkConfig) -> bool:
    return bool(
        config.historical_prompt_blinding
        and config.evaluation_mode
        in {"training_diagnostic", "frozen_holdout", "causal_online_replay"}
    )


def blind_historical_prompt(
    config: BenchmarkConfig,
    system: str,
    user: str,
) -> BlindedPrompt:
    """Return the exact prompt view that may be sent to the historical LLM."""

    try:
        payload = json.loads(user)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = user

    anchor = _find_anchor_date(payload) or config.test_start
    real_to_alias, aliases_to_real = _alias_maps(config)
    blinded_payload = _blind_value(
        payload,
        anchor=anchor,
        real_to_alias=real_to_alias,
        key_name="",
    )
    blinded_system = _blind_string(system, anchor=anchor, real_to_alias=real_to_alias)
    if isinstance(blinded_payload, str):
        blinded_user = blinded_payload
    else:
        blinded_user = json.dumps(blinded_payload, sort_keys=True, default=str)
    _validate_blinded_prompt(
        blinded_system,
        blinded_user,
        real_to_alias=real_to_alias,
    )
    return BlindedPrompt(
        system=blinded_system,
        user=blinded_user,
        aliases_to_real=aliases_to_real,
        anchor_date=anchor,
        removed_absolute_fields=tuple(sorted(_DROP_VALUE_KEYS)),
        removed_sections=tuple(sorted(_DROP_SECTION_KEYS)),
    )


def restore_historical_output(value: Any, aliases_to_real: Mapping[str, str]) -> Any:
    """Restore engine identifiers in parsed model output, not calendar dates."""

    _validate_model_output(value, aliases_to_real)
    sanitized = _sanitize_model_output(value, aliases_to_real, key_name="")
    return _restore_historical_value(sanitized, aliases_to_real)


def _restore_historical_value(value: Any, aliases_to_real: Mapping[str, str]) -> Any:

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        seen: set[str] = set()
        for key, item in value.items():
            restored_key = _restore_string(str(key), aliases_to_real)
            collision_key = restored_key.casefold()
            if collision_key in seen:
                raise ValueError(f"Historical output alias collision for key {restored_key!r}")
            seen.add(collision_key)
            output[restored_key] = _restore_historical_value(item, aliases_to_real)
        return output
    if isinstance(value, list):
        return [_restore_historical_value(item, aliases_to_real) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_historical_value(item, aliases_to_real) for item in value)
    if isinstance(value, str):
        return _restore_string(value, aliases_to_real)
    return value


def _alias_maps(config: BenchmarkConfig) -> tuple[dict[str, str], dict[str, str]]:
    asset = str(config.symbol or "AAPL").upper()
    company = str(config.company_name or "").strip()
    real_to_alias = dict(_FIXED_ALIASES)
    real_to_alias[asset] = "ASSET_1"
    if company:
        real_to_alias[company] = "ANONYMOUS_COMPANY"
        real_to_alias[f"{company} Inc"] = "ANONYMOUS_COMPANY"
        real_to_alias[f"{company} Inc."] = "ANONYMOUS_COMPANY"
    aliases_to_real = {
        "ASSET_1": asset,
        "ANONYMOUS_COMPANY": company or asset,
        "MARKET_1": "SPY",
        "MARKET_2": "QQQ",
        "MARKET_3": "IWM",
        "MARKET_4": "DIA",
        "SENTIMENT_1": "^VIX",
        "RATES_1": "^TNX",
    }
    return real_to_alias, aliases_to_real


def _find_anchor_date(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("decision_date", "as_of_date"):
            candidate = value.get(key)
            if isinstance(candidate, str) and _DATE_PATTERN.search(candidate):
                return _DATE_PATTERN.search(candidate).group(1)
        for item in value.values():
            candidate = _find_anchor_date(item)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for item in value:
            candidate = _find_anchor_date(item)
            if candidate:
                return candidate
    return ""


def _blind_value(value: Any, *, anchor: str, real_to_alias: Mapping[str, str], key_name: str) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.casefold()
            if (
                lowered in _DROP_SECTION_KEYS
                or lowered in _DROP_VALUE_KEYS
                or lowered.endswith(_DROP_VALUE_SUFFIXES)
            ):
                continue
            blinded_key = _blind_string(key, anchor=anchor, real_to_alias=real_to_alias)
            if lowered == "positions" and isinstance(item, dict):
                out[blinded_key] = {
                    _blind_string(str(symbol), anchor=anchor, real_to_alias=real_to_alias): (
                        "LONG" if _number(shares) > 0 else "SHORT" if _number(shares) < 0 else "FLAT"
                    )
                    for symbol, shares in item.items()
                }
                continue
            if lowered in {"name", "company_name"}:
                out[blinded_key] = "ANONYMOUS_COMPANY"
                continue
            if lowered == "official_success_metric" and isinstance(item, dict):
                out[blinded_key] = {
                    "comparison": "same_anonymous_asset_buy_and_hold",
                    "requirement": "Outperform the same anonymous asset after declared costs.",
                }
                continue
            if _is_number(item) and not _numeric_field_allowed(lowered, key_name):
                continue
            out[blinded_key] = _blind_value(
                item,
                anchor=anchor,
                real_to_alias=real_to_alias,
                key_name=lowered,
            )
        return out
    if isinstance(value, list):
        output = []
        for item in value:
            if _is_number(item) and not _numeric_field_allowed(key_name, key_name):
                continue
            output.append(
                _blind_value(item, anchor=anchor, real_to_alias=real_to_alias, key_name=key_name)
            )
        return output
    if isinstance(value, tuple):
        return tuple(
            _blind_value(item, anchor=anchor, real_to_alias=real_to_alias, key_name=key_name)
            for item in value
            if not _is_number(item) or _numeric_field_allowed(key_name, key_name)
        )
    if isinstance(value, str):
        return _blind_string(
            value,
            anchor=anchor,
            real_to_alias=real_to_alias,
            redact_free_text_numbers=True,
        )
    return value


def _blind_string(
    value: str,
    *,
    anchor: str,
    real_to_alias: Mapping[str, str],
    redact_free_text_numbers: bool = False,
) -> str:
    text = str(value)
    identity_pairs = sorted(real_to_alias.items(), key=lambda item: len(item[0]), reverse=True)
    for real, alias in identity_pairs:
        if not real:
            continue
        pattern = re.escape(real) if real.startswith("^") else _identifier_pattern(real)

        def replace_identity(match: re.Match[str]) -> str:
            suffix = "_" if match.end() < len(text) and text[match.end()].isdigit() else ""
            return f"{alias}{suffix}"

        text = re.sub(pattern, replace_identity, text, flags=re.IGNORECASE)

    if real_to_alias.get("AAPL") == "ASSET_1":
        for term in _AAPL_SEMANTIC_TERMS:
            text = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
                "ISSUER_TERM_REDACTED",
                text,
                flags=re.IGNORECASE,
            )

    try:
        anchor_value = date.fromisoformat(anchor)
    except ValueError:
        anchor_value = None

    def replace_date(match: re.Match[str]) -> str:
        if anchor_value is None:
            return "RELATIVE_DATE"
        try:
            delta = (date.fromisoformat(match.group(1)) - anchor_value).days
        except ValueError:
            return "RELATIVE_DATE"
        if delta == 0:
            return "T0"
        return f"T{delta:+d}D"

    text = _MONTH_DATE_PATTERN.sub("NATURAL_DATE_REDACTED", text)
    text = _ISO_WEEK_PATTERN.sub("ISO_WEEK_REDACTED", text)
    text = _SLASH_DATE_PATTERN.sub("SLASH_DATE_REDACTED", text)
    text = _COMPACT_DATE_PATTERN.sub("COMPACT_DATE_REDACTED", text)
    text = _DATE_PATTERN.sub(replace_date, text)
    if redact_free_text_numbers:
        text = _FREE_TEXT_NUMBER.sub(_redact_free_text_number, text)
    return text


def _restore_string(value: str, aliases_to_real: Mapping[str, str]) -> str:
    text = str(value)
    for alias, real in sorted(aliases_to_real.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(_identifier_pattern(alias), real, text, flags=re.IGNORECASE)
    return text


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _redact_free_text_number(match: re.Match[str]) -> str:
    # Free text has no machine-verifiable unit. Even a unit-scale number can be
    # a raw EPS value or penny-stock price, so retain numerical evidence only in
    # explicitly allowlisted structured fields.
    return "SCALE_REDACTED"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_field_allowed(key: str, parent: str) -> bool:
    key = str(key or "").casefold()
    parent = str(parent or "").casefold()
    if parent in _ALLOWED_NUMERIC_CONTAINERS:
        return True
    if key in _ALLOWED_NUMERIC_EXACT:
        return True
    if re.fullmatch(r"r(?:1|5|10|20|60|63|120|126|252)", key):
        return True
    if re.fullmatch(r"vol(?:5|10|20|60|63|120|126|252)", key):
        return True
    if re.fullmatch(r"outcome_(?:1|5|20|60)(?:d)?", key):
        return True
    if re.fullmatch(
        r"(?:(?:aapl|spy|qqq)_|(?:asset|market)_\d+_)?return_(?:1|5|20|60|120|252)d",
        key,
    ):
        return True
    if re.fullmatch(r"volatility_(?:5|10|20|60|63|120|126|252)d", key):
        return True
    if re.fullmatch(r"sma(?:20|50|100|150|200)_distance", key):
        return True
    if re.fullmatch(r"drawdown_(?:20|60|120|252)d", key):
        return True
    return key in {"gap_1d", "volume_z20"}


def _validate_blinded_prompt(
    system: str,
    user: str,
    *,
    real_to_alias: Mapping[str, str],
) -> None:
    combined = f"{system}\n{user}"
    _reject_real_identities_or_dates(combined, real_to_alias.keys(), field_name="prompt")
    try:
        payload = json.loads(user)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = user
    _validate_blinded_user_value(payload, key_name="")


def _validate_blinded_user_value(value: Any, *, key_name: str) -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            if (
                key in _DROP_SECTION_KEYS
                or key in _DROP_VALUE_KEYS
                or key.endswith(_DROP_VALUE_SUFFIXES)
            ):
                raise ValueError(f"Historical prompt retained forbidden field {raw_key!r}")
            if _is_number(item) and not _numeric_field_allowed(key, key_name):
                raise ValueError(f"Historical prompt retained non-scale-free numeric field {raw_key!r}")
            _validate_blinded_user_value(item, key_name=key)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if _is_number(item) and not _numeric_field_allowed(key_name, key_name):
                raise ValueError(f"Historical prompt retained non-scale-free numeric list {key_name!r}")
            _validate_blinded_user_value(item, key_name=key_name)
        return
    if isinstance(value, str):
        if _FREE_TEXT_NUMBER.search(value):
            raise ValueError("Historical prompt retained a number in free text")


def _validate_model_output(value: Any, aliases_to_real: Mapping[str, str]) -> None:
    known_aliases = {alias.casefold() for alias in aliases_to_real}
    real_identities = {
        *[real for real in aliases_to_real.values() if real],
        *_FIXED_ALIASES.keys(),
    }
    semantic_terms = (
        _AAPL_SEMANTIC_TERMS
        if str(aliases_to_real.get("ASSET_1") or "").upper() == "AAPL"
        else ()
    )

    def visit(item: Any, *, key_name: str = "", parent_name: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                inspect_text(str(key))
                lowered = str(key).casefold()
                if _is_number(child) and not _output_numeric_allowed(lowered, key_name):
                    raise ValueError(f"Unknown numerical field in historical model output: {key!r}")
                visit(child, key_name=lowered, parent_name=key_name)
        elif isinstance(item, (list, tuple)):
            for child in item:
                if _is_number(child) and not _output_numeric_allowed(key_name, parent_name):
                    raise ValueError(
                        f"Unknown numerical list in historical model output: {key_name!r}"
                    )
                visit(child, key_name=key_name, parent_name=parent_name)
        elif isinstance(item, str):
            inspect_text(item)

    def inspect_text(text: str) -> None:
        _reject_real_identities_or_dates(text, real_identities, field_name="model output")
        for term in semantic_terms:
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
                text,
                flags=re.IGNORECASE,
            ):
                raise ValueError(f"Historical model output exposed issuer term {term!r}")
        for match in _KNOWN_ALIAS_PATTERN.finditer(text):
            if match.group(0).casefold() not in known_aliases:
                raise ValueError(f"Unknown historical output alias {match.group(0)!r}")

    visit(value)


def _sanitize_model_output(
    value: Any,
    aliases_to_real: Mapping[str, str],
    *,
    key_name: str,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_model_output(item, aliases_to_real, key_name=str(key).casefold())
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_model_output(item, aliases_to_real, key_name=key_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_model_output(item, aliases_to_real, key_name=key_name)
            for item in value
        )
    if isinstance(value, str):
        known_aliases = {alias.casefold() for alias in aliases_to_real}
        if key_name in _ALLOWED_OUTPUT_STRING_KEYS and (
            value.casefold() in known_aliases or value in _ALLOWED_OUTPUT_STRING_VALUES
        ):
            return value
        return _HISTORICAL_TEXT_REDACTED
    return value


def _reject_real_identities_or_dates(
    text: str,
    identities: Any,
    *,
    field_name: str,
) -> None:
    for identity in sorted({str(item) for item in identities if item}, key=len, reverse=True):
        if identity.startswith("^"):
            matched = re.search(re.escape(identity), text, flags=re.IGNORECASE)
        else:
            matched = re.search(_identifier_pattern(identity), text, flags=re.IGNORECASE)
        if matched:
            raise ValueError(f"Historical {field_name} retained real identity {identity!r}")
    for pattern in (
        _DATE_PATTERN,
        _COMPACT_DATE_PATTERN,
        _ISO_WEEK_PATTERN,
        _SLASH_DATE_PATTERN,
        _MONTH_DATE_PATTERN,
    ):
        if pattern.search(text):
            raise ValueError(f"Historical {field_name} retained an exact calendar date")


def _identifier_pattern(value: str) -> str:
    return rf"(?<![A-Za-z]){re.escape(value)}(?![A-Za-z])"


def _output_numeric_allowed(key: str, parent: str) -> bool:
    key = str(key or "").casefold()
    parent = str(parent or "").casefold()
    if key in {"recommended_exposure_band", "target_weights"} or parent in {
        "recommended_exposure_band",
        "target_weights",
    }:
        return True
    return key in _ALLOWED_OUTPUT_NUMERIC_KEYS
