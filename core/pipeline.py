"""Shared daily pipeline utilities for training/backtest flows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from .capsules import build_capsule
from .news_fetcher import fetch_news_with_reason


@dataclass
class PromptBundle:
    """Container for a system/user prompt pair."""

    system: Dict[str, Any]
    user: Dict[str, Any]

    def as_messages(self) -> List[Dict[str, Any]]:
        return [self.system, self.user]


@dataclass
class DailyContext:
    """Structured data produced for a trading day."""

    capsule: Dict[str, Any]
    provider_label: str
    news_reason: str
    articles: Sequence[Dict[str, Any]]
    factor_prompt: PromptBundle
    policy_prompt: PromptBundle


@lru_cache(maxsize=None)
def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _prov_label(reason: str) -> str:
    parts = (reason or "").split(":")
    if not parts:
        return "unknown"
    if parts[0] == "cache" and len(parts) > 1:
        return f"cache->{parts[1]}"
    return parts[0]


def _build_regime(price_row: Dict[str, Any]) -> Dict[str, Any]:
    price_val = price_row.get("price", price_row.get("close", 1.0))
    try:
        price = float(price_val)
    except Exception:
        price = 1.0
    price = price if price != 0 else 1.0
    atr = float(price_row.get("atr", 0.0) or 0.0)
    trend_up = 1 if price_row.get("trend_up", 0) == 1 else 0
    return {
        "market": "up" if trend_up == 1 else "down_or_sideways",
        "vol_bucket": "high" if (atr / max(1.0, price)) > 0.02 else "low",
    }


def prepare_daily_context(cfg, date_iso: str, price_row: Dict[str, Any]) -> DailyContext:
    """Fetch headlines, build a capsule and prepare prompts for a trading day."""

    articles: Sequence[Dict[str, Any]]
    reason: str
    provider_used = "Off"

    if getattr(cfg, "K_news_per_day", 0) > 0 and getattr(cfg, "news_source", "Off") != "Off":
        articles, reason = fetch_news_with_reason(cfg.symbol, date_iso, cfg.K_news_per_day)
        provider_used = _prov_label(reason)
    else:
        articles, reason = [], "K_news_per_day=0"

    regime = _build_regime(price_row)
    article_list = list(articles)
    capsule = build_capsule(date_iso, cfg.symbol, price_row, article_list, [], regime)
    capsule["headlines_source"] = provider_used

    factor_prompt = PromptBundle(
        system={"role": "system", "content": _load_prompt(os.path.join("prompts", "factor_head.txt"))},
        user={"role": "user", "content": json.dumps(capsule, sort_keys=True)},
    )
    policy_prompt = PromptBundle(
        system={"role": "system", "content": _load_prompt(os.path.join("prompts", "policy_head.txt"))},
        user={"role": "user", "content": json.dumps({"capsule": capsule})},
    )

    return DailyContext(
        capsule=capsule,
        provider_label=provider_used,
        news_reason=reason,
        articles=article_list,
        factor_prompt=factor_prompt,
        policy_prompt=policy_prompt,
    )
