"""Shared daily pipeline utilities for training/backtest flows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .capsules import build_capsule
from .memory import MemoryBank
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
    memory_retrieval: Dict[str, Sequence[Dict[str, Any]]] = field(default_factory=dict)
    memory_highlights: Sequence[Dict[str, Any]] = field(default_factory=list)
    portfolio_state: Dict[str, Any] = field(default_factory=dict)


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


def _should_use_memory(retrieval_cfg: Any) -> bool:
    if retrieval_cfg is None:
        return False
    keys = ("k_shallow", "k_intermediate", "k_deep")
    for key in keys:
        if getattr(retrieval_cfg, key, 0) > 0:
            return True
    return False


_MEMORY_CACHE: Dict[Tuple[str, str], MemoryBank] = {}


def _resolve_memory_bank(cfg, memory_bank: Optional[MemoryBank]) -> Optional[MemoryBank]:
    if memory_bank is not None:
        return memory_bank
    path = getattr(cfg, "memory_path", None)
    if not path:
        return None
    emb_model = getattr(cfg, "embedding_model", "text-embedding-3-small")
    key = (path, emb_model)
    bank = _MEMORY_CACHE.get(key)
    if bank is None:
        bank = MemoryBank(path=path, emb_model=emb_model)
        _MEMORY_CACHE[key] = bank
    return bank


def _sanitize_memory_items(layer: str, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for it in items or []:
        sanitized.append(
            {
                "layer": layer,
                "id": it.get("id"),
                "text": it.get("text", ""),
                "meta": it.get("meta", {}) or {},
                "importance": float(it.get("importance", 0.0) or 0.0),
                "seen_date": it.get("seen_date"),
                "access": int(it.get("access", 0) or 0),
            }
        )
    return sanitized


def prepare_daily_context(
    cfg,
    date_iso: str,
    price_row: Dict[str, Any],
    *,
    memory_bank: Optional[MemoryBank] = None,
    portfolio_state: Optional[Dict[str, Any]] = None,
) -> DailyContext:
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

    retrieval_cfg = getattr(cfg, "retrieval", None)
    memory_layers: Dict[str, Sequence[Dict[str, Any]]] = {
        "shallow": [],
        "intermediate": [],
        "deep": [],
    }
    memory_highlights: List[Dict[str, Any]] = []

    if _should_use_memory(retrieval_cfg):
        bank = _resolve_memory_bank(cfg, memory_bank)
        if bank is not None:
            query = json.dumps(capsule, sort_keys=True)
            try:
                shallow, intermediate, deep = bank.retrieve(query, date_iso, retrieval_cfg)
            except Exception:
                shallow, intermediate, deep = [], [], []
            memory_layers = {
                "shallow": _sanitize_memory_items("shallow", shallow),
                "intermediate": _sanitize_memory_items("intermediate", intermediate),
                "deep": _sanitize_memory_items("deep", deep),
            }
            for layer in ("shallow", "intermediate", "deep"):
                for item in memory_layers[layer]:
                    memory_highlights.append(
                        {
                            "layer": layer,
                            "text": item.get("text", ""),
                            "meta": item.get("meta", {}),
                            "seen_date": item.get("seen_date"),
                            "importance": item.get("importance", 0.0),
                        }
                    )

    factor_payload = dict(capsule)
    factor_payload["memory_highlights"] = memory_highlights
    portfolio_payload = dict(portfolio_state or {})
    policy_payload = {
        "capsule": capsule,
        "memory_highlights": memory_highlights,
        "portfolio_state": portfolio_payload,
    }

    factor_prompt = PromptBundle(
        system={"role": "system", "content": _load_prompt(os.path.join("prompts", "factor_head.txt"))},
        user={"role": "user", "content": json.dumps(factor_payload, sort_keys=True)},
    )
    policy_prompt = PromptBundle(
        system={"role": "system", "content": _load_prompt(os.path.join("prompts", "policy_head.txt"))},
        user={"role": "user", "content": json.dumps(policy_payload, sort_keys=True)},
    )

    return DailyContext(
        capsule=capsule,
        provider_label=provider_used,
        news_reason=reason,
        articles=article_list,
        memory_retrieval=memory_layers,
        memory_highlights=memory_highlights,
        portfolio_state=portfolio_payload,
        factor_prompt=factor_prompt,
        policy_prompt=policy_prompt,
    )
