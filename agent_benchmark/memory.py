from __future__ import annotations

import math
from typing import Any, Dict, List

from .llm_client import embed_text
from .schemas import BenchmarkConfig, SecretConfig
from .storage import BenchmarkStore


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(size))
    norm_a = math.sqrt(sum(value * value for value in a[:size])) or 1.0
    norm_b = math.sqrt(sum(value * value for value in b[:size])) or 1.0
    return dot / (norm_a * norm_b)


class HybridMemory:
    def __init__(self, store: BenchmarkStore, config: BenchmarkConfig, secrets: SecretConfig):
        self.store = store
        self.config = config
        self.secrets = secrets

    @property
    def model(self) -> str:
        return self.config.model or "unselected-model"

    @property
    def mode(self) -> str:
        return self.config.mode

    def add(
        self,
        *,
        portfolio_scope: str,
        decision_timestamp: str,
        knowledge_timestamp: str,
        source_run_id: str,
        memory_type: str,
        content: str,
        symbol: str = "",
        outcome_horizon: str = "",
        outcome_available_at: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        embedding = embed_text(content, self.secrets, provider=self.config.embedding_provider)
        return self.store.add_memory(
            model=self.model,
            mode=self.mode,
            portfolio_scope=portfolio_scope,
            symbol=symbol,
            decision_timestamp=decision_timestamp,
            knowledge_timestamp=knowledge_timestamp,
            source_run_id=source_run_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            outcome_horizon=outcome_horizon,
            outcome_available_at=outcome_available_at,
            metadata=metadata or {},
        )

    def retrieve(self, *, decision_timestamp: str, query: str, limit: int = 12) -> List[Dict[str, Any]]:
        candidates = self.store.eligible_memory(
            model=self.model,
            mode=self.mode,
            before_or_at=decision_timestamp,
            limit=max(200, limit * 10),
        )
        if not candidates:
            return []
        if self.config.memory_retrieval == "structured":
            return candidates[:limit]
        query_embedding = embed_text(query, self.secrets, provider=self.config.embedding_provider)
        scored = []
        for item in candidates:
            semantic = _cosine(query_embedding, item.get("embedding") or [])
            recency_hint = 0.02 if item.get("memory_type") == "lesson" else 0.0
            scored.append((semantic + recency_hint, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        output = []
        for score, item in scored[:limit]:
            slim = dict(item)
            slim.pop("embedding", None)
            slim["retrieval_score"] = round(score, 6)
            output.append(slim)
        return output
