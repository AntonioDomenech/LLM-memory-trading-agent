from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Mapping

from .llm_client import embed_text
from .schemas import BenchmarkConfig, SecretConfig
from .storage import BenchmarkStore

LEGACY_MEMORY_NAMESPACE = "legacy"
LEGACY_POLICY_FINGERPRINT = "legacy"

# Dates, cash balances, and test windows deliberately do not belong here: a
# compatible policy must be able to reuse the same clean historical snapshot
# across independent chronological replays.
_POLICY_FINGERPRINT_FIELDS = (
    "memory_policy_version",
    "memory_feature_schema_version",
    "benchmark_contract_version",
    "mode",
    "symbol",
    "selected_symbols",
    "model",
    "fill_timing",
    "historical_price_basis",
    "historical_cadence",
    "live_frequency",
    "decision_cadence",
    "minimum_holding_days",
    "action_hysteresis_confirmations",
    "event_drawdown_trigger",
    "event_volatility_trigger",
    "decision_process",
    "single_stock_action_space",
    "allow_short",
    "max_leverage",
    "max_gross_exposure",
    "opportunity_cost_policy",
    "exposure_critic_enabled",
    "outcome_learning_mode",
    "online_learning_horizon_days",
    "online_policy_enabled",
    "online_policy_max_neighbors",
    "online_policy_min_samples",
    "online_policy_min_neighbor_separation_days",
    "online_policy_min_feature_overlap",
    "online_policy_risk_off_probability",
    "online_policy_min_confidence",
    "online_policy_min_active_return",
    "slippage_bps",
    "commission_per_trade",
    "commission_per_share",
    "memory_mode",
    "memory_retrieval",
    "prompt_detail_level",
    "max_daily_turnover",
    "turnover_edge_multiplier",
)


def build_policy_fingerprint(config: BenchmarkConfig) -> str:
    """Return a stable compatibility id for memories produced by *config*."""

    payload: Dict[str, Any] = {}
    for field in _POLICY_FINGERPRINT_FIELDS:
        value = getattr(config, field, None)
        if field == "selected_symbols":
            value = sorted(str(symbol).upper() for symbol in (value or []))
        elif field == "symbol":
            value = str(value or "").upper()
        payload[field] = value
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(size))
    norm_a = math.sqrt(sum(value * value for value in a[:size])) or 1.0
    norm_b = math.sqrt(sum(value * value for value in b[:size])) or 1.0
    return dot / (norm_a * norm_b)


def _flatten_features(value: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_features(item, path))
        else:
            flattened[path] = item
    return flattened


def _structured_similarity(query: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    """Compare point-in-time market state without using generated prose."""

    query_values = _flatten_features(query)
    candidate_values = _flatten_features(candidate)
    if not query_values:
        return 0.0

    scores: List[float] = []
    for key, expected in query_values.items():
        if key not in candidate_values:
            scores.append(0.0)
            continue
        actual = candidate_values[key]
        if (
            isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
        ):
            left = float(expected)
            right = float(actual)
            if not math.isfinite(left) or not math.isfinite(right):
                scores.append(1.0 if left == right else 0.0)
                continue
            scale = max(abs(left), abs(right), 1e-9)
            scores.append(max(0.0, 1.0 - abs(left - right) / scale))
        else:
            scores.append(1.0 if expected == actual else 0.0)
    return sum(scores) / len(scores)


class HybridMemory:
    def __init__(
        self,
        store: BenchmarkStore,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        *,
        run_id: str = "",
        memory_namespace: str | None = None,
        base_snapshot_id: str | None = None,
        policy_fingerprint: str | None = None,
        online_stream_id: str | None = None,
    ):
        self.store = store
        self.config = config
        self.secrets = secrets
        self.run_id = str(run_id or "")
        self.memory_namespace = str(
            memory_namespace if memory_namespace is not None else getattr(config, "memory_namespace", "legacy")
        ).strip() or LEGACY_MEMORY_NAMESPACE
        self.base_snapshot_id = str(
            base_snapshot_id if base_snapshot_id is not None else getattr(config, "memory_base_snapshot_id", "")
        ).strip()
        configured_stream_id = str(getattr(config, "memory_online_stream_id", "") or "").strip()
        self.online_stream_id = str(
            online_stream_id
            if online_stream_id is not None
            else (configured_stream_id or self.run_id)
        ).strip()

        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE:
            self.policy_fingerprint = LEGACY_POLICY_FINGERPRINT
            self.base_snapshot_id = ""
            # A caller may bind run_id universally; legacy configs retain their
            # historical unscoped behavior until explicitly opted into a named
            # namespace and snapshot.
            self.online_stream_id = ""
        else:
            if not self.base_snapshot_id:
                raise ValueError("Scoped memory requires memory_base_snapshot_id")
            self.policy_fingerprint = str(policy_fingerprint or build_policy_fingerprint(config))

    @property
    def model(self) -> str:
        return self.config.model or "unselected-model"

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def context(self) -> Dict[str, str]:
        return {
            "memory_namespace": self.memory_namespace,
            "policy_fingerprint": self.policy_fingerprint,
            "base_snapshot_id": self.base_snapshot_id,
            "online_stream_id": self.online_stream_id,
        }

    def add(
        self,
        *,
        portfolio_scope: str,
        decision_timestamp: str,
        knowledge_timestamp: str,
        source_run_id: str,
        memory_type: str,
        content: str = "",
        symbol: str = "",
        outcome_horizon: str = "",
        outcome_available_at: str = "",
        metadata: Dict[str, Any] | None = None,
        state_features: Dict[str, Any] | None = None,
        counterfactual_outcomes: Dict[str, Any] | None = None,
        memory_layer: str | None = None,
        pending_experience_id: int | None = None,
    ) -> int:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE:
            layer = memory_layer or "legacy"
            if layer != "legacy":
                raise ValueError("Base/online layers require a non-legacy memory namespace")
            online_stream_id = ""
        else:
            layer = memory_layer or "online"
            if layer not in {"base", "online"}:
                raise ValueError("Scoped memory must use the base or online layer")
            if layer == "online" and not self.online_stream_id:
                raise ValueError("Adding online memory requires an online stream or run id")
            online_stream_id = self.online_stream_id if layer == "online" else ""

        # Structured retrieval is intentionally independent of LLM prose and
        # therefore neither requires nor creates a text embedding.
        embedding: List[float] = []
        if self.config.memory_retrieval != "structured" and content:
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
            memory_namespace=self.memory_namespace,
            policy_fingerprint=self.policy_fingerprint,
            base_snapshot_id=self.base_snapshot_id,
            memory_layer=layer,
            online_stream_id=online_stream_id,
            state_features=state_features or {},
            counterfactual_outcomes=counterfactual_outcomes or {},
            pending_experience_id=pending_experience_id,
        )

    def add_base(self, **memory: Any) -> int:
        """Append a record to the explicitly named reusable base snapshot."""

        memory["memory_layer"] = "base"
        return self.add(**memory)

    def add_pending_experience(
        self,
        *,
        source_run_id: str,
        portfolio_scope: str,
        symbol: str,
        decision_timestamp: str,
        outcome_available_at: str,
        outcome_horizon: str,
        chosen_action: str,
        state_features: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE or not self.online_stream_id:
            raise ValueError("Pending online learning requires scoped memory and an online stream")
        return self.store.add_pending_experience(
            memory_namespace=self.memory_namespace,
            policy_fingerprint=self.policy_fingerprint,
            base_snapshot_id=self.base_snapshot_id,
            online_stream_id=self.online_stream_id,
            source_run_id=source_run_id,
            portfolio_scope=portfolio_scope,
            symbol=symbol,
            decision_timestamp=decision_timestamp,
            outcome_available_at=outcome_available_at,
            outcome_horizon=outcome_horizon,
            chosen_action=chosen_action,
            state_features=state_features,
            metadata=metadata,
        )

    def due_pending_experiences(self, *, as_of: str, limit: int = 200) -> List[Dict[str, Any]]:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE or not self.online_stream_id:
            return []
        return self.store.list_due_pending_experiences(
            memory_namespace=self.memory_namespace,
            policy_fingerprint=self.policy_fingerprint,
            base_snapshot_id=self.base_snapshot_id,
            online_stream_id=self.online_stream_id,
            as_of=as_of,
            limit=limit,
        )

    def recent_experiences(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE or not self.online_stream_id:
            return []
        return self.store.list_recent_pending_experiences(
            memory_namespace=self.memory_namespace,
            policy_fingerprint=self.policy_fingerprint,
            base_snapshot_id=self.base_snapshot_id,
            online_stream_id=self.online_stream_id,
            limit=limit,
        )

    def mark_experience_matured(
        self,
        experience_id: int,
        *,
        matured_at: str,
        counterfactual_outcomes: Dict[str, Any],
        matured_memory_id: int | None = None,
    ) -> bool:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE or not self.online_stream_id:
            return False
        return self.store.mark_pending_experience_matured(
            experience_id,
            memory_namespace=self.memory_namespace,
            policy_fingerprint=self.policy_fingerprint,
            base_snapshot_id=self.base_snapshot_id,
            online_stream_id=self.online_stream_id,
            matured_at=matured_at,
            counterfactual_outcomes=counterfactual_outcomes,
            matured_memory_id=matured_memory_id,
        )

    def load_live_state(self) -> Dict[str, Any] | None:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE or not self.online_stream_id:
            return None
        return self.store.get_live_state(
            memory_namespace=self.memory_namespace,
            policy_fingerprint=self.policy_fingerprint,
            base_snapshot_id=self.base_snapshot_id,
            online_stream_id=self.online_stream_id,
        )

    def register_base_snapshot(self, *, content_hash: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE:
            return {"content_hash": content_hash, "metadata": metadata}
        return self.store.register_base_snapshot(
            memory_namespace=self.memory_namespace,
            base_snapshot_id=self.base_snapshot_id,
            feature_schema_version=str(getattr(self.config, "memory_feature_schema_version", "legacy")),
            content_hash=content_hash,
            metadata=metadata,
        )

    def save_live_state(
        self,
        *,
        portfolio: Dict[str, Any],
        schedule_state: Dict[str, Any],
        last_snapshot_at: str,
    ) -> None:
        if self.memory_namespace == LEGACY_MEMORY_NAMESPACE or not self.online_stream_id:
            raise ValueError("Durable live state requires scoped memory and an online stream")
        self.store.save_live_state(
            memory_namespace=self.memory_namespace,
            policy_fingerprint=self.policy_fingerprint,
            base_snapshot_id=self.base_snapshot_id,
            online_stream_id=self.online_stream_id,
            portfolio=portfolio,
            schedule_state=schedule_state,
            last_snapshot_at=last_snapshot_at,
        )

    def retrieve(
        self,
        *,
        decision_timestamp: str,
        query: str = "",
        structured_query: Dict[str, Any] | None = None,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        scope: Dict[str, Any] = {
            "memory_namespace": self.memory_namespace,
            "policy_fingerprint": self.policy_fingerprint,
        }
        if self.memory_namespace != LEGACY_MEMORY_NAMESPACE:
            scope["base_snapshot_id"] = self.base_snapshot_id
            scope["online_stream_id"] = self.online_stream_id or None
        # Similarity retrieval must be allowed to rediscover an old crisis
        # regime instead of silently restricting itself to the latest 200
        # lessons. Recency-only and prose modes retain the bounded query.
        candidate_limit = (
            None
            if self.config.memory_retrieval == "structured" and structured_query
            else max(200, limit * 10)
        )
        candidates = self.store.eligible_memory(
            model=self.model,
            mode=self.mode,
            before_or_at=decision_timestamp,
            limit=candidate_limit,
            **scope,
        )
        if not candidates:
            return []

        if self.config.memory_retrieval == "structured":
            if not structured_query:
                return [self._without_embedding(item) for item in candidates[:limit]]
            ranked = [
                (_structured_similarity(structured_query, item.get("state_features") or {}), index, item)
                for index, item in enumerate(candidates)
            ]
            ranked.sort(key=lambda value: (-value[0], value[1]))
            return [self._with_score(item, score) for score, _, item in ranked[:limit]]

        query_embedding = embed_text(query, self.secrets, provider=self.config.embedding_provider) if query else []
        scored = []
        for index, item in enumerate(candidates):
            semantic = _cosine(query_embedding, item.get("embedding") or [])
            recency_hint = 0.02 if item.get("memory_type") == "lesson" else 0.0
            if structured_query:
                structured = _structured_similarity(structured_query, item.get("state_features") or {})
                score = 0.65 * semantic + 0.35 * structured + recency_hint
            else:
                score = semantic + recency_hint
            scored.append((score, index, item))
        scored.sort(key=lambda value: (-value[0], value[1]))
        return [self._with_score(item, score) for score, _, item in scored[:limit]]

    @staticmethod
    def _without_embedding(item: Dict[str, Any]) -> Dict[str, Any]:
        output = dict(item)
        output.pop("embedding", None)
        return output

    @classmethod
    def _with_score(cls, item: Dict[str, Any], score: float) -> Dict[str, Any]:
        output = cls._without_embedding(item)
        output["retrieval_score"] = round(score, 6)
        return output
