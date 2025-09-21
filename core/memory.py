
import json, os, math, hashlib, datetime as _dt, numpy as np
from .logger import get_logger
from .embeddings import embed_texts
log = get_logger()


def _deterministic_id(text: str) -> str:
    """Return a stable identifier for the provided text."""
    normalized = (text or "").strip().encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()


def _parse_iso_date(value):
    """Return ``datetime.date`` for ISO strings or ``None`` when parsing fails."""

    if not isinstance(value, str) or not value:
        return None
    try:
        return _dt.datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _item_date(item):
    """Derive the most relevant date attached to a memory item."""

    if not isinstance(item, dict):
        return None
    seen = _parse_iso_date(item.get("seen_date"))
    if seen is not None:
        return seen
    meta = item.get("meta") or {}
    if isinstance(meta, dict):
        return _parse_iso_date(meta.get("date"))
    return None
class MemoryBank:
    """Persistent multi-layer memory store for daily trading context."""

    def __init__(self, path="data/memory_bank.json", emb_model="text-embedding-3-small"):
        """Create a memory bank backed by ``path`` using ``emb_model`` for vectors."""

        self.path = path
        self.emb_model = emb_model
        self.layers = {"shallow": [], "intermediate": [], "deep": []}
        self.load()

    def load(self) -> None:
        """Load memory items from disk, keeping an empty structure on failure."""

        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.layers = json.load(f)
            except Exception:
                log.warning("Could not read memory_bank.json; starting fresh.")

    def save(self) -> None:
        """Persist the memory layers to ``self.path``."""

        directory = os.path.dirname(self.path) or "."
        if directory not in ("", "."):
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.layers, f, indent=2)

    def _embed(self, texts):
        """Embed ``texts`` using the configured embedding model."""

        return embed_texts(texts, model=self.emb_model)

    def add_item(self, layer: str, text: str, meta, base_importance: float = 10.0, seen_date: str = None):
        """Insert or update a memory item in ``layer``."""

        clean_text = (text or "").strip()
        item_id = _deterministic_id(clean_text)
        meta = meta or {}
        layer_items = self.layers.setdefault(layer, [])
        target_date = meta.get("date")

        existing = None
        for it in layer_items:
            if it.get("id") == item_id:
                existing = it
                break
            existing_meta = it.get("meta") or {}
            if target_date is not None and existing_meta.get("date") == target_date:
                existing = it
                break

        embedding = self._embed([clean_text])[0]
        if existing is not None:
            existing.update(
                {
                    "id": item_id,
                    "text": clean_text,
                    "meta": meta,
                    "importance": float(base_importance),
                    "seen_date": seen_date,
                    "embedding": embedding,
                }
            )
        else:
            layer_items.append(
                {
                    "id": item_id,
                    "text": clean_text,
                    "meta": meta,
                    "importance": float(base_importance),
                    "seen_date": seen_date,
                    "access": 0,
                    "embedding": embedding,
                }
            )
        self.save()

    def promote(self, n_sh_to_int=5, n_int_to_deep=2):
        """Promote top ``n`` items from shallow→intermediate and intermediate→deep."""

        promotions = [
            ("shallow", "intermediate", n_sh_to_int),
            ("intermediate", "deep", n_int_to_deep),
        ]
        for src, dst, count in promotions:
            sorted_src = sorted(self.layers[src], key=lambda x: x.get("importance", 0), reverse=True)
            top = sorted_src[:count]
            self.layers[src] = sorted_src[count:]
            self.layers[dst].extend(top)
        self.save()

    def prune_layer(self, layer, *, min_importance=None, before_date=None, max_items=None):
        """Remove stale items from ``layer`` based on importance, recency or size."""

        if layer not in self.layers:
            return 0

        items = list(self.layers.get(layer) or [])
        if not items:
            return 0

        if min_importance is not None:
            try:
                min_importance = float(min_importance)
            except (TypeError, ValueError) as exc:
                raise ValueError("min_importance must be numeric") from exc

        cutoff = None
        if before_date is not None:
            cutoff = _parse_iso_date(before_date)
            if cutoff is None:
                raise ValueError("before_date must be YYYY-MM-DD")

        limit = None
        if max_items is not None:
            try:
                limit = int(max_items)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_items must be an integer") from exc
            limit = max(0, limit)

        filtered = []
        for it in items:
            importance = it.get("importance", 0.0)
            try:
                importance_val = float(importance)
            except (TypeError, ValueError):
                importance_val = 0.0
            if min_importance is not None and importance_val < min_importance:
                continue
            if cutoff is not None:
                item_date = _item_date(it)
                if item_date is not None and item_date < cutoff:
                    continue
            filtered.append(it)

        if limit is not None and len(filtered) > limit:
            def _key(it):
                imp = it.get("importance", 0.0)
                try:
                    imp_val = float(imp)
                except (TypeError, ValueError):
                    imp_val = 0.0
                dt_obj = _item_date(it)
                date_ord = dt_obj.toordinal() if dt_obj is not None else -1
                try:
                    access_val = int(it.get("access", 0) or 0)
                except (TypeError, ValueError):
                    access_val = 0
                return (imp_val, date_ord, access_val, it.get("id", ""))

            filtered = sorted(filtered, key=_key, reverse=True)[:limit]

        removed = len(items) - len(filtered)
        if removed <= 0:
            return 0

        self.layers[layer] = list(filtered)
        self.save()
        return removed

    def snapshot(self):
        """Return a summary of how many memories exist per layer."""

        return {layer: len(items) for layer, items in self.layers.items()}

    def _recency_weight(self, event_date: str, on_date: str, Q: int) -> float:
        """Compute an exponential decay weight favouring recent experiences."""

        d_evt = _parse_iso_date(event_date) or _parse_iso_date("1970-01-01")
        d_on = _parse_iso_date(on_date)
        if d_evt is None or d_on is None:
            return 0.5
        delta = max(0, (d_on - d_evt).days)
        return math.exp(-delta / max(1, Q))

    def retrieve(self, query_text: str, on_date: str, cfg):
        """Retrieve relevant memories from all layers for the given ``query_text``."""

        query_vec = self._embed([query_text])[0]
        needs_save = False

        def score_layer(items, Q, alpha, k):
            """Score ``items`` using similarity, importance and recency."""

            nonlocal needs_save
            if not items or k <= 0:
                return []

            valid_items = []
            vectors = []
            expected_dim = None

            for it in items:
                embedding = it.get("embedding")
                if not embedding:
                    text = it.get("text", "").strip()
                    if text:
                        try:
                            regenerated = self._embed([text])[0]
                        except Exception:
                            regenerated = None
                        if regenerated is not None:
                            try:
                                emb_arr = np.asarray(regenerated, dtype=float)
                            except (TypeError, ValueError):
                                emb_arr = None
                            if emb_arr is not None and emb_arr.ndim == 1 and emb_arr.size > 0:
                                embedding = emb_arr.tolist()
                                it["embedding"] = embedding
                                needs_save = True
                if embedding is None:
                    continue
                try:
                    emb_arr = np.asarray(embedding, dtype=float)
                except (TypeError, ValueError):
                    continue
                if emb_arr.ndim != 1 or emb_arr.size == 0:
                    continue
                if expected_dim is None:
                    expected_dim = emb_arr.size
                elif emb_arr.size != expected_dim:
                    continue
                vectors.append(emb_arr)
                valid_items.append(it)

            if not valid_items:
                return []

            matrix = np.vstack(vectors)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms

            qv = np.array(query_vec, dtype=float)
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            relevance = (matrix @ qv).tolist()

            importances = [it.get("importance", 0.0) for it in valid_items]
            max_imp = max(1e-6, max(importances))
            scored = []
            for it, rel, imp in zip(valid_items, relevance, importances):
                rec = self._recency_weight(it.get("seen_date", "1970-01-01"), on_date, Q)
                gamma = rel + (imp / max_imp) * alpha + rec
                scored.append((gamma, it))

            scored.sort(key=lambda x: (x[0], x[1].get("id", "")), reverse=True)
            return [it for _, it in scored[:k]]

        shallow = score_layer(self.layers["shallow"], cfg.Q_shallow, cfg.alpha_shallow, cfg.k_shallow)
        intermediate = score_layer(
            self.layers["intermediate"], cfg.Q_intermediate, cfg.alpha_intermediate, cfg.k_intermediate
        )
        deep = score_layer(self.layers["deep"], cfg.Q_deep, cfg.alpha_deep, cfg.k_deep)

        access_dirty = False
        for item in shallow + intermediate + deep:
            prev_raw = item.get("access", 0)
            try:
                prev_access = int(prev_raw)
            except (TypeError, ValueError):
                prev_access = 0
            new_access = prev_access + 1
            if prev_raw != new_access:
                access_dirty = True
            item["access"] = new_access

        if access_dirty:
            needs_save = True

        if needs_save:
            self.save()

        return shallow, intermediate, deep
