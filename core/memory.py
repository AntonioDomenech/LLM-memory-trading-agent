
import json, os, math, hashlib, numpy as np
from .logger import get_logger
from .embeddings import embed_texts
log = get_logger()


def _deterministic_id(text: str) -> str:
    """Return a stable identifier for the provided text."""
    normalized = (text or "").strip().encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()
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

    def snapshot(self):
        """Return a summary of how many memories exist per layer."""

        return {layer: len(items) for layer, items in self.layers.items()}

    def _recency_weight(self, event_date: str, on_date: str, Q: int) -> float:
        """Compute an exponential decay weight favouring recent experiences."""

        import datetime as _dt

        try:
            d_evt = _dt.datetime.strptime(event_date or "1970-01-01", "%Y-%m-%d").date()
            d_on = _dt.datetime.strptime(on_date, "%Y-%m-%d").date()
            delta = max(0, (d_on - d_evt).days)
        except Exception:
            return 0.5
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

        if needs_save:
            self.save()

        for item in shallow + intermediate + deep:
            item["access"] = int(item.get("access", 0)) + 1

        return shallow, intermediate, deep
