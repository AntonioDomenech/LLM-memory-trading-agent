
import json, os, math, numpy as np
from .logger import get_logger
from .embeddings import embed_texts
log = get_logger()
class MemoryBank:
    def __init__(self, path="data/memory_bank.json", emb_model="text-embedding-3-small"):
        self.path = path; self.emb_model = emb_model
        self.layers = {"shallow": [], "intermediate": [], "deep": []}; self.load()
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f: self.layers = json.load(f)
            except Exception: log.warning("Could not read memory_bank.json; starting fresh.")
    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f: json.dump(self.layers, f, indent=2)
    def _embed(self, texts): return embed_texts(texts, model=self.emb_model)
    def add_item(self, layer: str, text: str, meta, base_importance: float = 10.0, seen_date: str = None):
        emb = self._embed([text])[0]
        item = {"id": f"id_{abs(hash(text))}", "text": text.strip(), "meta": meta, "importance": float(base_importance),
                "seen_date": seen_date, "access": 0, "embedding": emb}
        self.layers[layer].append(item)
    def promote(self, n_sh_to_int=5, n_int_to_deep=2):
        for src, dst, n in [("shallow","intermediate",n_sh_to_int),("intermediate","deep",n_int_to_deep)]:
            sorted_src = sorted(self.layers[src], key=lambda x: x.get("importance",0), reverse=True)
            top = sorted_src[:n]; self.layers[src] = sorted_src[n:]; self.layers[dst].extend(top)
        self.save()
    def snapshot(self): return {k: len(v) for k, v in self.layers.items()}
    def _recency_weight(self, event_date: str, on_date: str, Q: int) -> float:
        import datetime as _dt
        try:
            d_evt = _dt.datetime.strptime(event_date or "1970-01-01", "%Y-%m-%d").date()
            d_on = _dt.datetime.strptime(on_date, "%Y-%m-%d").date()
            delta = max(0,(d_on-d_evt).days)
        except Exception: return 0.5
        return math.exp(-delta/max(1,Q))
    def retrieve(self, query_text: str, on_date: str, cfg):
        q = self._embed([query_text])[0]
        needs_save = False
        def score_layer(items, Q, alpha, k):
            nonlocal needs_save
            if not items or k<=0: return []
            valid_items = []
            vectors = []
            expected_dim = None
            for it in items:
                emb = it.get("embedding")
                if not emb:
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
                                emb = emb_arr.tolist()
                                it["embedding"] = emb
                                needs_save = True
                if emb is None:
                    continue
                try:
                    emb_arr = np.asarray(emb, dtype=float)
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
            M = np.vstack(vectors)
            norms = np.linalg.norm(M, axis=1, keepdims=True); norms[norms==0]=1.0; M = M / norms
            qv = np.array(q, dtype=float); qv = qv/(np.linalg.norm(qv)+1e-9)
            rel = (M @ qv).tolist()
            imps = [it.get("importance",0.0) for it in valid_items]; max_imp = max(1e-6, max(imps))
            scored = []
            for it, r, imp in zip(valid_items, rel, imps):
                rec = self._recency_weight(it.get("seen_date","1970-01-01"), on_date, Q)
                gamma = r + (imp/max_imp)*alpha + rec; scored.append((gamma, it))
            scored.sort(key=lambda x: (x[0], x[1].get("id","")), reverse=True)
            return [it for _, it in scored[:k]]
        S = score_layer(self.layers["shallow"], cfg.Q_shallow, cfg.alpha_shallow, cfg.k_shallow)
        I = score_layer(self.layers["intermediate"], cfg.Q_intermediate, cfg.alpha_intermediate, cfg.k_intermediate)
        D = score_layer(self.layers["deep"], cfg.Q_deep, cfg.alpha_deep, cfg.k_deep)
        if needs_save: self.save()
        for it in S+I+D: it["access"] = int(it.get("access",0))+1
        return S,I,D
