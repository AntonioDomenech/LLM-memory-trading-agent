
import os, hashlib, numpy as np
def _hash_embed(text: str, dim: int = 512):
    h = hashlib.sha256(text.encode("utf-8")).digest()
    v = np.frombuffer((h* ((dim*4)//len(h)+1))[:dim*4], dtype=np.uint32)
    vec = (v % 997) / 997.0
    return vec.tolist()
def embed_texts(texts, model: str = "text-embedding-3-small"):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return [_hash_embed(t, dim=1536 if "small" in model else 3072) for t in texts]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return [_hash_embed(t, dim=1536 if "small" in model else 3072) for t in texts]
def cosine(a, b):
    import numpy as np
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na*nb))
