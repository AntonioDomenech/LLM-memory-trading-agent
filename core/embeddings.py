
import hashlib
import os
from typing import Iterable, List, Sequence

import numpy as np


def _hash_embed(text: str, dim: int = 512) -> List[float]:
    """Return a deterministic pseudo-embedding for ``text`` using SHA256.

    The helper is used as an offline fallback whenever the OpenAI API key is
    missing or a real embedding request fails. The generated vectors are
    stable across runs so cached memories remain comparable.

    Args:
        text: Arbitrary string to embed.
        dim: Target dimensionality of the pseudo-embedding.

    Returns:
        A list of ``dim`` floating point values in the ``[0, 1)`` interval.
    """

    h = hashlib.sha256(text.encode("utf-8")).digest()
    repeated = (h * ((dim * 4) // len(h) + 1))[: dim * 4]
    v = np.frombuffer(repeated, dtype=np.uint32)
    vec = (v % 997) / 997.0
    return vec.tolist()


def embed_texts(texts: Sequence[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Embed ``texts`` with OpenAI and gracefully fall back to ``_hash_embed``.

    Args:
        texts: Iterable of strings to embed.
        model: Embedding model identifier expected by the OpenAI client.

    Returns:
        A list of embedding vectors, one per input text.
    """

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return [_hash_embed(t, dim=1536 if "small" in model else 3072) for t in texts]
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]
    except Exception:
        return [_hash_embed(t, dim=1536 if "small" in model else 3072) for t in texts]


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""

    arr_a = np.array(list(a), dtype=float)
    arr_b = np.array(list(b), dtype=float)
    na = np.linalg.norm(arr_a)
    nb = np.linalg.norm(arr_b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (na * nb))
