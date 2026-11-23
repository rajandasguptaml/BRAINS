"""Case encoder: turns the text of a case into one L2-normalised vector.

The paper uses the diagnostic LLM itself as the text encoder (hidden state of the [CLS]-equivalent position). This
reference implementation uses a sentence-transformer by default so that it runs on a laptop; set `encoder.name` in
the config to any sentence-transformers model. `hash` selects an offline hashing encoder used by the tests.
"""
from __future__ import annotations

import re
import zlib

import numpy as np


class HashEncoder:
    """Deterministic bag-of-words hashing encoder. No downloads, no learning; only for tests and smoke runs."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9.>-]+", t.lower())
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = zlib.crc32(f.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class CaseEncoder:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.name = name
        if name == "hash":
            self._impl = HashEncoder()
            self.dim = self._impl.dim
        else:
            from sentence_transformers import SentenceTransformer

            self._impl = SentenceTransformer(name, device=device)
            self.dim = int(self._impl.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if isinstance(self._impl, HashEncoder):
            return self._impl.encode(texts, batch_size)
        emb = self._impl.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32)
