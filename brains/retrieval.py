"""Case Retrieval Module: FAISS case memory with optional cross-encoder reranking.

Following Sec. II-D of the paper: every training case is encoded and stored in a FAISS vector store; at inference the
input case is encoded, the top-N most similar cases are retrieved by cosine similarity, reranked (the paper uses
bge-reranker-large) and the top-K are kept as the support set R.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .encoder import CaseEncoder
from .schema import PatientRecord, load_jsonl, record_to_text, save_jsonl


@dataclass
class RetrievedCase:
    record: PatientRecord
    score: float
    index: int  # position in the memory, used to fetch the stored vector


class CaseMemory:
    def __init__(self, encoder: CaseEncoder, reranker_name: str | None = None):
        self.encoder = encoder
        self.reranker_name = reranker_name
        self._reranker = None
        self.records: list[PatientRecord] = []
        self.embeddings: np.ndarray | None = None
        self._index = None

    # ----------------------------------------------------------------------------------------- build / persist
    def build(self, records: Iterable[PatientRecord]) -> "CaseMemory":
        import faiss

        self.records = list(records)
        texts = [record_to_text(r, include_outcome=True) for r in self.records]
        self.embeddings = self.encoder.encode(texts)
        self._index = faiss.IndexFlatIP(self.embeddings.shape[1])  # inner product on unit vectors = cosine
        self._index.add(self.embeddings)
        return self

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_jsonl(self.records, folder / "memory_records.jsonl")
        np.save(folder / "memory_embeddings.npy", self.embeddings)
        (folder / "memory_meta.json").write_text(
            json.dumps({"encoder": self.encoder.name, "reranker": self.reranker_name, "n": len(self.records)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: str | Path, encoder: CaseEncoder) -> "CaseMemory":
        import faiss

        folder = Path(folder)
        meta = json.loads((folder / "memory_meta.json").read_text(encoding="utf-8"))
        mem = cls(encoder, meta.get("reranker"))
        mem.records = load_jsonl(folder / "memory_records.jsonl")
        mem.embeddings = np.load(folder / "memory_embeddings.npy")
        mem._index = faiss.IndexFlatIP(mem.embeddings.shape[1])
        mem._index.add(mem.embeddings)
        return mem

    # ----------------------------------------------------------------------------------------- retrieval
    def __len__(self) -> int:
        return len(self.records)

    def _rerank(self, query_text: str, cands: list[RetrievedCase]) -> list[RetrievedCase]:
        if not self.reranker_name:
            return cands
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_name)
        pairs = [(query_text, record_to_text(c.record, include_outcome=True)) for c in cands]
        scores = self._reranker.predict(pairs)
        for c, s in zip(cands, scores):
            c.score = float(s)
        return sorted(cands, key=lambda c: c.score, reverse=True)

    def retrieve(
        self, query_text: str, top_k: int = 5, top_n: int = 100, exclude_ids: Iterable[str] = ()
    ) -> list[RetrievedCase]:
        """Top-N by cosine similarity, optional reranking, top-K returned (Sec. II-D)."""
        if self._index is None or len(self.records) == 0:
            return []
        q = self.encoder.encode([query_text])
        n = min(max(top_n, top_k), len(self.records))
        scores, idx = self._index.search(q, n)
        excluded = set(exclude_ids)
        cands = [
            RetrievedCase(self.records[i], float(s), int(i))
            for s, i in zip(scores[0], idx[0])
            if i >= 0 and self.records[i].id not in excluded
        ]
        cands = self._rerank(query_text, cands)
        return cands[:top_k]

    def vectors(self, cases: list[RetrievedCase]) -> np.ndarray:
        """Stored embeddings of retrieved cases, shape (K, dim)."""
        if not cases:
            return np.zeros((0, self.embeddings.shape[1]), dtype=np.float32)
        return self.embeddings[[c.index for c in cases]]
