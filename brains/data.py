"""Dataset and collator.

For every record the input case is encoded, the top-K similar cases are retrieved from the case memory (the record
itself is excluded), and the prompt is built with the <RAGHere> slot. During training, m in [0, max_mask] of the K
retrieved cases are masked at random in each iteration (Sec. III-B, dynamic masking), and the loss is computed only
on the assistant's response tokens.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .prompt import build_prompt
from .retrieval import CaseMemory
from .schema import PatientRecord, format_outcomes, record_to_text


class BRAINSDataset(Dataset):
    def __init__(
        self,
        records: list[PatientRecord],
        memory: CaseMemory,
        tokenizer,
        top_k: int = 5,
        top_n: int = 100,
        max_len: int = 384,
        train: bool = True,
        max_mask: int = 4,
        seed: int = 0,
    ):
        self.records = records
        self.memory = memory
        self.tok = tokenizer
        self.top_k = top_k
        self.max_len = max_len
        self.train = train
        self.max_mask = max_mask
        self.rng = np.random.default_rng(seed)

        # retrieval is deterministic, so do it once
        self.query_texts = [record_to_text(r, include_outcome=False) for r in records]
        self.query_vecs = memory.encoder.encode(self.query_texts)
        self.retrieved = [
            memory.retrieve(q, top_k=top_k, top_n=top_n, exclude_ids={r.id}) for q, r in zip(self.query_texts, records)
        ]
        self.dim = memory.embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        cases = self.retrieved[i]
        vecs = np.zeros((self.top_k, self.dim), dtype=np.float32)
        mask = np.zeros(self.top_k, dtype=bool)
        if cases:
            v = self.memory.vectors(cases)
            vecs[: len(v)] = v
            mask[: len(v)] = True
        if self.train and mask.sum() > 1:
            m = int(self.rng.integers(0, min(self.max_mask, int(mask.sum()) - 1) + 1))
            if m:
                drop = self.rng.choice(np.flatnonzero(mask), size=m, replace=False)
                mask[drop] = False

        prompt_ids = self.tok(build_prompt(self.query_texts[i]), add_special_tokens=False)["input_ids"]
        target_ids = self.tok(format_outcomes(rec.diagnoses), add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            target_ids = target_ids + [self.tok.eos_token_id]
        if self.tok.bos_token_id is not None and self.tok.bos_token_id != self.tok.eos_token_id:
            prompt_ids = [self.tok.bos_token_id] + prompt_ids
        budget = self.max_len - len(target_ids)
        if len(prompt_ids) > budget:  # keep the end of the prompt, where the patient data and the slot are
            prompt_ids = prompt_ids[-budget:]
        if self.train:
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
        else:
            input_ids = prompt_ids
            labels = [-100] * len(prompt_ids)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "query_vec": torch.from_numpy(self.query_vecs[i]),
            "case_vecs": torch.from_numpy(vecs),
            "case_mask": torch.from_numpy(mask),
            "gold": list(rec.diagnoses),
            "id": rec.id,
        }

    def collate(self, batch: list[dict]) -> dict:
        pad = self.tok.pad_token_id
        L = max(len(b["input_ids"]) for b in batch)
        # left-pad in eval so generation starts right after the prompt for every row; right-pad in training
        left = not self.train
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            n = len(b["input_ids"])
            sl = slice(L - n, L) if left else slice(0, n)
            ids[j, sl] = b["input_ids"]
            lab[j, sl] = b["labels"]
            att[j, sl] = 1
        return {
            "input_ids": ids,
            "attention_mask": att,
            "labels": lab,
            "query_vec": torch.stack([b["query_vec"] for b in batch]),
            "case_vecs": torch.stack([b["case_vecs"] for b in batch]),
            "case_mask": torch.stack([b["case_mask"] for b in batch]),
            "gold": [b["gold"] for b in batch],
            "id": [b["id"] for b in batch],
        }
