"""Case Fusion Layer (Sec. II-E, Eq. 1).

The encoded input case t acts as the query and the retrieved case vectors A = [w_0, ..., w_{K-1}] act as keys and
values of a cross-attention block: Attn(Q, K, V) = softmax(QK^T / sqrt(d_k)) V, with Q = W_Q t, K = W_K A, V = W_V A.
The result is a single vector in the LLM's embedding space, which replaces the embedding of the <RAGHere> token in
the prompt. Masked positions (dynamic masking during training, or fewer than K retrieved cases) are excluded from
the attention.

The paper fuses the token-level hidden states of each retrieved case; this implementation fuses one pooled vector per
case, which keeps the layer independent of the encoder's sequence length.
"""
from __future__ import annotations

import torch
from torch import nn


def pick_heads(d_model: int, preferred: int = 8) -> int:
    for h in (preferred, 8, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


class CaseFusionLayer(nn.Module):
    def __init__(self, d_model: int, d_case: int, n_heads: int | str = "auto", dropout: float = 0.1):
        super().__init__()
        heads = pick_heads(d_model) if n_heads == "auto" else int(n_heads)
        self.q_proj = nn.Linear(d_case, d_model)
        self.kv_proj = nn.Linear(d_case, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_case = d_case

    def forward(self, query_vec: torch.Tensor, case_vecs: torch.Tensor, case_mask: torch.Tensor) -> torch.Tensor:
        """
        query_vec : (B, d_case)       encoded input case t
        case_vecs : (B, K, d_case)    encoded retrieved cases
        case_mask : (B, K) bool       True where a case is present and not masked
        returns   : (B, d_model)      fused case vector
        """
        if case_mask.dtype != torch.bool:
            case_mask = case_mask.bool()
        # never attend over an empty set: if a row has no valid case, let it attend to a zero vector instead
        empty = ~case_mask.any(dim=1)
        if empty.any():
            case_mask = case_mask.clone()
            case_mask[empty, 0] = True
            case_vecs = case_vecs.clone()
            case_vecs[empty, 0] = 0.0
        q = self.q_proj(query_vec).unsqueeze(1)  # (B, 1, D)
        kv = self.kv_proj(case_vecs)  # (B, K, D)
        attended, _ = self.attn(q, kv, kv, key_padding_mask=~case_mask, need_weights=False)
        h = self.norm1(q + attended)
        h = self.norm2(h + self.ffn(h))
        return h.squeeze(1)
