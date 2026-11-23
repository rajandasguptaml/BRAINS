"""BRAINS model: a causal LM whose <RAGHere> embedding is replaced by the Case Fusion Layer output."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .fusion import CaseFusionLayer
from .prompt import RAG_HERE, add_special_tokens


def load_tokenizer_and_lm(lm_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(lm_name)
    added = add_special_tokens(tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    lm = AutoModelForCausalLM.from_pretrained(lm_name)
    if added or lm.get_input_embeddings().num_embeddings < len(tok):
        lm.resize_token_embeddings(len(tok))
    return tok, lm


def apply_lora(lm, r: int = 8, alpha: int = 32, dropout: float = 0.05, target_modules="all-linear"):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, task_type="CAUSAL_LM")
    return get_peft_model(lm, cfg)


class BRAINSModel(nn.Module):
    def __init__(self, lm, tokenizer, d_case: int, fusion_heads="auto", fusion_dropout: float = 0.1):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.raghere_id = tokenizer.convert_tokens_to_ids(RAG_HERE)
        d_model = lm.get_input_embeddings().embedding_dim
        self.fusion = CaseFusionLayer(d_model, d_case, fusion_heads, fusion_dropout)

    # --------------------------------------------------------------------------------------------- embeddings
    def _inputs_embeds(self, input_ids, query_vec, case_vecs, case_mask) -> torch.Tensor:
        emb = self.lm.get_input_embeddings()(input_ids)
        fused = self.fusion(query_vec, case_vecs, case_mask).to(emb.dtype)  # (B, D)
        emb = emb.clone()
        for b in range(input_ids.shape[0]):
            pos = (input_ids[b] == self.raghere_id).nonzero(as_tuple=True)[0]
            if len(pos):
                emb[b, pos[0]] = fused[b]
        return emb

    def forward(self, input_ids, attention_mask, query_vec, case_vecs, case_mask, labels=None, **_):
        emb = self._inputs_embeds(input_ids, query_vec, case_vecs, case_mask)
        return self.lm(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate_text(self, input_ids, attention_mask, query_vec, case_vecs, case_mask, max_new_tokens=24) -> list[str]:
        emb = self._inputs_embeds(input_ids, query_vec, case_vecs, case_mask)
        out = self.lm.generate(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # decoder-only models return only the new tokens when inputs_embeds is given; older versions return all
        if out.shape[1] > input_ids.shape[1]:
            out = out[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    # --------------------------------------------------------------------------------------------- persistence
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(folder / "adapter")  # LoRA adapter (or full weights if not wrapped)
        self.tokenizer.save_pretrained(folder / "tokenizer")
        torch.save(self.fusion.state_dict(), folder / "fusion.pt")
        (folder / "fusion_meta.json").write_text(
            json.dumps({"d_model": self.fusion.d_model, "d_case": self.fusion.d_case,
                        "heads": self.fusion.attn.num_heads}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: str | Path, lm_name: str, device: str = "cpu") -> "BRAINSModel":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        folder = Path(folder)
        tok = AutoTokenizer.from_pretrained(folder / "tokenizer")
        lm = AutoModelForCausalLM.from_pretrained(lm_name)
        lm.resize_token_embeddings(len(tok))
        if (folder / "adapter" / "adapter_config.json").exists():
            lm = PeftModel.from_pretrained(lm, folder / "adapter")
        meta = json.loads((folder / "fusion_meta.json").read_text(encoding="utf-8"))
        model = cls(lm, tok, meta["d_case"], fusion_heads=meta["heads"])
        model.fusion.load_state_dict(torch.load(folder / "fusion.pt", map_location=device))
        return model.to(device).eval()
