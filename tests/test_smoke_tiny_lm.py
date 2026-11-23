"""End-to-end smoke test with a tiny causal LM (downloads ~2 MB from the Hugging Face Hub)."""
import os

import pytest
import torch
from torch.utils.data import DataLoader

from brains.data import BRAINSDataset
from brains.encoder import CaseEncoder
from brains.model import BRAINSModel, apply_lora, load_tokenizer_and_lm
from brains.retrieval import CaseMemory
from brains.schema import by_split, parse_outcomes
from brains.synthetic import generate

TINY = "sshleifer/tiny-gpt2"


@pytest.mark.skipif(os.environ.get("HF_HUB_OFFLINE") == "1", reason="needs network to fetch the tiny model")
def test_train_step_save_load_generate(tmp_path):
    recs = generate(40, seed=0)
    train_recs = by_split(recs)["train"]
    enc = CaseEncoder("hash")
    mem = CaseMemory(enc).build(train_recs)
    mem.save(tmp_path)

    tok, lm = load_tokenizer_and_lm(TINY)
    lm = apply_lora(lm, r=2, alpha=4, dropout=0.0, target_modules=["c_attn"])
    model = BRAINSModel(lm, tok, enc.dim)
    assert model.raghere_id != tok.unk_token_id

    ds = BRAINSDataset(train_recs, mem, tok, top_k=5, top_n=20, max_len=256, train=True, max_mask=4)
    batch = ds.collate([ds[0], ds[1]])
    assert (batch["input_ids"] == model.raghere_id).sum().item() == 2  # one slot per row
    assert (batch["labels"] != -100).any()

    out = model(**{k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.fusion.parameters())

    model.save(tmp_path)
    loaded = BRAINSModel.load(tmp_path, TINY)
    eval_ds = BRAINSDataset(train_recs[:3], mem, tok, top_k=5, top_n=20, max_len=256, train=False)
    eb = eval_ds.collate([eval_ds[i] for i in range(3)])
    texts = loaded.generate_text(**{k: v for k, v in eb.items() if isinstance(v, torch.Tensor) and k != "labels"},
                                 max_new_tokens=4)
    assert len(texts) == 3 and all(isinstance(t, str) for t in texts)
    parse_outcomes(texts[0])  # must not raise
