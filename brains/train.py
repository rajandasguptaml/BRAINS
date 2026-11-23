"""LoRA fine-tuning of the diagnostic module with the Case Fusion Layer (Sec. III-B).

Usage:
    python -m brains.train --config configs/demo_tiny.yaml
    python -m brains.train --config configs/demo_tiny.yaml --max-steps 5    # smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_device
from .data import BRAINSDataset
from .encoder import CaseEncoder
from .model import BRAINSModel, apply_lora, load_tokenizer_and_lm
from .retrieval import CaseMemory
from .schema import by_split, load_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: dict) -> Path:
    set_seed(cfg["train"]["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(cfg["data"]["path"])
    splits = by_split(records)
    train_recs = splits.get("train", [])
    print(f"records: {len(records)} | train {len(train_recs)} | val {len(splits.get('val', []))} | test {len(splits.get('test', []))}")

    encoder = CaseEncoder(cfg["encoder"]["name"], cfg["encoder"]["device"])
    memory = CaseMemory(encoder, cfg["retrieval"]["reranker"]).build(train_recs)
    memory.save(out_dir)
    print(f"case memory: {len(memory)} cases, dim {encoder.dim}")

    tok, lm = load_tokenizer_and_lm(cfg["model"]["lm_name"])
    lora = cfg["model"]["lora"]
    lm = apply_lora(lm, lora["r"], lora["alpha"], lora["dropout"], lora["target_modules"])
    model = BRAINSModel(lm, tok, encoder.dim, cfg["model"]["fusion_heads"], cfg["model"]["fusion_dropout"]).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_train:,} of {n_all:,} ({100 * n_train / n_all:.2f}%)")

    ds = BRAINSDataset(
        train_recs, memory, tok,
        top_k=cfg["retrieval"]["top_k"], top_n=cfg["retrieval"]["top_n"],
        max_len=cfg["train"]["max_len"], train=True, max_mask=cfg["train"]["max_mask"], seed=cfg["train"]["seed"],
    )
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=ds.collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=0.0)
    total = cfg["train"]["epochs"] * len(dl)
    if cfg["train"]["max_steps"]:
        total = min(total, int(cfg["train"]["max_steps"]))
    warm = min(cfg["train"]["warmup_steps"], max(total - 1, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(total - warm, 1))
    )

    model.train()
    step, t0, running = 0, time.time(), 0.0
    log = []
    for epoch in range(cfg["train"]["epochs"]):
        for batch in dl:
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**tensors)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running += loss.item()
            if step % cfg["train"]["log_every"] == 0 or step == total:
                avg = running / min(cfg["train"]["log_every"], step)
                print(f"epoch {epoch + 1} step {step}/{total} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
                log.append({"step": step, "loss": avg})
                running = 0.0
            if step >= total:
                break
        if step >= total:
            break

    model.save(out_dir)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"saved to {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (see configs/)")
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out:
        cfg["out_dir"] = args.out
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    train(cfg)


if __name__ == "__main__":
    main()
