"""Evaluation under the protocol of Table II: Correct and F1 over all cases, and precision / recall / F1 broken down
by case type (single, double, triple diagnoses).

Usage:
    python -m brains.evaluate --checkpoint outputs/demo_tiny
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import resolve_device
from .data import BRAINSDataset
from .encoder import CaseEncoder
from .model import BRAINSModel
from .retrieval import CaseMemory
from .schema import by_split, load_jsonl, parse_outcomes


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(golds: list[list[str]], preds: list[set[str]]) -> dict:
    """Micro-averaged label metrics overall and per case type; Correct = exact set match."""
    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0, "correct": 0})
    for g, p in zip(golds, preds):
        g = set(g)
        ctype = {1: "single", 2: "double", 3: "triple"}.get(len(g), "other")
        for key in ("all", ctype):
            a = agg[key]
            a["tp"] += len(g & p)
            a["fp"] += len(p - g)
            a["fn"] += len(g - p)
            a["n"] += 1
            a["correct"] += int(g == p)
    out = {}
    for key, a in agg.items():
        p, r, f = prf(a["tp"], a["fp"], a["fn"])
        out[key] = {"n": a["n"], "correct": a["correct"] / a["n"] if a["n"] else 0.0, "precision": p, "recall": r, "f1": f}
    return out


def format_table(res: dict) -> str:
    cols = ["single", "double", "triple"]
    head = "| Split | n | Correct | F1 | " + " | ".join(f"{c} P | {c} R | {c} F1" for c in cols) + " |"
    sep = "|" + "---|" * (4 + 3 * len(cols))
    a = res["all"]
    cells = [f"{a['n']}", f"{a['correct']:.3f}", f"{a['f1']:.3f}"]
    for c in cols:
        r = res.get(c, {"precision": 0, "recall": 0, "f1": 0})
        cells += [f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}"]
    return "\n".join([head, sep, "| test | " + " | ".join(cells) + " |"])


def evaluate(checkpoint: str | Path, split: str | None = None, batch_size: int | None = None) -> dict:
    ckpt = Path(checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    device = resolve_device(cfg["train"]["device"])
    split = split or cfg["eval"]["split"]
    batch_size = batch_size or cfg["eval"]["batch_size"]

    records = by_split(load_jsonl(cfg["data"]["path"])).get(split, [])
    encoder = CaseEncoder(cfg["encoder"]["name"], cfg["encoder"]["device"])
    memory = CaseMemory.load(ckpt, encoder)
    model = BRAINSModel.load(ckpt, cfg["model"]["lm_name"], device)
    ds = BRAINSDataset(records, memory, model.tokenizer, top_k=cfg["retrieval"]["top_k"], top_n=cfg["retrieval"]["top_n"],
                       max_len=cfg["train"]["max_len"], train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate)

    golds, preds, texts = [], [], []
    for batch in dl:
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"}
        outs = model.generate_text(**tensors, max_new_tokens=cfg["eval"]["max_new_tokens"])
        for g, t in zip(batch["gold"], outs):
            golds.append(g)
            preds.append(parse_outcomes(t))
            texts.append(t)
    res = score(golds, preds)
    res["_examples"] = [{"gold": g, "pred": sorted(p), "text": t} for g, p, t in list(zip(golds, preds, texts))[:5]]
    (ckpt / f"eval_{split}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(format_table(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    evaluate(args.checkpoint, args.split, args.batch_size)


if __name__ == "__main__":
    main()
