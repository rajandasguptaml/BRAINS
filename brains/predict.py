"""Single-case inference with the retrieved evidence, the way a clinician would inspect it.

Usage:
    python -m brains.predict --checkpoint outputs/demo_tiny --age 76 --mmse 22 --cdr 1.0 --etiv 1450 --nwbv 0.68 --education 12 --ses 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import resolve_device
from .encoder import CaseEncoder
from .model import BRAINSModel
from .prompt import build_prompt
from .retrieval import CaseMemory
from .schema import LABELS, PatientRecord, parse_outcomes, record_to_text


def predict(checkpoint: str | Path, rec: PatientRecord) -> dict:
    ckpt = Path(checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    device = resolve_device(cfg["train"]["device"])
    encoder = CaseEncoder(cfg["encoder"]["name"], cfg["encoder"]["device"])
    memory = CaseMemory.load(ckpt, encoder)
    model = BRAINSModel.load(ckpt, cfg["model"]["lm_name"], device)
    tok = model.tokenizer

    query = record_to_text(rec, include_outcome=False)
    cases = memory.retrieve(query, top_k=cfg["retrieval"]["top_k"], top_n=cfg["retrieval"]["top_n"])
    k = cfg["retrieval"]["top_k"]
    vecs = np.zeros((k, memory.embeddings.shape[1]), dtype=np.float32)
    mask = np.zeros(k, dtype=bool)
    if cases:
        v = memory.vectors(cases)
        vecs[: len(v)] = v
        mask[: len(v)] = True

    ids = tok(build_prompt(query), add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None and tok.bos_token_id != tok.eos_token_id:
        ids = [tok.bos_token_id] + ids
    input_ids = torch.tensor([ids], device=device)
    text = model.generate_text(
        input_ids, torch.ones_like(input_ids),
        torch.from_numpy(encoder.encode([query])).to(device),
        torch.from_numpy(vecs)[None].to(device), torch.from_numpy(mask)[None].to(device),
        max_new_tokens=cfg["eval"]["max_new_tokens"],
    )[0]
    return {
        "input": query,
        "raw_output": text,
        "diagnoses": sorted(parse_outcomes(text), key=LABELS.index),
        "evidence": [{"score": round(c.score, 4), "case": record_to_text(c.record, include_outcome=True)} for c in cases],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--gender", default="F")
    ap.add_argument("--handedness", default="R")
    ap.add_argument("--education", type=int, required=True)
    ap.add_argument("--ses", type=int, required=True)
    ap.add_argument("--mmse", type=float, required=True)
    ap.add_argument("--cdr", type=float, required=True)
    ap.add_argument("--etiv", type=float, required=True)
    ap.add_argument("--nwbv", type=float, required=True)
    a = ap.parse_args()
    rec = PatientRecord("query", a.age, a.gender, a.handedness, a.education, a.ses, a.mmse, a.cdr, a.etiv, a.nwbv,
                        diagnoses=[LABELS[0]])  # placeholder label; not used for the query
    res = predict(a.checkpoint, rec)
    print("Patient:\n" + res["input"])
    print("\nModel output:", res["raw_output"].strip())
    print("Diagnoses:", "; ".join(res["diagnoses"]) or "(none parsed)")
    print("\nRetrieved evidence:")
    for i, e in enumerate(res["evidence"], 1):
        print(f"  [{i}] similarity {e['score']:.3f}\n      " + e["case"].replace("\n", "\n      "))


if __name__ == "__main__":
    main()
