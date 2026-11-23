"""YAML config loading with defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "data": {"path": "data/synthetic/cohort.jsonl"},
    "encoder": {"name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
    "retrieval": {"top_k": 5, "top_n": 100, "reranker": None},
    "model": {
        "lm_name": "HuggingFaceTB/SmolLM2-135M",
        "lora": {"r": 8, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "fusion_heads": "auto",
        "fusion_dropout": 0.1,
    },
    "train": {
        "epochs": 1,
        "batch_size": 2,
        "lr": 1.0e-4,
        "warmup_steps": 20,
        "max_len": 384,
        "max_mask": 4,
        "seed": 0,
        "device": "auto",
        "log_every": 20,
        "max_steps": None,
    },
    "eval": {"batch_size": 4, "max_new_tokens": 24, "split": "test"},
    "out_dir": "outputs/demo_tiny",
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
