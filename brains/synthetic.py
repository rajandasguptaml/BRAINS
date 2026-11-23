"""Synthetic cohort generator.

The clinical dataset used in the paper (1,105 institutional records) cannot be redistributed. This generator produces
records with the same schema and with feature ranges that follow published normal and impaired values, so that every
script in the repository runs end to end. Synthetic results are not comparable with the paper.

Usage:
    python -m brains.synthetic --n 1200 --out data/synthetic/cohort.jsonl --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np

from .schema import LABELS, SUBTYPES, PatientRecord, save_jsonl

# per-stage feature profiles: (mmse range, cdr, nwbv mean, age mean)
_STAGE = {
    "No Cognitive Impairment": ((27, 30), 0.0, 0.78, 70),
    "Very Mild Alzheimer's Disease": ((24, 27), 0.5, 0.74, 74),
    "Mild Alzheimer's Disease": ((20, 24), 1.0, 0.70, 77),
    "Moderate Alzheimer's Disease": ((13, 20), 2.0, 0.66, 80),
    "Severe Alzheimer's Disease": ((5, 13), 3.0, 0.62, 83),
}


def _sample_labels(rng: np.random.Generator) -> list[str]:
    k = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    if k == 1:
        return [rng.choice(LABELS)]
    # multi-label cases: adjacent stages, mirroring co-occurring or borderline presentations
    start = rng.integers(0, len(LABELS) - k + 1)
    return LABELS[start : start + k]


def generate(n: int, seed: int = 0, splits=(0.8, 0.1, 0.1)) -> list[PatientRecord]:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        labels = _sample_labels(rng)
        severe = labels[-1]  # most severe label drives the measurements
        (lo, hi), cdr, nwbv_mu, age_mu = _STAGE[severe]
        gender = rng.choice(["F", "M"])
        etiv = rng.normal(1430 if gender == "F" else 1540, 110)
        rec = PatientRecord(
            id=f"S{i:05d}",
            age=int(np.clip(rng.normal(age_mu, 7), 55, 96)),
            gender=str(gender),
            handedness=str(rng.choice(["R", "L"], p=[0.9, 0.1])),
            education_years=int(np.clip(rng.normal(13, 3), 6, 20)),
            ses=int(rng.integers(1, 6)),
            mmse=float(np.clip(rng.uniform(lo, hi), 0, 30).round()),
            cdr=float(cdr if rng.random() > 0.15 else max(0.0, cdr - 0.5)),
            etiv=float(np.clip(etiv, 1150, 1850).round()),
            nwbv=float(np.clip(rng.normal(nwbv_mu, 0.025), 0.55, 0.85).round(3)),
            diagnoses=list(labels),
            subtype=None if severe == LABELS[0] else str(rng.choice(SUBTYPES)),
        )
        records.append(rec)
    # assign splits
    idx = rng.permutation(n)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)
    for j, k in enumerate(idx):
        records[k].split = "train" if j < n_train else ("val" if j < n_train + n_val else "test")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="data/synthetic/cohort.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    recs = generate(args.n, args.seed)
    save_jsonl(recs, args.out)
    from collections import Counter

    types = Counter(r.case_type for r in recs)
    splits = Counter(r.split for r in recs)
    print(f"wrote {len(recs)} records to {args.out}; case types {dict(types)}; splits {dict(splits)}")


if __name__ == "__main__":
    main()
