"""Patient record schema, label set and text serialisation.

The prompt in Fig. 1 of the paper asks the model to select one or more diagnostic outcomes from five options
(no cognitive impairment through severe Alzheimer's disease). Those five options are the prediction targets here.
Section II-B of the paper also describes five Alzheimer's subtypes (early-onset, late-onset, familial, sporadic,
atypical); the subtype is kept as an optional field of the record.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# (roman numeral, label) in the order used by the prompt
OPTIONS: list[tuple[str, str]] = [
    ("i", "No Cognitive Impairment"),
    ("ii", "Very Mild Alzheimer's Disease"),
    ("iii", "Mild Alzheimer's Disease"),
    ("iv", "Moderate Alzheimer's Disease"),
    ("v", "Severe Alzheimer's Disease"),
]
LABELS: list[str] = [label for _, label in OPTIONS]
NUMERAL_OF: dict[str, str] = {label: num for num, label in OPTIONS}
LABEL_OF: dict[str, str] = {num: label for num, label in OPTIONS}

SUBTYPES: list[str] = [
    "Early-Onset Alzheimer's Disease",
    "Late-Onset Alzheimer's Disease",
    "Familial Alzheimer's Disease",
    "Sporadic Alzheimer's Disease",
    "Atypical Alzheimer's Disease",
]

# typical clinical ranges shown in parentheses in the prompt (Fig. 1)
NORMAL_RANGES = {
    "mmse": "Normal: 24-30",
    "cdr": "Normal: 0.0",
    "etiv": "Normal: 1350-1600",
    "nwbv": "Normal: >0.75",
}


@dataclass
class PatientRecord:
    id: str
    age: int
    gender: str  # "F" or "M"
    handedness: str  # "R" or "L"
    education_years: int
    ses: int  # socioeconomic status, 1 (highest) to 5
    mmse: float
    cdr: float
    etiv: float  # mL
    nwbv: float
    diagnoses: list[str] = field(default_factory=list)  # one to three labels from LABELS
    subtype: str | None = None
    split: str = "train"

    def __post_init__(self) -> None:
        unknown = [d for d in self.diagnoses if d not in LABELS]
        if unknown:
            raise ValueError(f"unknown diagnosis label(s) {unknown}; allowed: {LABELS}")
        if not 1 <= len(self.diagnoses) <= 3:
            raise ValueError("a record carries one to three diagnoses")

    @property
    def case_type(self) -> str:
        return {1: "single", 2: "double", 3: "triple"}[len(self.diagnoses)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PatientRecord":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def record_to_text(rec: PatientRecord, include_outcome: bool) -> str:
    """Serialise a record the way the cases are written in Fig. 1 of the paper."""
    lines = [
        f"Age: {rec.age} years old",
        f"MMSE: {rec.mmse:g} ({NORMAL_RANGES['mmse']})",
        f"CDR: {rec.cdr:.1f} ({NORMAL_RANGES['cdr']})",
        f"eTIV (mL): {rec.etiv:.0f} ({NORMAL_RANGES['etiv']})",
        f"nWBV: {rec.nwbv:.2f} ({NORMAL_RANGES['nwbv']})",
        f"Education: {rec.education_years} years",
        f"SES: {rec.ses}",
    ]
    if include_outcome:
        lines.append("Diagnostic Outcomes: " + ", ".join(rec.diagnoses))
    return "\n".join(lines)


def format_outcomes(diagnoses: Iterable[str]) -> str:
    """Target string, e.g. 'Diagnostic Outcomes: [i] [iii] [iv]'."""
    nums = sorted((NUMERAL_OF[d] for d in diagnoses), key=lambda n: [o for o, _ in OPTIONS].index(n))
    return "Diagnostic Outcomes: " + " ".join(f"[{n}]" for n in nums)


_OUTCOME_RE = re.compile(r"\[(iii|ii|iv|i|v)\]")


def parse_outcomes(text: str) -> set[str]:
    """Read the predicted label set back from generated text."""
    return {LABEL_OF[m] for m in _OUTCOME_RE.findall(text) if m in LABEL_OF}


def load_jsonl(path: str | Path) -> list[PatientRecord]:
    with open(path, encoding="utf-8") as f:
        return [PatientRecord.from_dict(json.loads(line)) for line in f if line.strip()]


def save_jsonl(records: Iterable[PatientRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def by_split(records: Iterable[PatientRecord]) -> dict[str, list[PatientRecord]]:
    out: dict[str, list[PatientRecord]] = {}
    for r in records:
        out.setdefault(r.split, []).append(r)
    return out


def iter_case_types(records: Iterable[PatientRecord]) -> Iterator[tuple[str, PatientRecord]]:
    for r in records:
        yield r.case_type, r
