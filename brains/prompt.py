"""Clinical prompt with the retrieval slot, following Fig. 1 of the paper."""
from __future__ import annotations

from .schema import OPTIONS

RAG_OPEN, RAG_HERE, RAG_CLOSE = "<RAG>", "<RAGHere>", "</RAG>"
SPECIAL_TOKENS = [RAG_OPEN, RAG_HERE, RAG_CLOSE]

_OPTIONS_TEXT = ", ".join(f"({num}) {label}" for num, label in OPTIONS)

SYSTEM_PROMPT = (
    "You are now a professional neurologist specializing in Alzheimer's disease. Based on the patient's "
    "neurocognitive and brain volume data, such as MMSE, CDR, eTIV, nWBV, age, education, and SES, please make a "
    "clinical judgment. The values in parentheses represent the typical clinical range where relevant. Select the "
    f"appropriate diagnosis from the following options (multiple selections allowed): {_OPTIONS_TEXT}."
)


def build_prompt(case_text: str) -> str:
    """Prompt whose <RAGHere> token is replaced by the fused case vector. `case_text` excludes the outcome."""
    return (
        f"System: {SYSTEM_PROMPT}\n"
        f"User: Patient Data:\n{case_text}\n"
        "Below are reference neurocognitive records and diagnostic outcomes for similar patients: "
        f"{RAG_OPEN}{RAG_HERE}{RAG_CLOSE}\n"
        "Assistant: "
    )


def add_special_tokens(tokenizer) -> int:
    """Register the retrieval tokens; returns the number of tokens added (0 if already present)."""
    present = set(tokenizer.get_vocab().keys())
    missing = [t for t in SPECIAL_TOKENS if t not in present]
    if not missing:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": missing})
