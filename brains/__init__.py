"""BRAINS: Biomedical Retrieval-Augmented Intelligence for Neurodegeneration Screening.

Reference implementation of the method described in

    Rajan Das Gupta, Md Kishor Morol, Nafiz Fahad, Md Tanzib Hosain, Sumaya Binte Zilani Choya, Md Jakir Hossen.
    "BRAINS: A Retrieval-Augmented System for Alzheimer's Detection and Monitoring." ICMLA 2025.
    https://doi.org/10.1109/ICMLA66185.2025.00224

Modules
-------
schema      patient record schema, label set and text serialisation
synthetic   synthetic cohort generator (the clinical data cannot be redistributed)
encoder     case encoder (sentence-transformer, or an offline hashing encoder for tests)
retrieval   FAISS case memory with optional cross-encoder reranking
fusion      Case Fusion Layer (cross-attention over retrieved cases)
prompt      clinical prompt with the <RAG><RAGHere></RAG> slot
model       causal LM + fusion layer; the fused vector replaces the <RAGHere> embedding
data        dataset and collator with dynamic masking of retrieved cases
train       LoRA fine-tuning
evaluate    Table II protocol: Correct and F1 overall, P/R/F1 by single/double/triple case type
predict     single-case inference with the retrieved evidence
"""

__version__ = "0.1.0"
