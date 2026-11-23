# Roadmap

Where this repository is going, and how the work is shared.

## v0.1 — paper release (this version)

- Paper PDF, LaTeX source and BibTeX in `paper/`.
- Project page in `docs/`, served at https://rajandasguptaml.github.io/BRAINS/.
- Data card in `data/README.md`: schema of the 1,105-record clinical dataset, preprocessing, and how to request access.
  The records themselves are not redistributable.
- Reference implementation of the method in `brains/`, reconstructed from the paper: case encoder, FAISS case memory
  with reranking, Case Fusion Layer, `<RAGHere>` injection into the prompt embedding, LoRA fine-tuning with dynamic
  masking of retrieved cases, and the single/double/triple-case evaluation protocol of Table II.
- Synthetic cohort generator so that every script runs end to end on a laptop with any small Hugging Face causal LM.

## v0.2 — original training artefacts

- Training configurations and launch scripts used for the LLaMA2-13B runs in the paper (pre-training: 10 epochs,
  batch 64, lr 1e-4, 1,000 warm-up steps, block 2048; fine-tuning: 15 epochs, batch 4, lr 1e-5, LoRA r=8, alpha=32).
- Preprocessing scripts that turn institutional exports into the JSONL schema in `data/README.md`.
- Exact evaluation split definitions and the scoring script that produced Table II.

## v0.3 — released weights and benchmark

- LoRA adapters for the diagnostic module, if the data use agreement allows releasing weights trained on the records.
- A synthetic-but-realistic benchmark cohort with single, double and triple diagnostic cases, so that other groups can
  compare retrieval strategies under the paper's protocol without access to institutional data.
- Model card and evaluation report.

## v1.0 — beyond Alzheimer's

- Retrieval grounding checks: measure whether an answer depends on the retrieved cases, and abstain when the case
  memory holds nothing similar (the open questions raised in the paper's conclusion).
- Extension of the case memory to other neurodegenerative conditions.
- Hosted demo (Hugging Face Space) running on the synthetic cohort.

## Sharing plan

1. Link this repository from the arXiv listing (comments field), the ACL/IEEE author pages and Google Scholar.
2. Submit to Papers with Code under the ICMLA 2025 entry so the results table is indexed.
3. Pin the repository on the GitHub profile and link it from the publications page of the personal site.
4. Announce once per milestone (v0.1, v0.3, v1.0) on LinkedIn and X with the project page link, not more often.
5. Cross-reference the ELITE Research Lab platform repository (`BRAINS-R-D`) which deploys the method inside a
   multi-agent clinical assistant.
