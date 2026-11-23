# Data card

No patient data is stored in this repository. This page describes the data used in the paper, how it was prepared,
how to request access, and the synthetic stand-in that every script in this repository can run on.

## 1. Alzheimer's disease clinical dataset (fine-tuning and evaluation)

| | |
|---|---|
| Size | 1,105 patient records collected from medical institutions |
| Labels | 1 Early-Onset AD, 2 Late-Onset AD, 3 Familial AD, 4 Sporadic AD, 5 Atypical AD |
| Features | MMSE, CDR, eTIV, nWBV, age, gender, handedness, education (years), socioeconomic status (SES) |
| Preprocessing | normalisation, categorical encoding, outlier removal |
| Case types | single-, double- and triple-diagnosis cases, used for the breakdown in Table II of the paper |
| Access | institutional data under a data use agreement; it cannot be redistributed. Contact the corresponding authors (Md Kishor Morol, Md Jakir Hossen) for collaboration requests. |

Clinical meaning of the features (Table I of the paper):

| Feature | Clinical relevance |
|---|---|
| MMSE | global cognitive function (normal 24-30) |
| CDR | dementia severity rating (0 = none, 0.5 = very mild, 1 = mild, 2 = moderate) |
| eTIV | estimated total intracranial volume, used for volume normalisation (mL) |
| nWBV | normalised whole-brain volume, a brain-atrophy indicator |
| Age | primary risk factor |
| Education | reflects cognitive reserve |
| Gender | influences disease manifestation |
| SES | socioeconomic status, 1 (highest) to 5 |

## 2. Pre-training corpus (foundational knowledge)

Textual reports and summaries derived from Alzheimer's disease literature and neurocognitive evaluation guidelines,
plus structured annotations from clinical databases (NACC, ADNI). Sentences that reference visual elements
("Figure", "see image") were removed because the model does not process images. NACC and ADNI data are available to
researchers through their own application processes:

- NACC: https://naccdata.org/
- ADNI: https://adni.loni.usc.edu/

## 3. JSONL schema used by this repository

Each record is one JSON object per line. The same schema is used for institutional exports (kept in `data/private/`,
which is git-ignored) and for synthetic records.

```json
{
  "id": "S0001",
  "age": 76,
  "gender": "F",
  "handedness": "R",
  "education_years": 12,
  "ses": 3,
  "mmse": 22,
  "cdr": 1.0,
  "etiv": 1450,
  "nwbv": 0.68,
  "diagnoses": ["Mild Alzheimer's Disease"],
  "stage": "mild",
  "split": "train"
}
```

`diagnoses` holds one, two or three labels; its length defines the case type (single, double, triple). The text
serialisation that the encoder and the prompt use is produced by `brains.schema.record_to_text`.

## 4. Synthetic cohort

`python -m brains.synthetic --n 1200 --out data/synthetic/cohort.jsonl --seed 0` writes a synthetic cohort with the
schema above. Feature distributions follow published normal and impaired ranges (MMSE, CDR, eTIV, nWBV) and the
label mix is balanced across the five classes and the three case types. The synthetic data carries no information
about real patients and exists only so that the pipeline can be exercised and tested. Numbers obtained on it are not
comparable with the paper.
