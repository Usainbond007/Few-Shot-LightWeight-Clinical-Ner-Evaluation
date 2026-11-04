# Dataset and local input files

## Dataset source used by the saved experiment

The completed experiment is based on the **SNOMED CT Entity Linking Challenge v1.2.1**, distributed through PhysioNet:

- Landing page: https://physionet.org/content/snomed-ct-entity-challenge/1.2.1/
- DOI: https://doi.org/10.13026/az93-vm52
- The 272 discharge notes are derived from MIMIC-IV-Note v2.2 and contain SNOMED CT span annotations.

PhysioNet identifies this as a credentialed-access resource. Access requires the applicable training and a signed Data Use Agreement. MIMIC-derived clinical text must not be republished in a public GitHub repository.

## Should the CSV files be committed?

**No. Do not add `train_notes.csv` or `train_annotations.csv` to a public repository.** Keep them on authorised local/Colab storage and obtain them through PhysioNet under the current terms. `.gitignore` excludes `data/raw/*.csv` as a safety measure.

## Expected local layout

```text
data/raw/
├── train_notes.csv
└── train_annotations.csv
```

The executed loader (`src/data/validated.py`) expects:

### `train_notes.csv`
- `note_id`
- `text`

### `train_annotations.csv`
- `note_id`
- `start`
- `end`
- `span`
- `entity_type`

**Schema distinction:** The PhysioNet source annotation file documents `note_id`, `start`, `end` and `concept_id`. The project loader additionally requires literal mention text (`span`) and a coarse project label (`entity_type`). Derive `span` from note text and offsets, and derive `entity_type` using the documented SNOMED-to-project-label mapping. Do not simply rename `concept_id` to `entity_type`. The original CSVs alone are not sufficient for this loader. The exact `concept_id`-to-`Problem`/`Treatment`/`Test` mapping artefact used for the saved run is not included in this bundle. Recover it from the original preparation workflow before claiming exact reproducibility; do not invent a mapping.

The saved run audit reports 272 notes and 75,490 input annotation rows. It rejected 20 span/text mismatches and 28 duplicate annotation keys, leaving 75,442 accepted annotations.

## Synthetic data

`data/synthetic/clinical_samples.jsonl` contains synthetic examples for demonstrations/tests only. It is not the PhysioNet dataset and cannot reproduce the reported clinical results.
