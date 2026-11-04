"""
src/data/synthetic_generator.py
───────────────────────────────
Synthetic de-identified clinical note generator.

Acts as a **drop-in placeholder** for the real MIMIC-IV-Note dataset.
Generates realistic-looking clinical sentences with BIO-tagged spans for
every entity type defined in the project config.

Usage:
    generator = SyntheticClinicalDataset(entity_types=[...], n_samples=1000)
    samples   = generator.generate()          # list[ClinicalSample]
    generator.save("data/synthetic/train.jsonl")
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Type alias ────────────────────────────────────────────────────────────────
Span = dict  # {"text": str, "start": int, "end": int, "label": str}


@dataclass
class ClinicalSample:
    """One (de-identified) clinical sentence + its labelled spans."""

    text: str
    spans: list[Span] = field(default_factory=list)
    note_id: str = ""
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Template bank ─────────────────────────────────────────────────────────────
_TEMPLATES: dict[str, list[str]] = {
    "Problem": [
        "Patient presents with {entity}.",
        "History of {entity} noted.",
        "Diagnosed with {entity} in 20XX.",
        "Complaints of {entity} for 3 days.",
        "Known case of {entity}.",
        "Evaluation for {entity}.",
        "Admission for management of {entity}.",
        "Worsening {entity} over the past week.",
        "New onset {entity} reported.",
        "Chronic {entity} on follow-up.",
    ],
    "Treatment": [
        "Patient underwent {entity}.",
        "Scheduled for {entity} next week.",
        "Post {entity} recovery noted.",
        "{entity} performed without complications.",
        "Referred for {entity}.",
        "History of {entity} two years ago.",
    ],
    "Test": [
        "{entity} ordered to rule out infection.",
        "Results of {entity} pending.",
        "{entity} reveals no acute findings.",
        "Repeat {entity} scheduled.",
        "{entity} performed and reviewed.",
        "Abnormal {entity} noted.",
    ],
    "Drug": [
        "Started on {entity}.",
        "{entity} prescribed for pain management.",
        "Continue {entity} as directed.",
        "Switched from previous agent to {entity}.",
        "{entity} held due to renal function.",
        "Allergic to {entity}.",
    ],
    "Dosage": [
        "Metformin {entity} by mouth.",
        "Lisinopril {entity} daily.",
        "Aspirin {entity} as needed.",
        "Levothyroxine {entity} in the morning.",
        "Amoxicillin {entity} three times daily.",
    ],
    "Duration": [
        "Symptoms lasting {entity}.",
        "Treated for {entity} without improvement.",
        "Follow-up in {entity}.",
        "Course of antibiotics for {entity}.",
        "Observed for {entity} in ED.",
    ],
    "Frequency": [
        "Metoprolol {entity}.",
        "Insulin administered {entity}.",
        "Vitals checked {entity}.",
        "Medication taken {entity} before meals.",
        "Blood glucose monitored {entity}.",
    ],
}

_ENTITIES: dict[str, list[str]] = {
    "Problem": [
        "type 2 diabetes mellitus",
        "hypertension",
        "acute kidney injury",
        "congestive heart failure",
        "chronic obstructive pulmonary disease",
        "urinary tract infection",
        "pneumonia",
        "atrial fibrillation",
        "sepsis",
        "deep vein thrombosis",
        "pulmonary embolism",
        "acute myocardial infarction",
        "iron deficiency anemia",
        "major depressive disorder",
        "hyperlipidemia",
        "osteoarthritis",
        "hypothyroidism",
        "gastroesophageal reflux disease",
        "chronic kidney disease stage 3",
        "obstructive sleep apnea",
    ],
    "Treatment": [
        "coronary artery bypass grafting",
        "percutaneous coronary intervention",
        "hip replacement surgery",
        "appendectomy",
        "cholecystectomy",
        "physical therapy",
        "radiation therapy",
        "chemotherapy",
        "hemodialysis",
        "mechanical ventilation",
        "blood transfusion",
        "lumbar puncture",
        "bronchoscopy",
        "colonoscopy",
    ],
    "Test": [
        "complete blood count",
        "comprehensive metabolic panel",
        "chest X-ray",
        "CT scan of the abdomen",
        "MRI of the brain",
        "electrocardiogram",
        "echocardiogram",
        "urinalysis",
        "blood culture",
        "arterial blood gas",
        "thyroid function tests",
        "HbA1c level",
        "prothrombin time",
        "lipid panel",
    ],
    "Drug": [
        "metformin",
        "lisinopril",
        "atorvastatin",
        "metoprolol",
        "aspirin",
        "warfarin",
        "amoxicillin",
        "furosemide",
        "amlodipine",
        "omeprazole",
        "insulin glargine",
        "levothyroxine",
        "prednisone",
        "albuterol",
    ],
    "Dosage": [
        "500 mg",
        "10 mg",
        "40 mg",
        "81 mg",
        "20 mg",
        "5 mg",
        "1000 mg",
        "25 mcg",
        "100 units",
        "0.5 mg",
    ],
    "Duration": [
        "3 days",
        "2 weeks",
        "one month",
        "6 months",
        "two years",
        "several hours",
        "overnight",
        "48 hours",
        "a few minutes",
        "10 days",
    ],
    "Frequency": [
        "twice daily",
        "once a day",
        "every 8 hours",
        "as needed",
        "three times a day",
        "every morning",
        "at bedtime",
        "every 6 hours",
        "weekly",
        "monthly",
    ],
}


class SyntheticClinicalDataset:
    """
    Generates synthetic de-identified clinical text samples.

    Parameters
    ----------
    entity_types : list[str]
        Which entity categories to include.
    n_samples : int
        Total number of sentences to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        entity_types: Optional[list[str]] = None,
        n_samples: int = 2000,
        seed: int = 42,
    ) -> None:
        self.entity_types = entity_types or list(_TEMPLATES.keys())
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Validate
        for et in self.entity_types:
            if et not in _TEMPLATES:
                raise ValueError(
                    f"Unknown entity type '{et}'. "
                    f"Available: {list(_TEMPLATES.keys())}"
                )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> list[ClinicalSample]:
        """Generate `n_samples` labelled clinical sentences."""
        samples: list[ClinicalSample] = []
        for i in range(self.n_samples):
            entity_type = self.rng.choice(self.entity_types)
            sample = self._generate_sample(entity_type, note_id=f"SYN-{i:06d}")
            samples.append(sample)
        return samples

    def save(self, path: str | Path) -> None:
        """Save generated samples as JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = self.generate()
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")
        print(f"[SyntheticClinicalDataset] Saved {len(samples)} samples → {path}")

    @staticmethod
    def load(path: str | Path) -> list[ClinicalSample]:
        """Load previously saved JSONL back into ClinicalSample objects."""
        path = Path(path)
        samples: list[ClinicalSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                samples.append(
                    ClinicalSample(
                        text=d["text"],
                        spans=d.get("spans", []),
                        note_id=d.get("note_id", ""),
                        source=d.get("source", "synthetic"),
                    )
                )
        return samples

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_sample(self, entity_type: str, note_id: str) -> ClinicalSample:
        template = self.rng.choice(_TEMPLATES[entity_type])
        entity_text = self.rng.choice(_ENTITIES[entity_type])

        sentence = template.format(entity=entity_text)

        # Locate the span
        match = re.search(re.escape(entity_text), sentence)
        if match is None:
            # Fallback (should not happen)
            return ClinicalSample(text=sentence, spans=[], note_id=note_id)

        span: Span = {
            "text": entity_text,
            "start": match.start(),
            "end": match.end(),
            "label": entity_type,
        }
        return ClinicalSample(text=sentence, spans=[span], note_id=note_id)


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = SyntheticClinicalDataset(n_samples=5)
    for s in gen.generate():
        print(s.text)
        print("  spans:", s.spans)
