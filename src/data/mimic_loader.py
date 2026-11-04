"""
src/data/mimic_loader.py
────────────────────────
MIMIC-IV-Note dataset loader.

⚠️  PLACEHOLDER — The MIMIC-IV-Note dataset is large and requires a
PhysioNet data-use agreement. This module provides the loading interface
so that real data can be swapped in without changing downstream code.

When you have access to the data, set:
    data.mimic_notes_dir = "<path to mimic-iv-note/2.2/note/>"
in configs/default.yaml, or pass the path to MIMICNoteLoader directly.

The loader will iterate over discharge.csv.gz and yield ClinicalSample
objects in the same format as the synthetic generator.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Iterator, Optional

from .synthetic_generator import ClinicalSample

# ── Known column names in MIMIC-IV-Note discharge.csv.gz ─────────────────────
_NOTE_ID_COL = "note_id"
_TEXT_COL = "text"


class MIMICNoteLoader:
    """
    Loads de-identified clinical notes from MIMIC-IV-Note.

    Parameters
    ----------
    notes_dir : str | Path
        Directory containing the MIMIC-IV-Note CSVs
        (e.g., ``note/discharge.csv.gz``).
    max_samples : int | None
        Limit the number of notes loaded (useful for debugging).
    split_sentences : bool
        If True, split each note into individual sentences before yielding.
    """

    DISCHARGE_FILE = "discharge.csv.gz"
    RADIOLOGY_FILE = "radiology.csv.gz"

    def __init__(
        self,
        notes_dir: str | Path,
        max_samples: Optional[int] = None,
        split_sentences: bool = True,
    ) -> None:
        self.notes_dir = Path(notes_dir)
        self.max_samples = max_samples
        self.split_sentences = split_sentences

        if not self.notes_dir.exists():
            raise FileNotFoundError(
                f"[MIMICNoteLoader] Notes directory not found: {self.notes_dir}\n"
                "  → Please download MIMIC-IV-Note from PhysioNet and update\n"
                "    data.mimic_notes_dir in configs/default.yaml.\n"
                "  → In the meantime, use SyntheticClinicalDataset as a placeholder."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def iter_samples(self, note_type: str = "discharge") -> Iterator[ClinicalSample]:
        """Yield ClinicalSample objects from the chosen note type."""
        fname = (
            self.DISCHARGE_FILE if note_type == "discharge" else self.RADIOLOGY_FILE
        )
        fpath = self.notes_dir / fname

        if not fpath.exists():
            raise FileNotFoundError(
                f"[MIMICNoteLoader] File not found: {fpath}\n"
                "  → Ensure MIMIC-IV-Note is downloaded and the path is correct."
            )

        count = 0
        opener = gzip.open if fpath.suffix == ".gz" else open
        with opener(fpath, "rt", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                note_id = row.get(_NOTE_ID_COL, f"mimic-{count}")
                text = row.get(_TEXT_COL, "")
                if not text.strip():
                    continue

                if self.split_sentences:
                    for sent in self._split_sentences(text):
                        yield ClinicalSample(
                            text=sent,
                            spans=[],   # Spans populated downstream by GLiNER
                            note_id=note_id,
                            source="mimic-iv-note",
                        )
                        count += 1
                        if self.max_samples and count >= self.max_samples:
                            return
                else:
                    yield ClinicalSample(
                        text=text,
                        spans=[],
                        note_id=note_id,
                        source="mimic-iv-note",
                    )
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        return

    def load_all(self, note_type: str = "discharge") -> list[ClinicalSample]:
        """Load all samples into memory (use with caution for large datasets)."""
        return list(self.iter_samples(note_type))

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naïve sentence splitter suitable for clinical notes."""
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        # Remove very short fragments and excessive whitespace
        return [p.strip() for p in parts if len(p.strip()) > 15]


class MIMICAnonymisedSampler:
    """
    Re-samples the MIMIC loader output to build balanced few-shot episodes.

    NOTE: MIMIC notes are already de-identified per the PhysioNet DUA,
    but no PHI should ever be logged or stored beyond what is required
    for research.
    """

    def __init__(self, samples: list[ClinicalSample]) -> None:
        self.samples = samples

    def sample_by_entity(
        self, entity_type: str, k: int, seed: int = 42
    ) -> list[ClinicalSample]:
        """Return k random samples that contain at least one span of `entity_type`."""
        import random

        rng = random.Random(seed)
        eligible = [
            s for s in self.samples if any(sp["label"] == entity_type for sp in s.spans)
        ]
        if len(eligible) < k:
            raise ValueError(
                f"Only {len(eligible)} samples for entity type '{entity_type}', "
                f"but k={k} requested."
            )
        return rng.sample(eligible, k)
