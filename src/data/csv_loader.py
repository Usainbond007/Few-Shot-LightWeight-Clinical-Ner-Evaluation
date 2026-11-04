"""
src/data/csv_loader.py
───────────────────────
Loads the real dataset from:
  - train_notes.csv       : note_id, text
  - train_annotations.csv : note_id, start, end, entity_text, entity_type

Both files are expected in data/raw/ by default.
Falls back to the synthetic generator if files are not found.

Column name auto-detection handles common variants
(e.g. "label" vs "entity_type", "span_start" vs "start", etc.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .synthetic_generator import ClinicalSample

logger = logging.getLogger(__name__)

# ── Column name aliases ────────────────────────────────────────────────────────
_NOTE_ID_ALIASES   = ["note_id", "id", "document_id", "doc_id", "hadm_id"]
_TEXT_ALIASES      = ["text", "note_text", "content", "discharge_text"]
_START_ALIASES     = ["start", "span_start", "start_char", "begin", "offset_start"]
_END_ALIASES       = ["end",   "span_end",   "end_char",   "stop",  "offset_end"]
_ENTITY_ALIASES    = ["span", "entity_text", "span_text", "mention", "text", "entity"]
_LABEL_ALIASES     = ["entity_type", "label", "type", "category", "ner_tag"]


def _resolve_col(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str:
    """Return the first alias that matches a column in df."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    if required:
        raise KeyError(
            f"Could not find any of {aliases} in columns {list(df.columns)}. "
            "Please check your CSV files or update the alias list in csv_loader.py."
        )
    return ""


class CSVDataLoader:
    """
    Loads clinical NER data from two CSV files.

    Parameters
    ----------
    notes_path : str | Path
        Path to train_notes.csv  (or similar)
    annotations_path : str | Path
        Path to train_annotations.csv  (or similar)
    entity_types : list[str] | None
        If given, filter annotations to these entity types only.
    max_samples : int | None
        Cap on the number of notes to load.
    """

    def __init__(
        self,
        notes_path: str | Path,
        annotations_path: str | Path,
        entity_types: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.notes_path = Path(notes_path)
        self.annotations_path = Path(annotations_path)
        self.entity_types = entity_types
        self.max_samples = max_samples

        if not self.notes_path.exists():
            raise FileNotFoundError(
                f"[CSVDataLoader] Notes file not found: {self.notes_path}\n"
                "  → Place train_notes.csv in data/raw/ or update the path in\n"
                "    configs/default.yaml under data.notes_csv."
            )
        if not self.annotations_path.exists():
            raise FileNotFoundError(
                f"[CSVDataLoader] Annotations file not found: {self.annotations_path}\n"
                "  → Place train_annotations.csv in data/raw/ or update the path."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> list[ClinicalSample]:
        """
        Load and merge notes + annotations into ClinicalSample objects.

        Returns
        -------
        list[ClinicalSample]
            One sample per note (or per sentence if split_sentences=True in future).
        """
        logger.info(f"[CSVDataLoader] Loading notes from {self.notes_path}")
        notes_df = pd.read_csv(self.notes_path)

        logger.info(f"[CSVDataLoader] Loading annotations from {self.annotations_path}")
        ann_df = pd.read_csv(self.annotations_path)

        # ── Resolve column names ───────────────────────────────────────────────
        note_id_col   = _resolve_col(notes_df, _NOTE_ID_ALIASES)
        text_col      = _resolve_col(notes_df, _TEXT_ALIASES)

        ann_note_col  = _resolve_col(ann_df, _NOTE_ID_ALIASES)
        start_col     = _resolve_col(ann_df, _START_ALIASES)
        end_col       = _resolve_col(ann_df, _END_ALIASES)
        entity_col    = _resolve_col(ann_df, _ENTITY_ALIASES)
        label_col     = _resolve_col(ann_df, _LABEL_ALIASES)

        logger.info(
            f"[CSVDataLoader] Detected columns — "
            f"notes: [{note_id_col}, {text_col}], "
            f"annotations: [{ann_note_col}, {start_col}, {end_col}, "
            f"{entity_col}, {label_col}]"
        )

        # ── Filter entity types if requested ──────────────────────────────────
        if self.entity_types:
            ann_df = ann_df[ann_df[label_col].isin(self.entity_types)]

        # ── Group annotations by note_id ──────────────────────────────────────
        ann_by_note: dict[str, list[dict]] = {}
        for _, row in ann_df.iterrows():
            nid = str(row[ann_note_col])
            span = {
                "text":  str(row[entity_col]),
                "start": int(row[start_col]),
                "end":   int(row[end_col]),
                "label": str(row[label_col]),
            }
            ann_by_note.setdefault(nid, []).append(span)

        # ── Build ClinicalSample objects ──────────────────────────────────────
        samples: list[ClinicalSample] = []
        notes_iter = notes_df.iterrows()

        for _, note_row in notes_iter:
            nid = str(note_row[note_id_col])
            text = str(note_row[text_col])
            spans = ann_by_note.get(nid, [])

            # Validate and clamp span offsets
            clean_spans = []
            for sp in spans:
                if sp["start"] < 0 or sp["end"] > len(text):
                    logger.debug(
                        f"[CSVDataLoader] Clamping span {sp} in note {nid} "
                        f"(text len={len(text)})"
                    )
                    sp["start"] = max(0, sp["start"])
                    sp["end"]   = min(len(text), sp["end"])
                clean_spans.append(sp)

            samples.append(
                ClinicalSample(
                    text=text,
                    spans=clean_spans,
                    note_id=nid,
                    source="csv",
                )
            )

            if self.max_samples and len(samples) >= self.max_samples:
                break

        logger.info(
            f"[CSVDataLoader] Loaded {len(samples)} notes, "
            f"{sum(len(s.spans) for s in samples)} total spans."
        )
        return samples

    def stats(self) -> dict:
        """Return a quick statistics dict (entity type counts, etc.)"""
        samples = self.load()
        from collections import Counter
        label_counts: Counter = Counter()
        for s in samples:
            for sp in s.spans:
                label_counts[sp["label"]] += 1
        return {
            "n_notes": len(samples),
            "n_spans": sum(label_counts.values()),
            "entity_counts": dict(label_counts),
        }


def load_dataset(
    notes_path: str | Path,
    annotations_path: str | Path,
    entity_types: Optional[list[str]] = None,
    fallback_n_samples: int = 3000,
    seed: int = 42,
) -> list[ClinicalSample]:
    """
    Convenience function: load real CSV data or fall back to synthetic.

    Parameters
    ----------
    notes_path, annotations_path : paths to your CSV files
    entity_types : filter list (or None = all)
    fallback_n_samples : how many synthetic samples to generate if CSVs missing
    seed : random seed for synthetic generation

    Returns
    -------
    list[ClinicalSample]
    """
    try:
        loader = CSVDataLoader(
            notes_path=notes_path,
            annotations_path=annotations_path,
            entity_types=entity_types,
        )
        return loader.load()
    except FileNotFoundError as e:
        logger.warning(str(e))
        logger.warning(
            "[load_dataset] Falling back to synthetic data. "
            "Place train_notes.csv and train_annotations.csv in data/raw/ "
            "to use real data."
        )
        from .synthetic_generator import SyntheticClinicalDataset
        gen = SyntheticClinicalDataset(
            entity_types=entity_types,
            n_samples=fallback_n_samples,
            seed=seed,
        )
        return gen.generate()
