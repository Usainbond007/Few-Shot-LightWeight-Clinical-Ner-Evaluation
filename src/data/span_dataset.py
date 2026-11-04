"""
src/data/span_dataset.py
────────────────────────
Converts ClinicalSample lists into:

1. ``SpanDataset``      – flat list of (sentence, span_text, label) items
2. ``FewShotEpisode``   – a single N-way K-shot episode
3. ``EpisodeBuilder``   – builds random episodes from a SpanDataset
4. ``collate_episodes`` – batches episodes for DataLoader usage
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .synthetic_generator import ClinicalSample

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class SpanItem:
    """A single labelled span extracted from a clinical sentence."""

    sentence: str          # Full sentence text
    span_text: str         # Text of the entity span
    span_start: int        # Character offset start
    span_end: int          # Character offset end
    label: str             # Entity type label
    note_id: str = ""
    source: str = "synthetic"


@dataclass
class FewShotEpisode:
    """
    A single few-shot episode.

    Attributes
    ----------
    support : dict[str, list[SpanItem]]
        K support examples per entity class.
    query : list[SpanItem]
        Query spans to be classified.
    entity_types : list[str]
        Ordered list of entity classes in this episode.
    n_shots : int
        K in K-shot.
    """

    support: dict[str, list[SpanItem]]
    query: list[SpanItem]
    entity_types: list[str]
    n_shots: int
    episode_id: int = 0

    @property
    def n_way(self) -> int:
        return len(self.entity_types)


# ── SpanDataset ───────────────────────────────────────────────────────────────


class SpanDataset:
    """
    Flat dataset of SpanItem objects, indexed by entity type.

    Parameters
    ----------
    samples : list[ClinicalSample]
        Raw clinical samples with labelled spans.
    entity_types : list[str] | None
        If given, filter to these entity types only.
    """

    def __init__(
        self,
        samples: list[ClinicalSample],
        entity_types: Optional[list[str]] = None,
    ) -> None:
        self.entity_types = entity_types
        self._index: dict[str, list[SpanItem]] = {}
        self._build_index(samples)

    def _build_index(self, samples: list[ClinicalSample]) -> None:
        for sample in samples:
            for span in sample.spans:
                label = span["label"]
                if self.entity_types and label not in self.entity_types:
                    continue
                item = SpanItem(
                    sentence=sample.text,
                    span_text=span["text"],
                    span_start=span["start"],
                    span_end=span["end"],
                    label=label,
                    note_id=sample.note_id,
                    source=sample.source,
                )
                self._index.setdefault(label, []).append(item)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_by_type(self, entity_type: str) -> list[SpanItem]:
        return self._index.get(entity_type, [])

    def all_items(self) -> list[SpanItem]:
        items: list[SpanItem] = []
        for v in self._index.values():
            items.extend(v)
        return items

    @property
    def available_types(self) -> list[str]:
        return list(self._index.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._index.values())

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self._index.items()}
        return f"SpanDataset(total={len(self)}, counts={counts})"


# ── Episode Builder ───────────────────────────────────────────────────────────


class EpisodeBuilder:
    """
    Builds random N-way K-shot episodes from a SpanDataset.

    Parameters
    ----------
    dataset : SpanDataset
    entity_types : list[str]
        All entity types to cycle through.
    n_query : int
        Number of query spans per entity type per episode.
    seed : int
    """

    def __init__(
        self,
        dataset: SpanDataset,
        entity_types: Optional[list[str]] = None,
        n_query: int = 10,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.entity_types = entity_types or dataset.available_types
        self.n_query = n_query
        self.rng = random.Random(seed)

        # Validate that we have enough data for the largest shot size
        for et in self.entity_types:
            n_available = len(dataset.get_by_type(et))
            if n_available == 0:
                raise ValueError(f"No samples for entity type '{et}'.")

    def build_episode(
        self,
        n_shots: int,
        episode_id: int = 0,
        entity_types: Optional[list[str]] = None,
    ) -> FewShotEpisode:
        """
        Sample one episode.

        Parameters
        ----------
        n_shots : int
            K in K-shot (support set size per class).
        episode_id : int
            Episode index for reproducibility.
        entity_types : list[str] | None
            If given, restrict to this subset of entity types.
        """
        types = entity_types or self.entity_types
        support: dict[str, list[SpanItem]] = {}
        query: list[SpanItem] = []

        for et in types:
            pool = self.dataset.get_by_type(et)
            needed = n_shots + self.n_query

            if len(pool) < needed:
                # With replacement if we must
                selected = self.rng.choices(pool, k=needed)
            else:
                selected = self.rng.sample(pool, k=needed)

            support[et] = selected[:n_shots]
            query.extend(selected[n_shots:n_shots + self.n_query])

        self.rng.shuffle(query)
        return FewShotEpisode(
            support=support,
            query=query,
            entity_types=types,
            n_shots=n_shots,
            episode_id=episode_id,
        )

    def build_episodes(
        self,
        n_shots: int,
        n_episodes: int = 100,
        entity_types: Optional[list[str]] = None,
    ) -> list[FewShotEpisode]:
        """Build `n_episodes` independent episodes."""
        return [
            self.build_episode(n_shots, episode_id=i, entity_types=entity_types)
            for i in range(n_episodes)
        ]


# ── Collate helper ─────────────────────────────────────────────────────────────


def collate_episodes(episodes: list[FewShotEpisode]) -> dict:
    """
    Collate a list of FewShotEpisode objects into dict-of-lists format
    (useful for mini-batch processing).
    """
    return {
        "episodes": episodes,
        "n_shots": [ep.n_shots for ep in episodes],
        "n_way": [ep.n_way for ep in episodes],
        "episode_ids": [ep.episode_id for ep in episodes],
    }
