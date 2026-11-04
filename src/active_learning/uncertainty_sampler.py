"""
src/active_learning/uncertainty_sampler.py
───────────────────────────────────────────
Active learning loop for few-shot NER.

Motivation
----------
Instead of randomly picking K support examples, we can use the model's
own uncertainty to pick the most informative ones — saving annotation effort.

Strategies implemented
----------------------
1. **Least Confidence**   – pick examples where max(P(y|x)) is lowest.
2. **Margin Sampling**    – pick examples where P(y₁) − P(y₂) is smallest
                             (most confused between top-2 classes).
3. **Entropy Sampling**   – pick examples with highest predictive entropy
                             H(y|x) = −Σ p log p.
4. **Core-Set (greedy)**  – pick examples that maximise minimum distance
                             to already-selected support (diversity).

Usage
-----
    sampler = UncertaintySampler(strategy="entropy")
    indices = sampler.select(pool_embeddings, existing_support_embeddings, k=5)

    loop = ActiveLearningLoop(encoder, point_clf, span_dataset)
    final_support = loop.run(entity_types, budget=20, strategy="margin")
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Uncertainty Sampler ────────────────────────────────────────────────────────

class UncertaintySampler:
    """
    Selects the K most informative unlabelled examples from a pool.

    Parameters
    ----------
    strategy : str
        One of "least_confidence" | "margin" | "entropy" | "coreset"
    """

    STRATEGIES = ("least_confidence", "margin", "entropy", "coreset")

    def __init__(self, strategy: str = "entropy") -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose from {self.STRATEGIES}")
        self.strategy = strategy

    # ── Public API ────────────────────────────────────────────────────────────

    def select(
        self,
        pool_embeddings: np.ndarray,
        pool_probs: Optional[np.ndarray] = None,
        support_embeddings: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> np.ndarray:
        """
        Select K indices from the pool.

        Parameters
        ----------
        pool_embeddings : np.ndarray, shape (N, D)
            Embeddings of unlabelled pool examples.
        pool_probs : np.ndarray, shape (N, C) | None
            Predicted class probabilities (required for all except coreset).
        support_embeddings : np.ndarray, shape (S, D) | None
            Already-selected support embeddings (required for coreset).
        k : int
            Number of examples to select.

        Returns
        -------
        np.ndarray of shape (k,) — indices into pool.
        """
        n = len(pool_embeddings)
        k = min(k, n)

        if self.strategy == "coreset":
            return self._coreset(pool_embeddings, support_embeddings, k)

        if pool_probs is None:
            raise ValueError(f"pool_probs is required for strategy={self.strategy!r}")

        scores = self._score(pool_probs)
        # Higher score = more uncertain = more informative
        return np.argsort(scores)[-k:]

    def score(self, pool_probs: np.ndarray) -> np.ndarray:
        """Return uncertainty scores (higher = more uncertain) for each pool item."""
        return self._score(pool_probs)

    # ── Strategies ────────────────────────────────────────────────────────────

    def _score(self, probs: np.ndarray) -> np.ndarray:
        if self.strategy == "least_confidence":
            return 1.0 - probs.max(axis=1)

        elif self.strategy == "margin":
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            return 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])

        elif self.strategy == "entropy":
            clipped = np.clip(probs, 1e-9, 1.0)
            return -np.sum(clipped * np.log(clipped), axis=1)

        else:
            raise ValueError(f"Unexpected strategy: {self.strategy!r}")

    def _coreset(
        self,
        pool: np.ndarray,
        support: Optional[np.ndarray],
        k: int,
    ) -> np.ndarray:
        """
        Greedy K-Center Core-Set selection.

        Iteratively picks the pool example with the largest
        minimum distance to the current support set.
        """
        selected_indices: list[int] = []
        n = len(pool)

        if support is not None and len(support) > 0:
            # Initialise distances from pool to existing support
            dists = self._min_dist_to_set(pool, support)
        else:
            # Pick a random starting point
            first = int(np.random.randint(n))
            selected_indices.append(first)
            dists = self._min_dist_to_set(pool, pool[first:first+1])

        for _ in range(k - len(selected_indices)):
            next_idx = int(np.argmax(dists))
            selected_indices.append(next_idx)
            # Update minimum distances with the newly added point
            new_dists = self._min_dist_to_set(pool, pool[next_idx:next_idx+1])
            dists = np.minimum(dists, new_dists)
            dists[selected_indices] = -1.0  # already selected

        return np.array(selected_indices[:k])

    @staticmethod
    def _min_dist_to_set(pool: np.ndarray, ref_set: np.ndarray) -> np.ndarray:
        """
        For each point in pool, compute minimum L2 distance to ref_set.
        Returns array of shape (N,).
        """
        # (N, S) pairwise L2
        diffs = pool[:, None, :] - ref_set[None, :, :]   # (N, S, D)
        dists = np.sqrt((diffs ** 2).sum(axis=-1))        # (N, S)
        return dists.min(axis=1)                          # (N,)


# ── Active Learning Loop ──────────────────────────────────────────────────────

class ActiveLearningLoop:
    """
    Orchestrates active learning for few-shot support set construction.

    Given a pool of unlabelled span embeddings and a prototype classifier,
    iteratively queries the most informative examples and refits the
    classifier.

    Parameters
    ----------
    encoder : GLiNEREncoder | SBERTEncoder
        Span encoder (must have .encode_span_items()).
    classifier : PointPrototype | GaussianPrototype
        Prototype classifier.
    span_dataset : SpanDataset
        Pool of labelled spans (labels used only for final evaluation,
        not during selection — simulating an annotation oracle).
    strategy : str
        Uncertainty sampling strategy.
    seed : int
    """

    def __init__(
        self,
        encoder,
        classifier,
        span_dataset,
        strategy: str = "entropy",
        seed: int = 42,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.span_dataset = span_dataset
        self.sampler = UncertaintySampler(strategy=strategy)
        self.rng = np.random.RandomState(seed)

    def run(
        self,
        entity_types: list[str],
        budget: int = 20,
        init_k: int = 2,
        step: int = 2,
    ) -> dict[str, list]:
        """
        Run the active learning loop.

        Parameters
        ----------
        entity_types : list[str]
        budget : int
            Total number of support examples to select per entity type.
        init_k : int
            Number of random examples to start with per entity type.
        step : int
            Number of examples to add per AL iteration.

        Returns
        -------
        dict mapping entity_type → list of selected SpanItems
        """
        logger.info(
            f"[ActiveLearningLoop] Running AL | budget={budget} | "
            f"init_k={init_k} | step={step} | strategy={self.sampler.strategy}"
        )

        support: dict[str, list] = {et: [] for et in entity_types}
        pool: dict[str, list] = {
            et: self.span_dataset.get_by_type(et).copy()
            for et in entity_types
        }

        # ── Seed: pick init_k random examples per type ─────────────────────
        for et in entity_types:
            if len(pool[et]) < init_k:
                init_items = pool[et]
            else:
                idxs = self.rng.choice(len(pool[et]), size=init_k, replace=False)
                init_items = [pool[et][i] for i in idxs]
            support[et].extend(init_items)
            for item in init_items:
                pool[et].remove(item)

        n_selected = init_k
        iteration = 0

        while n_selected < budget:
            iteration += 1
            n_to_add = min(step, budget - n_selected)

            # ── Fit classifier on current support ──────────────────────────
            all_support = []
            all_labels = []
            for et, items in support.items():
                all_support.extend(items)
                all_labels.extend([et] * len(items))

            if not all_support:
                break

            support_embs = self.encoder.encode_span_items(all_support)
            self.classifier.fit(support_embs, all_labels)

            # ── Score pool examples ────────────────────────────────────────
            for et in entity_types:
                if not pool[et]:
                    continue
                pool_items = pool[et]
                pool_embs = self.encoder.encode_span_items(pool_items)
                pool_probs = self.classifier.predict_proba(pool_embs)
                scores = self.sampler.score(pool_probs)

                # Select top-uncertain from this entity type
                n_et = max(1, n_to_add // len(entity_types))
                top_idxs = np.argsort(scores)[-n_et:][::-1]

                for idx in top_idxs:
                    item = pool_items[idx]
                    support[et].append(item)
                    pool[et].remove(item)

            n_selected += n_to_add
            logger.debug(f"  AL iter {iteration}: {n_selected}/{budget} selected")

        logger.info(
            f"[ActiveLearningLoop] Done. "
            f"Selected {sum(len(v) for v in support.values())} total spans."
        )
        return support
