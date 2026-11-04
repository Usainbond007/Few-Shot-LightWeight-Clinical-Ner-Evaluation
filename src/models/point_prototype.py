"""
src/models/point_prototype.py
──────────────────────────────
Point Prototype classifier for few-shot NER.

Each class prototype is the **mean embedding** of its K support examples.
Classification uses cosine (or Euclidean) distance to the nearest prototype.

Reference: Snell et al. (2017). Prototypical Networks for Few-Shot Learning.
           NeurIPS 2017. arXiv:1703.05175
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class PointPrototype:
    """
    Point Prototype (mean-embedding) few-shot classifier.

    Workflow
    --------
    1. ``fit(support_embeddings, support_labels)``   – compute per-class means.
    2. ``predict(query_embeddings)``                  – classify by nearest prototype.
    3. ``predict_proba(query_embeddings)``            – return soft probabilities via
                                                        distance-to-logit conversion.

    Parameters
    ----------
    distance_metric : str
        "cosine" (default) or "euclidean".
    temperature : float
        Softmax temperature applied to negative distances (lower → sharper).
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        temperature: float = 1.0,
    ) -> None:
        if distance_metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unknown distance_metric: {distance_metric!r}")
        self.distance_metric = distance_metric
        self.temperature = temperature

        # Fitted state
        self.prototypes_: Optional[np.ndarray] = None   # (C, D)
        self.classes_: Optional[list[str]] = None       # length C

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        support_embeddings: np.ndarray,
        support_labels: list[str],
    ) -> "PointPrototype":
        """
        Compute class prototypes from support embeddings.

        Parameters
        ----------
        support_embeddings : np.ndarray, shape (N, D)
        support_labels : list[str], length N
        """
        if support_embeddings.ndim != 2:
            raise ValueError("support_embeddings must be 2-D (N, D).")
        if len(support_labels) != len(support_embeddings):
            raise ValueError("support_embeddings and support_labels must have equal length.")

        unique_labels = sorted(set(support_labels))
        prototypes: list[np.ndarray] = []

        for label in unique_labels:
            mask = np.array([l == label for l in support_labels])
            class_embs = support_embeddings[mask]       # (K, D)
            prototype = class_embs.mean(axis=0)         # (D,)
            prototypes.append(prototype)

        self.classes_ = unique_labels
        self.prototypes_ = np.stack(prototypes)         # (C, D)

        logger.debug(
            f"[PointPrototype] Fitted {len(unique_labels)} prototypes, "
            f"dim={self.prototypes_.shape[1]}"
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, query_embeddings: np.ndarray) -> list[str]:
        """Return the predicted class label for each query embedding."""
        self._check_fitted()
        distances = self._compute_distances(query_embeddings)   # (Q, C)
        nearest = np.argmin(distances, axis=1)                  # (Q,)
        return [self.classes_[i] for i in nearest]

    def predict_proba(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for each query embedding.

        Probabilities are computed by applying a softmax over negative distances.
        """
        self._check_fitted()
        distances = self._compute_distances(query_embeddings)   # (Q, C)
        # Convert distances to logits: lower distance → higher score
        logits = -distances / self.temperature                  # (Q, C)
        # Numerically stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs                                            # (Q, C)

    def predict_with_confidence(
        self, query_embeddings: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """
        Returns (predicted_labels, confidence_scores).

        Confidence is the max probability from ``predict_proba``.
        """
        probs = self.predict_proba(query_embeddings)
        labels = [self.classes_[i] for i in probs.argmax(axis=1)]
        confidence = probs.max(axis=1)
        return labels, confidence

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_prototypes(self) -> dict[str, np.ndarray]:
        """Return {label: prototype_vector} dictionary."""
        self._check_fitted()
        return dict(zip(self.classes_, self.prototypes_))

    def prototype_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise cosine similarities between prototypes.
        Useful for understanding inter-class confusion.
        """
        self._check_fitted()
        norm = self.prototypes_ / (
            np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-8
        )
        return norm @ norm.T     # (C, C)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_distances(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Compute (Q, C) distance matrix from queries to prototypes."""
        return cdist(query_embeddings, self.prototypes_, metric=self.distance_metric)

    def _check_fitted(self) -> None:
        if self.prototypes_ is None:
            raise RuntimeError("PointPrototype is not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        n_classes = len(self.classes_) if self.classes_ else "?"
        return (
            f"PointPrototype(metric={self.distance_metric!r}, "
            f"classes={n_classes}, temp={self.temperature})"
        )
