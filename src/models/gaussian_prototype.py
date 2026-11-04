"""
src/models/gaussian_prototype.py
──────────────────────────────────
Gaussian Prototype classifier for few-shot NER.

Each class is modelled as a **multivariate Gaussian** (mean + covariance).
Classification uses Mahalanobis distance or Bhattacharyya distance.

The Gaussian prototype captures intra-class variance, making it more
expressive than a point prototype — particularly useful when support sets
are small and the embedding space is anisotropic.

References
----------
- Mensink et al. (2013). Distance-Based Image Classification: Generalizing to
  New Classes at Near-Zero Cost. TPAMI.
- Murphy (2012). Machine Learning: A Probabilistic Perspective, Ch. 4.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

logger = logging.getLogger(__name__)


class GaussianPrototype:
    """
    Gaussian Prototype (mean + covariance) few-shot classifier.

    Workflow
    --------
    1. ``fit(support_embeddings, support_labels)``   – estimate per-class Gaussians.
    2. ``predict(query_embeddings)``                  – classify by minimum distance.
    3. ``predict_proba(query_embeddings)``            – return soft probabilities.

    Parameters
    ----------
    distance_metric : str
        "mahalanobis" (default) or "bhattacharyya".
    covariance_type : str
        "full" – full covariance matrix per class (requires K > D, use regularisation).
        "diagonal" – diagonal approximation (more robust with few shots).
    regularisation : float
        Ridge term added to the diagonal of each covariance for numerical stability.
        Recommended: 1e-4 to 1e-2.
    temperature : float
        Softmax temperature applied to negative distances.
    """

    def __init__(
        self,
        distance_metric: str = "mahalanobis",
        covariance_type: str = "full",
        regularisation: float = 1e-4,
        temperature: float = 1.0,
        shrinkage: float = 0.0,
    ) -> None:
        if distance_metric not in ("mahalanobis", "bhattacharyya"):
            raise ValueError(f"Unknown distance_metric: {distance_metric!r}")
        if covariance_type not in ("full", "diagonal"):
            raise ValueError(f"Unknown covariance_type: {covariance_type!r}")

        self.distance_metric = distance_metric
        self.covariance_type = covariance_type
        self.regularisation = regularisation
        self.temperature = temperature
        if not 0 <= shrinkage <= 1: raise ValueError("shrinkage must be in [0,1]")
        self.shrinkage = shrinkage

        # Fitted state
        self.classes_: Optional[list[str]] = None
        self.means_: Optional[np.ndarray] = None          # (C, D)
        self.covariances_: Optional[list[np.ndarray]] = None  # list of (D, D) or (D,)
        self.precisions_: Optional[list[np.ndarray]] = None   # list of (D, D)
        self.log_dets_: Optional[np.ndarray] = None       # (C,)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        support_embeddings: np.ndarray,
        support_labels: list[str],
    ) -> "GaussianPrototype":
        """
        Estimate per-class Gaussian parameters.

        Parameters
        ----------
        support_embeddings : np.ndarray, shape (N, D)
        support_labels : list[str], length N
        """
        if support_embeddings.ndim != 2:
            raise ValueError("support_embeddings must be 2-D (N, D).")
        n, d = support_embeddings.shape

        unique_labels = sorted(set(support_labels))
        residuals = np.concatenate([support_embeddings[np.array(support_labels) == label] -
            support_embeddings[np.array(support_labels) == label].mean(0) for label in unique_labels])
        self.pooled_var_ = (residuals**2).sum(0) / max(n-len(unique_labels), 1)
        means, covariances, precisions, log_dets = [], [], [], []

        for label in unique_labels:
            mask = np.array([l == label for l in support_labels])
            class_embs = support_embeddings[mask]   # (K, D)
            k = len(class_embs)

            mu = class_embs.mean(axis=0)            # (D,)
            cov, prec, log_det = self._estimate_gaussian(class_embs, mu, d, k)

            means.append(mu)
            covariances.append(cov)
            precisions.append(prec)
            log_dets.append(log_det)

        self.classes_ = unique_labels
        self.means_ = np.stack(means)               # (C, D)
        self.covariances_ = covariances
        self.precisions_ = precisions
        self.log_dets_ = np.array(log_dets)         # (C,)

        logger.debug(
            f"[GaussianPrototype] Fitted {len(unique_labels)} Gaussians, "
            f"dim={d}, type={self.covariance_type}"
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, query_embeddings: np.ndarray) -> list[str]:
        """Return the predicted class label for each query embedding."""
        self._check_fitted()
        distances = self._compute_distances(query_embeddings)   # (Q, C)
        nearest = np.argmin(distances, axis=1)
        return [self.classes_[i] for i in nearest]

    def predict_proba(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Return class probabilities for each query embedding."""
        self._check_fitted()
        distances = self._compute_distances(query_embeddings)   # (Q, C)
        logits = -distances / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict_with_confidence(
        self, query_embeddings: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """Returns (predicted_labels, confidence_scores)."""
        probs = self.predict_proba(query_embeddings)
        labels = [self.classes_[i] for i in probs.argmax(axis=1)]
        confidence = probs.max(axis=1)
        return labels, confidence

    # ── Distance computation ───────────────────────────────────────────────────

    def _compute_distances(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute distance from each query to each class Gaussian.

        Returns
        -------
        np.ndarray, shape (Q, C)
        """
        q = len(query_embeddings)
        c = len(self.classes_)
        dist_matrix = np.zeros((q, c))

        for c_idx in range(c):
            mu = self.means_[c_idx]                    # (D,)
            delta = query_embeddings - mu              # (Q, D)

            if self.distance_metric == "mahalanobis":
                dist_matrix[:, c_idx] = self._mahalanobis_batch(
                    delta, self.precisions_[c_idx]
                )
            else:  # bhattacharyya
                dist_matrix[:, c_idx] = self._bhattacharyya_batch(
                    query_embeddings, c_idx
                )

        return dist_matrix

    def _mahalanobis_batch(
        self, delta: np.ndarray, precision: np.ndarray
    ) -> np.ndarray:
        """
        Compute squared Mahalanobis distance for a batch of delta vectors.

        Mahalanobis²(x, μ, Σ) = (x - μ)ᵀ Σ⁻¹ (x - μ)
        """
        if self.covariance_type == "diagonal":
            # precision is a 1-D vector (D,) of 1/σ²
            return np.sum(delta ** 2 * precision, axis=1)   # (Q,)
        else:
            # precision is (D, D) — use (delta @ Prec) * delta → sum
            return np.einsum("qd,de,qe->q", delta, precision, delta)  # (Q,)

    def _bhattacharyya_batch(
        self, queries: np.ndarray, c_idx: int
    ) -> np.ndarray:
        """
        Approximate Bhattacharyya distance from queries to class c_idx.

        Bhat(p||q) ≈ ⅛ (μ₁-μ₂)ᵀ Σ̄⁻¹ (μ₁-μ₂) + ½ ln(|Σ̄| / √(|Σ₁||Σ₂|))
        where Σ̄ = (Σ₁ + Σ₂) / 2.

        For query → class distance, we treat each query point as a
        Gaussian with zero variance (delta function) and compute an
        approximation that reduces to Mahalanobis in that limit.
        """
        # Fall back to Mahalanobis for queries (point mass approximation)
        mu = self.means_[c_idx]
        delta = queries - mu
        return self._mahalanobis_batch(delta, self.precisions_[c_idx])

    # ── Covariance estimation ──────────────────────────────────────────────────

    def _estimate_gaussian(
        self,
        class_embs: np.ndarray,
        mu: np.ndarray,
        d: int,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate covariance, precision, and log-determinant for one class.

        Returns
        -------
        cov : np.ndarray   – covariance matrix (D, D) or diagonal (D,)
        prec : np.ndarray  – precision matrix (D, D) or diagonal (D,)
        log_det : float    – log-determinant of covariance
        """
        eps = self.regularisation

        if self.covariance_type == "diagonal":
            # Diagonal covariance: variance per dimension
            if k > 1:
                var = class_embs.var(axis=0, ddof=1)  # (D,)
            else:
                var = np.ones(d)
            if self.shrinkage:
                var = (1-self.shrinkage)*var + self.shrinkage*self.pooled_var_
            var = var + eps
            prec = 1.0 / var                  # (D,)
            log_det = np.sum(np.log(var))
            return var, prec, log_det

        else:  # "full"
            if k > 1:
                cov = np.cov(class_embs.T, ddof=1)    # (D, D)
            else:
                cov = np.eye(d)
            cov += eps * np.eye(d)                     # Ridge regularisation

            # Cholesky-based inversion for numerical stability
            try:
                c_factor, lower = cho_factor(cov, lower=True)
                prec = cho_solve((c_factor, lower), np.eye(d))
                # log|Σ| = 2 * sum(log(diag(L)))
                log_det = 2.0 * np.sum(np.log(np.abs(np.diag(c_factor))))
            except LinAlgError:
                logger.warning(
                    "[GaussianPrototype] Cholesky failed; falling back to "
                    "pseudo-inverse for one class."
                )
                prec = np.linalg.pinv(cov)
                sign, log_det = np.linalg.slogdet(cov)
                if sign <= 0:
                    log_det = 0.0

            return cov, prec, log_det

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_gaussians(self) -> dict[str, dict]:
        """Return {label: {"mean": ..., "cov": ...}} dictionary."""
        self._check_fitted()
        return {
            label: {
                "mean": self.means_[i],
                "cov": self.covariances_[i],
            }
            for i, label in enumerate(self.classes_)
        }

    def _check_fitted(self) -> None:
        if self.means_ is None:
            raise RuntimeError("GaussianPrototype is not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        n_classes = len(self.classes_) if self.classes_ else "?"
        return (
            f"GaussianPrototype(metric={self.distance_metric!r}, "
            f"cov={self.covariance_type!r}, reg={self.regularisation}, "
            f"classes={n_classes})"
        )
