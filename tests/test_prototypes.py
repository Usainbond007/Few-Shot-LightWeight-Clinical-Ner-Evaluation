"""
tests/test_prototypes.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from src.models.point_prototype import PointPrototype
from src.models.gaussian_prototype import GaussianPrototype


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_centers(n_classes: int, dim: int) -> list:
    """Fixed well-separated unit-vector centres scaled by 50 * class_index."""
    centers = []
    for c in range(n_classes):
        v = np.zeros(dim)
        v[c % dim] = 50.0 * (c + 1)   # orthogonal + large magnitude
        centers.append(v)
    return centers


def make_support(n_classes=3, k=5, dim=16, seed=42):
    """Generate support embeddings tightly clustered around fixed centres."""
    rng = np.random.RandomState(seed)
    centers = _make_centers(n_classes, dim)
    embeddings, labels = [], []
    for c in range(n_classes):
        embs = centers[c] + rng.randn(k, dim) * 0.1   # very tight cluster
        embeddings.append(embs)
        labels.extend([f"Class{c}"] * k)
    return np.vstack(embeddings), labels


def make_query(n_classes=3, q=10, dim=16, seed=99):
    """Generate query embeddings near the SAME fixed centres as support."""
    rng = np.random.RandomState(seed)
    centers = _make_centers(n_classes, dim)
    embeddings, labels = [], []
    for c in range(n_classes):
        embs = centers[c] + rng.randn(q, dim) * 0.1   # very tight cluster
        embeddings.append(embs)
        labels.extend([f"Class{c}"] * q)
    return np.vstack(embeddings), labels


# ── PointPrototype tests ───────────────────────────────────────────────────────

class TestPointPrototype:
    def test_fit_sets_classes(self):
        clf = PointPrototype()
        embs, labels = make_support()
        clf.fit(embs, labels)
        assert set(clf.classes_) == {"Class0", "Class1", "Class2"}

    def test_prototype_shape(self):
        clf = PointPrototype()
        embs, labels = make_support(dim=32)
        clf.fit(embs, labels)
        assert clf.prototypes_.shape == (3, 32)

    def test_predict_returns_correct_labels(self):
        clf = PointPrototype()
        support_embs, support_labels = make_support()
        clf.fit(support_embs, support_labels)
        query_embs, query_labels = make_query()
        preds = clf.predict(query_embs)
        assert all(p in clf.classes_ for p in preds)
        # High accuracy expected on separable clusters
        acc = sum(p == t for p, t in zip(preds, query_labels)) / len(query_labels)
        assert acc > 0.8

    def test_predict_proba_sums_to_one(self):
        clf = PointPrototype()
        embs, labels = make_support()
        clf.fit(embs, labels)
        q_embs, _ = make_query()
        probs = clf.predict_proba(q_embs)
        assert probs.shape == (len(q_embs), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_unfitted_raises(self):
        clf = PointPrototype()
        with pytest.raises(RuntimeError):
            clf.predict(np.random.randn(5, 16))

    def test_euclidean_metric(self):
        clf = PointPrototype(distance_metric="euclidean")
        embs, labels = make_support()
        clf.fit(embs, labels)
        q_embs, q_labels = make_query()
        preds = clf.predict(q_embs)
        acc = sum(p == t for p, t in zip(preds, q_labels)) / len(q_labels)
        assert acc > 0.8

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            PointPrototype(distance_metric="manhattan")


# ── GaussianPrototype tests ────────────────────────────────────────────────────

class TestGaussianPrototype:
    def test_fit_sets_classes(self):
        clf = GaussianPrototype()
        embs, labels = make_support()
        clf.fit(embs, labels)
        assert set(clf.classes_) == {"Class0", "Class1", "Class2"}

    def test_means_shape(self):
        clf = GaussianPrototype()
        embs, labels = make_support(dim=32)
        clf.fit(embs, labels)
        assert clf.means_.shape == (3, 32)

    def test_predict_high_accuracy(self):
        clf = GaussianPrototype(covariance_type="diagonal")
        support_embs, support_labels = make_support()
        clf.fit(support_embs, support_labels)
        q_embs, q_labels = make_query()
        preds = clf.predict(q_embs)
        acc = sum(p == t for p, t in zip(preds, q_labels)) / len(q_labels)
        assert acc > 0.8

    def test_predict_proba_sums_to_one(self):
        clf = GaussianPrototype()
        embs, labels = make_support()
        clf.fit(embs, labels)
        q_embs, _ = make_query()
        probs = clf.predict_proba(q_embs)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_diagonal_covariance(self):
        clf = GaussianPrototype(covariance_type="diagonal")
        embs, labels = make_support(k=3)  # few shots
        clf.fit(embs, labels)
        assert clf.covariances_[0].shape == (16,)

    def test_full_covariance_shape(self):
        clf = GaussianPrototype(covariance_type="full")
        embs, labels = make_support(k=10, dim=8)
        clf.fit(embs, labels)
        assert clf.covariances_[0].shape == (8, 8)

    def test_single_shot_stability(self):
        """Should not crash with k=1 support samples."""
        clf = GaussianPrototype(covariance_type="diagonal", regularisation=1e-2)
        rng = np.random.RandomState(0)
        embs = rng.randn(3, 16)
        labels = ["A", "B", "C"]
        clf.fit(embs, labels)
        q_embs = rng.randn(5, 16)
        preds = clf.predict(q_embs)
        assert len(preds) == 5

    def test_unfitted_raises(self):
        clf = GaussianPrototype()
        with pytest.raises(RuntimeError):
            clf.predict(np.random.randn(5, 16))
