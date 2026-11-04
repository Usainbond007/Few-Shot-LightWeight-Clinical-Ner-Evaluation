"""
src/models/temperature_calibration.py
───────────────────────────────────────
Post-hoc confidence calibration for prototype classifiers.

Implements:
  1. Temperature Scaling  – learns a single scalar T on a held-out set.
     New confidence = softmax(logits / T).
     T > 1 → softer (less confident), T < 1 → sharper.

  2. Platt Scaling       – learns a per-class sigmoid (a*score + b).
     Better for binary-style recalibration of the top-1 confidence.

  3. Isotonic Regression – non-parametric monotone calibration.
     Most flexible but needs more calibration data.

Usage
-----
    cal = TemperatureScaling()
    cal.fit(logits_val, labels_val)          # fit on held-out split
    calibrated_probs = cal.calibrate(logits_test)

Reference
---------
    Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import softmax as scipy_softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseCalibrator:
    """Abstract base for confidence calibrators."""

    def fit(self, logits: np.ndarray, labels: list[str]) -> "BaseCalibrator":
        raise NotImplementedError

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probability array (N, C)."""
        raise NotImplementedError

    def calibrate_confidence(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated max-probability confidence (N,)."""
        return self.calibrate(logits).max(axis=1)


# ── Temperature Scaling ────────────────────────────────────────────────────────

class TemperatureScaling(BaseCalibrator):
    """
    Temperature scaling calibration (Guo et al. 2017).

    A single temperature T is optimised to minimise NLL on a validation set.
    T is shared across all classes — it only reshapes the softmax distribution.

    Parameters
    ----------
    init_temp : float
        Starting temperature for optimisation.
    """

    def __init__(self, init_temp: float = 1.0) -> None:
        self.init_temp = init_temp
        self.temperature_: float = init_temp

    def fit(self, logits: np.ndarray, labels: list[str]) -> "TemperatureScaling":
        """
        Find optimal temperature T by minimising NLL on (logits, labels).

        Parameters
        ----------
        logits : np.ndarray, shape (N, C)
            Raw (uncalibrated) logit scores.
        labels : list[str], length N
            Ground-truth class labels.
        """
        classes = sorted(set(labels))
        label_idx = {c: i for i, c in enumerate(classes)}
        y = np.array([label_idx[l] for l in labels])

        def nll(t):
            t = max(t, 1e-6)
            probs = scipy_softmax(logits / t, axis=1)
            probs = np.clip(probs, 1e-9, 1.0)
            return -np.mean(np.log(probs[np.arange(len(y)), y]))

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.temperature_ = float(result.x)
        logger.info(f"[TemperatureScaling] Optimal T = {self.temperature_:.4f}")
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return softmax probabilities."""
        return scipy_softmax(logits / self.temperature_, axis=1)

    def __repr__(self) -> str:
        return f"TemperatureScaling(T={self.temperature_:.4f})"


# ── Platt Scaling ──────────────────────────────────────────────────────────────

class PlattScaling(BaseCalibrator):
    """
    Platt scaling: fit a logistic regression on top-1 confidence scores.

    Less principled than temperature scaling but sometimes more flexible
    when the model is severely miscalibrated.
    """

    def __init__(self) -> None:
        self._lr: Optional[LogisticRegression] = None

    def fit(self, logits: np.ndarray, labels: list[str]) -> "PlattScaling":
        probs = scipy_softmax(logits, axis=1)
        conf = probs.max(axis=1).reshape(-1, 1)   # (N, 1)

        classes = sorted(set(labels))
        label_idx = {c: i for i, c in enumerate(classes)}
        y_bin = np.array([
            1 if (label_idx.get(l, -1) == probs[i].argmax()) else 0
            for i, l in enumerate(labels)
        ])

        self._lr = LogisticRegression(C=1.0, solver="lbfgs")
        self._lr.fit(conf, y_bin)
        logger.info("[PlattScaling] Fitted logistic calibrator.")
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("PlattScaling not fitted. Call .fit() first.")
        probs = scipy_softmax(logits, axis=1)
        conf = probs.max(axis=1).reshape(-1, 1)
        scale = self._lr.predict_proba(conf)[:, 1]    # calibrated confidence
        # Rescale all class probs so the max matches the calibrated confidence
        max_probs = probs.max(axis=1, keepdims=True) + 1e-9
        return probs * (scale.reshape(-1, 1) / max_probs)


# ── Isotonic Regression ────────────────────────────────────────────────────────

class IsotonicCalibrator(BaseCalibrator):
    """
    Non-parametric isotonic regression calibrator (per-class).

    Fits one isotonic regressor per class on the class-specific
    probability vs. accuracy relationship.
    """

    def __init__(self) -> None:
        self._regressors: dict[int, IsotonicRegression] = {}
        self._classes: list[str] = []

    def fit(self, logits: np.ndarray, labels: list[str]) -> "IsotonicCalibrator":
        self._classes = sorted(set(labels))
        label_idx = {c: i for i, c in enumerate(self._classes)}
        y = np.array([label_idx[l] for l in labels])
        probs = scipy_softmax(logits, axis=1)

        for c_idx in range(len(self._classes)):
            # For class c: input=predicted prob, target=1 if correct
            p_c = probs[:, c_idx]
            correct = (y == c_idx).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p_c, correct)
            self._regressors[c_idx] = ir

        logger.info(f"[IsotonicCalibrator] Fitted {len(self._classes)} regressors.")
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        if not self._regressors:
            raise RuntimeError("IsotonicCalibrator not fitted. Call .fit() first.")
        probs = scipy_softmax(logits, axis=1)
        cal_probs = np.zeros_like(probs)
        for c_idx, ir in self._regressors.items():
            cal_probs[:, c_idx] = ir.predict(probs[:, c_idx])
        # Normalise rows
        row_sums = cal_probs.sum(axis=1, keepdims=True)
        return cal_probs / np.where(row_sums > 0, row_sums, 1.0)


# ── Factory ────────────────────────────────────────────────────────────────────

def get_calibrator(method: str = "temperature") -> BaseCalibrator:
    """
    Factory function.

    Parameters
    ----------
    method : "temperature" | "platt" | "isotonic"
    """
    mapping = {
        "temperature": TemperatureScaling,
        "platt": PlattScaling,
        "isotonic": IsotonicCalibrator,
    }
    if method not in mapping:
        raise ValueError(f"Unknown calibration method: {method!r}. Choose from {list(mapping)}")
    return mapping[method]()
