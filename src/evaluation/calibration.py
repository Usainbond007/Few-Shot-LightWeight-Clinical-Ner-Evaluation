"""
src/evaluation/calibration.py
──────────────────────────────
Calibration analysis for prototype classifier confidence scores.

Measures how well predicted confidence (max softmax probability) correlates
with actual accuracy. Produces:
  - Reliability diagrams (calibration curves)
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Overconfidence / Underconfidence diagnosis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Calibration analysis output."""

    ece: float          # Expected Calibration Error
    mce: float          # Maximum Calibration Error
    bin_confidences: np.ndarray   # Mean confidence per bin
    bin_accuracies: np.ndarray    # Mean accuracy per bin
    bin_counts: np.ndarray        # Number of samples per bin
    n_bins: int
    strategy: str

    def is_overconfident(self) -> bool:
        """Returns True if model is systematically overconfident."""
        valid = self.bin_counts > 0
        return float(
            np.mean(self.bin_confidences[valid] - self.bin_accuracies[valid])
        ) > 0.0

    def __str__(self) -> str:
        direction = "OVERCONFIDENT" if self.is_overconfident() else "UNDERCONFIDENT"
        return (
            f"CalibrationResult(ECE={self.ece:.4f}, MCE={self.mce:.4f}, "
            f"{direction})"
        )


class CalibrationAnalyzer:
    """
    Analyses calibration of prototype classifier confidence scores.

    Parameters
    ----------
    n_bins : int
        Number of bins for the reliability diagram.
    strategy : str
        "uniform" – equal-width confidence bins [0,1].
        "quantile" – equal-frequency bins.
    """

    def __init__(self, n_bins: int = 10, strategy: str = "uniform") -> None:
        if strategy not in ("uniform", "quantile"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.n_bins = n_bins
        self.strategy = strategy

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
    ) -> CalibrationResult:
        """
        Compute calibration metrics.

        Parameters
        ----------
        confidences : np.ndarray, shape (N,)
            Predicted confidence (max softmax probability) for each sample.
        correct : np.ndarray of bool, shape (N,)
            Whether each prediction was correct.

        Returns
        -------
        CalibrationResult
        """
        confidences = np.asarray(confidences, dtype=float)
        correct = np.asarray(correct, dtype=float)

        bin_edges = self._compute_bin_edges(confidences)
        bin_ids = np.digitize(confidences, bin_edges[:-1]) - 1
        bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)

        bin_confidences = np.zeros(self.n_bins)
        bin_accuracies = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins, dtype=int)

        for b in range(self.n_bins):
            mask = bin_ids == b
            if mask.sum() > 0:
                bin_confidences[b] = confidences[mask].mean()
                bin_accuracies[b] = correct[mask].mean()
                bin_counts[b] = mask.sum()

        # ECE: weighted average of |confidence - accuracy|
        total = bin_counts.sum()
        ece = float(
            np.sum(bin_counts * np.abs(bin_confidences - bin_accuracies)) / total
        ) if total > 0 else 0.0

        # MCE: max calibration error across non-empty bins
        valid = bin_counts > 0
        mce = float(
            np.max(np.abs(bin_confidences[valid] - bin_accuracies[valid]))
        ) if valid.any() else 0.0

        return CalibrationResult(
            ece=ece,
            mce=mce,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
            n_bins=self.n_bins,
            strategy=self.strategy,
        )

    def plot_reliability_diagram(
        self,
        result: CalibrationResult,
        title: str = "Reliability Diagram",
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Plot a reliability diagram (calibration curve).

        The ideal calibration line y = x is shown in dashed orange.
        Blue bars show observed accuracy per confidence bin.
        Red areas show the calibration gap.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        valid = result.bin_counts > 0
        x = result.bin_confidences
        y = result.bin_accuracies

        # Gap bars (red = overconfident, blue = underconfident)
        for i in range(result.n_bins):
            if result.bin_counts[i] == 0:
                continue
            conf_i = result.bin_confidences[i]
            acc_i = result.bin_accuracies[i]
            color = "#d62728" if conf_i > acc_i else "#1f77b4"
            ax.bar(conf_i, acc_i, width=0.1, alpha=0.7, color="#aec7e8", align="center")
            ax.bar(
                conf_i,
                acc_i - conf_i,
                bottom=conf_i,
                width=0.1,
                alpha=0.5,
                color=color,
                align="center",
            )

        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE={result.ece:.4f}  MCE={result.mce:.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)

        # Histogram of confidences
        ax2 = axes[1]
        ax2.bar(
            result.bin_confidences[valid],
            result.bin_counts[valid],
            width=0.08,
            color="#2ca02c",
            alpha=0.7,
        )
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Count")
        ax2.set_title("Confidence Distribution")
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"[CalibrationAnalyzer] Plot saved → {save_path}")

        return fig

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_bin_edges(self, confidences: np.ndarray) -> np.ndarray:
        if self.strategy == "uniform":
            return np.linspace(0.0, 1.0, self.n_bins + 1)
        else:  # quantile
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            return np.percentile(confidences, quantiles)
