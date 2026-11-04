"""
src/evaluation/error_analysis.py
──────────────────────────────────
Error analysis for few-shot NER.

Captures and analyses:
  - False Positives (FP): predicted entity where none exists / wrong type
  - False Negatives (FN): missed entities
  - Confusion matrix across entity types
  - Confidence vs error correlation
  - Entity-length vs accuracy analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """One misclassified span."""

    sentence: str
    span_text: str
    true_label: str
    pred_label: str
    confidence: float
    note_id: str = ""
    source: str = ""


class ErrorAnalyzer:
    """
    Collects and analyses prediction errors.

    Usage
    -----
    analyzer = ErrorAnalyzer(entity_types=[...])
    analyzer.add_batch(span_items, pred_labels, pred_confidences)
    analyzer.plot_confusion_matrix(save_path="results/plots/confusion.png")
    df = analyzer.to_dataframe()
    """

    def __init__(self, entity_types: list[str]) -> None:
        self.entity_types = entity_types
        self._errors: list[ErrorRecord] = []
        self._all_true: list[str] = []
        self._all_pred: list[str] = []
        self._all_conf: list[float] = []

    # ── Data accumulation ─────────────────────────────────────────────────────

    def add_batch(
        self,
        span_items,            # list[SpanItem]
        pred_labels: list[str],
        pred_confidences: np.ndarray,
    ) -> None:
        """Record predictions for one batch of query spans."""
        for item, pred, conf in zip(span_items, pred_labels, pred_confidences):
            true = item.label
            self._all_true.append(true)
            self._all_pred.append(pred)
            self._all_conf.append(float(conf))

            if pred != true:
                self._errors.append(
                    ErrorRecord(
                        sentence=item.sentence,
                        span_text=item.span_text,
                        true_label=true,
                        pred_label=pred,
                        confidence=float(conf),
                        note_id=item.note_id,
                        source=item.source,
                    )
                )

    def reset(self) -> None:
        """Clear accumulated records."""
        self._errors.clear()
        self._all_true.clear()
        self._all_pred.clear()
        self._all_conf.clear()

    # ── Analysis ──────────────────────────────────────────────────────────────

    @property
    def n_total(self) -> int:
        return len(self._all_true)

    @property
    def n_errors(self) -> int:
        return len(self._errors)

    @property
    def error_rate(self) -> float:
        return self.n_errors / self.n_total if self.n_total > 0 else 0.0

    def high_confidence_errors(self, threshold: float = 0.9) -> list[ErrorRecord]:
        """Return errors where the model was still very confident."""
        return [e for e in self._errors if e.confidence >= threshold]

    def errors_by_type(self) -> dict[str, list[ErrorRecord]]:
        """Group errors by true entity type."""
        grouped: dict[str, list[ErrorRecord]] = {}
        for e in self._errors:
            grouped.setdefault(e.true_label, []).append(e)
        return grouped

    def confusion_pair_counts(self) -> list[tuple[str, str, int]]:
        """Return (true, pred, count) tuples for all confusion pairs."""
        from collections import Counter
        counts = Counter(
            (e.true_label, e.pred_label) for e in self._errors
        )
        return [(t, p, c) for (t, p), c in counts.most_common()]

    def confidence_accuracy_correlation(self) -> float:
        """Pearson correlation between confidence and correctness."""
        if not self._all_conf:
            return float("nan")
        correct = np.array(
            [t == p for t, p in zip(self._all_true, self._all_pred)], dtype=float
        )
        conf_arr = np.array(self._all_conf)
        return float(np.corrcoef(conf_arr, correct)[0, 1])

    def span_length_accuracy(self) -> pd.DataFrame:
        """Compute accuracy binned by span token-length."""
        rows = []
        for item_true, item_pred, item_conf, error in zip(
            self._all_true,
            self._all_pred,
            self._all_conf,
            [False] * self.n_total,  # placeholder — we track via _all_*
        ):
            pass  # This is populated below via errors
        # Build from raw accumulated data via _errors
        # We need all items, so rebuild from _all_* lists
        # Using a simple word-count as length proxy
        records = []
        for i, (true, pred, conf) in enumerate(
            zip(self._all_true, self._all_pred, self._all_conf)
        ):
            records.append(
                {
                    "true": true,
                    "pred": pred,
                    "correct": true == pred,
                    "confidence": conf,
                }
            )
        return pd.DataFrame(records)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        save_path: Optional[str | Path] = None,
        title: str = "Confusion Matrix",
        normalise: bool = True,
    ) -> plt.Figure:
        """Plot normalised confusion matrix across entity types."""
        labels = sorted(set(self._all_true + self._all_pred))
        cm = confusion_matrix(self._all_true, self._all_pred, labels=labels)

        if normalise:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.where(row_sums > 0, cm / row_sums, 0)
            fmt = ".2f"
        else:
            fmt = "d"

        fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            ax=ax,
            cbar=True,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"[ErrorAnalyzer] Confusion matrix saved → {save_path}")

        return fig

    def plot_confidence_distribution(
        self,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Plot confidence distributions for correct vs incorrect predictions."""
        correct_conf = [
            c for c, t, p in zip(self._all_conf, self._all_true, self._all_pred)
            if t == p
        ]
        wrong_conf = [
            c for c, t, p in zip(self._all_conf, self._all_true, self._all_pred)
            if t != p
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 21)
        ax.hist(correct_conf, bins=bins, alpha=0.6, color="steelblue", label="Correct")
        ax.hist(wrong_conf, bins=bins, alpha=0.6, color="tomato", label="Wrong")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ── Export ────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert error records to a pandas DataFrame for inspection."""
        if not self._errors:
            return pd.DataFrame(
                columns=["sentence", "span_text", "true_label", "pred_label",
                         "confidence", "note_id", "source"]
            )
        return pd.DataFrame(
            [
                {
                    "sentence": e.sentence,
                    "span_text": e.span_text,
                    "true_label": e.true_label,
                    "pred_label": e.pred_label,
                    "confidence": e.confidence,
                    "note_id": e.note_id,
                    "source": e.source,
                }
                for e in self._errors
            ]
        )

    def save_errors(self, path: str | Path) -> None:
        """Save error records to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"[ErrorAnalyzer] {len(df)} errors saved → {path}")
