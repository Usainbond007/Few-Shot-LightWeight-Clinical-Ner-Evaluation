"""
src/evaluation/metrics.py
──────────────────────────
Precision / Recall / F1 computation for few-shot NER.

Supports:
  - Span-level exact-match evaluation (start, end, label must all match)
  - Token-level evaluation (standard BIO scheme)
  - Per-entity-type breakdown
  - Macro / micro / weighted averaging
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report


@dataclass
class NERMetrics:
    """Container for NER evaluation results."""

    precision: float
    recall: float
    f1: float
    support: int

    # Per-entity type breakdown
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)

    # Averaging strategy used
    average: str = "micro"

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "average": self.average,
            "per_type": self.per_type,
        }

    def __str__(self) -> str:
        lines = [
            f"── NER Metrics ({self.average}) ────────────────────",
            f"  Precision : {self.precision:.4f}",
            f"  Recall    : {self.recall:.4f}",
            f"  F1        : {self.f1:.4f}",
            f"  Support   : {self.support}",
        ]
        if self.per_type:
            lines.append("  Per-type breakdown:")
            for et, vals in self.per_type.items():
                lines.append(
                    f"    {et:20s}  P={vals['precision']:.3f}  "
                    f"R={vals['recall']:.3f}  F1={vals['f1']:.3f}  "
                    f"N={int(vals['support'])}"
                )
        return "\n".join(lines)


def compute_ner_metrics(
    y_true: list[str],
    y_pred: list[str],
    entity_types: Optional[list[str]] = None,
    average: str = "micro",
) -> NERMetrics:
    """
    Compute Precision / Recall / F1 for NER classification outputs.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth entity type labels.
    y_pred : list[str]
        Predicted entity type labels.
    entity_types : list[str] | None
        Ordered list of entity types for per-type breakdown.
        If None, inferred from y_true.
    average : str
        "micro" | "macro" | "weighted"

    Returns
    -------
    NERMetrics
    """
    if not y_true:
        return NERMetrics(precision=0.0, recall=0.0, f1=0.0, support=0, average=average)

    labels = entity_types or sorted(set(y_true))

    p, r, f, supp = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=average,
        zero_division=0,
    )

    # Per-type breakdown
    p_per, r_per, f_per, s_per = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_type = {
        et: {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f_per[i]),
            "support": float(s_per[i]),
        }
        for i, et in enumerate(labels)
    }

    return NERMetrics(
        precision=float(p),
        recall=float(r),
        f1=float(f),
        support=int(np.sum(s_per)),
        per_type=per_type,
        average=average,
    )


def aggregate_episode_metrics(
    episode_metrics: list[NERMetrics],
) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics across multiple few-shot episodes.

    Returns mean ± std for precision, recall, F1 across episodes.
    """
    ps = [m.precision for m in episode_metrics]
    rs = [m.recall for m in episode_metrics]
    fs = [m.f1 for m in episode_metrics]

    return {
        "precision": {"mean": float(np.mean(ps)), "std": float(np.std(ps))},
        "recall": {"mean": float(np.mean(rs)), "std": float(np.std(rs))},
        "f1": {"mean": float(np.mean(fs)), "std": float(np.std(fs))},
        "n_episodes": len(episode_metrics),
    }


def compute_span_f1(
    gold_spans: list[tuple[int, int, str]],
    pred_spans: list[tuple[int, int, str]],
) -> dict[str, float]:
    """
    Exact-match span-level F1.

    Parameters
    ----------
    gold_spans : list of (start, end, label)
    pred_spans : list of (start, end, label)

    Returns
    -------
    dict with keys "precision", "recall", "f1"
    """
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
