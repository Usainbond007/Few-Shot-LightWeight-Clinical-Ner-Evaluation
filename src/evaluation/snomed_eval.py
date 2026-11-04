"""
src/evaluation/snomed_eval.py
──────────────────────────────
SNOMED CT hierarchy-aware NER evaluation.

Standard exact-match F1 penalises any label mismatch equally —
e.g. predicting "Treatment" for a "Drug" span is treated the same as
predicting a completely unrelated class.

Hierarchy-aware evaluation gives partial credit when the predicted
label is a sibling or ancestor of the true label in the clinical
entity ontology.

We implement two complementary approaches:

1. **Soft F1** – replaces the binary TP indicator with a similarity
   score drawn from a concept distance matrix (0–1).
   similarity(true, pred) = 1 − (edge_distance / max_distance)

2. **Hierarchical grouping** – collapses fine-grained entity types to
   coarser semantic groups before computing standard F1.
   E.g. "Drug" + "Dosage" + "Frequency" → "Medication"

Usage
-----
    hier_eval = HierarchyAwareEvaluator(taxonomy=CLINICAL_TAXONOMY)
    soft_f1 = hier_eval.soft_f1(y_true, y_pred)
    coarse_metrics = hier_eval.coarse_f1(y_true, y_pred)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)

# ── Default clinical entity taxonomy ──────────────────────────────────────────
# Structured as a tree: {parent: [children]}
# Reflects the i2b2/n2c2 medication+problem+test hierarchy

CLINICAL_TAXONOMY: dict[str, list[str]] = {
    "Clinical Entity": ["Clinical Finding", "Medication", "Procedure", "Observation"],
    "Clinical Finding": ["Problem"],
    "Medication": ["Drug", "Dosage", "Frequency", "Duration"],
    "Procedure": ["Treatment", "Test"],
    "Observation": [],
}

# Flat label → coarse group mapping (derived from taxonomy above)
COARSE_GROUPS: dict[str, str] = {
    "Problem":   "Clinical Finding",
    "Drug":      "Medication",
    "Dosage":    "Medication",
    "Frequency": "Medication",
    "Duration":  "Medication",
    "Treatment": "Procedure",
    "Test":      "Procedure",
}


@dataclass
class HierarchyAwareResult:
    soft_f1: float
    coarse_precision: float
    coarse_recall: float
    coarse_f1: float
    similarity_matrix: dict[tuple[str, str], float]

    def __str__(self) -> str:
        return (
            f"HierarchyAware | Soft-F1={self.soft_f1:.4f} | "
            f"Coarse P={self.coarse_precision:.4f} "
            f"R={self.coarse_recall:.4f} "
            f"F1={self.coarse_f1:.4f}"
        )


class HierarchyAwareEvaluator:
    """
    Evaluates NER predictions with partial credit for ontologically
    related label confusions.

    Parameters
    ----------
    taxonomy : dict[str, list[str]]
        Tree structure mapping parent → list of children.
        Defaults to CLINICAL_TAXONOMY.
    coarse_groups : dict[str, str]
        Maps fine-grained labels to coarse semantic groups.
        Defaults to COARSE_GROUPS.
    """

    def __init__(
        self,
        taxonomy: Optional[dict[str, list[str]]] = None,
        coarse_groups: Optional[dict[str, str]] = None,
    ) -> None:
        self.taxonomy = taxonomy or CLINICAL_TAXONOMY
        self.coarse_groups = coarse_groups or COARSE_GROUPS
        self._sim_matrix = self._build_similarity_matrix()

    # ── Public API ────────────────────────────────────────────────────────────

    def soft_f1(self, y_true: list[str], y_pred: list[str]) -> float:
        """
        Compute soft F1 where TP is weighted by label similarity.

        similarity(A, B) = 1 if A == B
                         = 1 - (tree_distance(A,B) / max_distance)  otherwise
        """
        if not y_true:
            return 0.0

        soft_tp = sum(
            self._similarity(t, p)
            for t, p in zip(y_true, y_pred)
        )
        soft_fp = sum(
            1.0 - self._similarity(t, p)
            for t, p in zip(y_true, y_pred)
        )
        soft_fn = sum(
            1.0 - self._similarity(t, p)
            for t, p in zip(y_true, y_pred)
        )

        precision = soft_tp / (soft_tp + soft_fp) if (soft_tp + soft_fp) > 0 else 0.0
        recall    = soft_tp / (soft_tp + soft_fn) if (soft_tp + soft_fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def coarse_f1(
        self,
        y_true: list[str],
        y_pred: list[str],
        average: str = "macro",
    ) -> HierarchyAwareResult:
        """
        Map fine-grained labels to coarse groups, then compute standard F1.
        """
        y_true_c = [self.coarse_groups.get(l, l) for l in y_true]
        y_pred_c = [self.coarse_groups.get(l, l) for l in y_pred]

        coarse_labels = sorted(set(y_true_c))
        p, r, f, _ = precision_recall_fscore_support(
            y_true_c, y_pred_c,
            labels=coarse_labels,
            average=average,
            zero_division=0,
        )

        return HierarchyAwareResult(
            soft_f1=self.soft_f1(y_true, y_pred),
            coarse_precision=float(p),
            coarse_recall=float(r),
            coarse_f1=float(f),
            similarity_matrix=self._sim_matrix,
        )

    def similarity_table(self) -> str:
        """Pretty-print the label similarity matrix."""
        labels = sorted({l for pair in self._sim_matrix for l in pair})
        lines = ["Similarity Matrix:"]
        header = f"{'':15}" + "".join(f"{l:15}" for l in labels)
        lines.append(header)
        for l1 in labels:
            row = f"{l1:15}"
            for l2 in labels:
                sim = self._sim_matrix.get((l1, l2), self._sim_matrix.get((l2, l1), 0.0))
                row += f"{sim:<15.3f}"
            lines.append(row)
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _similarity(self, label_a: str, label_b: str) -> float:
        if label_a == label_b:
            return 1.0
        return self._sim_matrix.get(
            (label_a, label_b),
            self._sim_matrix.get((label_b, label_a), 0.0),
        )

    def _build_similarity_matrix(self) -> dict[tuple[str, str], float]:
        """Compute pairwise similarity from tree edit distances."""
        # Collect all leaf/internal labels mentioned in coarse_groups
        all_labels = list(COARSE_GROUPS.keys()) + list(set(COARSE_GROUPS.values()))
        all_labels = list(dict.fromkeys(all_labels))  # deduplicate, preserve order

        # Build adjacency for BFS
        edges: dict[str, list[str]] = {}
        for parent, children in self.taxonomy.items():
            for child in children:
                edges.setdefault(parent, []).append(child)
                edges.setdefault(child, []).append(parent)

        def bfs_distance(src: str, dst: str) -> int:
            if src == dst:
                return 0
            visited = {src}
            queue = [(src, 0)]
            while queue:
                node, dist = queue.pop(0)
                for nb in edges.get(node, []):
                    if nb == dst:
                        return dist + 1
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, dist + 1))
            return 999  # unreachable → max distance

        # Compute all pairwise distances between leaf labels
        sim_matrix: dict[tuple[str, str], float] = {}
        max_dist = 1

        for i, la in enumerate(all_labels):
            for lb in all_labels[i + 1:]:
                d = bfs_distance(la, lb)
                max_dist = max(max_dist, d)
                sim_matrix[(la, lb)] = d

        # Normalise to similarity [0, 1]
        for key in sim_matrix:
            raw_d = sim_matrix[key]
            sim_matrix[key] = max(0.0, 1.0 - raw_d / max_dist) if raw_d < 999 else 0.0

        return sim_matrix
