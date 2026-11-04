"""
src/models/sbert_baseline.py
─────────────────────────────
Sentence-BERT baseline encoder for few-shot NER.

Uses sentence-transformers to produce span embeddings and plugs into
the same PointPrototype / GaussianPrototype interface as the GLiNER encoder.

Serves as a comparison baseline to measure GLiNER's added value.

Install: pip install sentence-transformers
Model  : "all-MiniLM-L6-v2" (fast, 384-dim) or "all-mpnet-base-v2" (stronger, 768-dim)

Usage
-----
    encoder = SBERTEncoder(model_id="all-MiniLM-L6-v2")
    embs = encoder.encode_span_items(span_items)  # same API as GLiNEREncoder
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SBERTEncoder:
    """
    Sentence-BERT span encoder.

    Encodes the full sentence and uses the sentence embedding as
    the span representation. For better span specificity, a
    [CONTEXT] <sentence> [SEP] <span> template is used.

    Parameters
    ----------
    model_id : str
        Any sentence-transformers model ID.
    device : str
        "auto" | "cpu" | "cuda"
    use_span_template : bool
        If True, encode "[SEP] <span_text>" appended to sentence.
        Adds span context at the cost of slightly more computation.
    batch_size : int
    """

    def __init__(
        self,
        model_id: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        use_span_template: bool = True,
        batch_size: int = 128,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info(f"[SBERTEncoder] Loading '{model_id}' on {device}")
        self._model = SentenceTransformer(model_id, device=device)
        self.model_id = model_id
        self.use_span_template = use_span_template
        self.batch_size = batch_size
        self.embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"[SBERTEncoder] dim={self.embedding_dim}")

    # ── Public API (mirrors GLiNEREncoder) ────────────────────────────────────

    def encode_span_items(self, span_items) -> np.ndarray:
        """
        Encode a list of SpanItem objects into embeddings.

        Parameters
        ----------
        span_items : list[SpanItem]

        Returns
        -------
        np.ndarray of shape (N, embedding_dim)
        """
        texts = self._build_texts(span_items)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings  # (N, D)

    def encode_spans(
        self,
        sentences: list[str],
        spans: list[tuple[int, int]],
        entity_types: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Encode (sentence, span) pairs. Mirrors GLiNEREncoder API."""
        if self.use_span_template:
            texts = [
                f"{sent} [SEP] {sent[s:e]}"
                for sent, (s, e) in zip(sentences, spans)
            ]
        else:
            texts = sentences

        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_texts(self, span_items) -> list[str]:
        if self.use_span_template:
            return [
                f"{item.sentence} [SEP] {item.span_text}"
                for item in span_items
            ]
        return [item.sentence for item in span_items]

    def __repr__(self) -> str:
        return (
            f"SBERTEncoder(model={self.model_id!r}, "
            f"dim={self.embedding_dim}, "
            f"span_template={self.use_span_template})"
        )
