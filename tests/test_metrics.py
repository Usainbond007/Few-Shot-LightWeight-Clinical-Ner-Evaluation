"""
tests/test_metrics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_ner_metrics,
    aggregate_episode_metrics,
    compute_span_f1,
    NERMetrics,
)


class TestNERMetrics:
    def test_perfect_predictions(self):
        y = ["Drug", "Problem", "Drug", "Test"]
        m = compute_ner_metrics(y, y, entity_types=["Drug", "Problem", "Test"])
        assert m.f1 == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = ["Drug", "Drug", "Drug"]
        y_pred = ["Problem", "Problem", "Problem"]
        m = compute_ner_metrics(y_true, y_pred, entity_types=["Drug", "Problem"])
        assert m.f1 == pytest.approx(0.0)

    def test_per_type_breakdown(self):
        y_true = ["Drug", "Drug", "Problem", "Problem"]
        y_pred = ["Drug", "Problem", "Problem", "Problem"]
        m = compute_ner_metrics(y_true, y_pred, entity_types=["Drug", "Problem"])
        assert "Drug" in m.per_type
        assert "Problem" in m.per_type

    def test_empty_input(self):
        m = compute_ner_metrics([], [])
        assert m.f1 == 0.0

    def test_to_dict_contains_keys(self):
        y = ["Drug"]
        m = compute_ner_metrics(y, y)
        d = m.to_dict()
        assert "f1" in d and "precision" in d and "recall" in d


class TestAggregateEpisodeMetrics:
    def test_mean_std_computed(self):
        metrics = [
            NERMetrics(precision=0.8, recall=0.7, f1=0.75, support=10),
            NERMetrics(precision=0.9, recall=0.85, f1=0.87, support=10),
        ]
        agg = aggregate_episode_metrics(metrics)
        assert agg["f1"]["mean"] == pytest.approx(0.81, abs=0.01)
        assert agg["n_episodes"] == 2


class TestSpanF1:
    def test_exact_match(self):
        gold = [(0, 5, "Drug"), (10, 20, "Problem")]
        pred = [(0, 5, "Drug"), (10, 20, "Problem")]
        r = compute_span_f1(gold, pred)
        assert r["f1"] == pytest.approx(1.0)

    def test_no_match(self):
        gold = [(0, 5, "Drug")]
        pred = [(0, 5, "Problem")]  # wrong label
        r = compute_span_f1(gold, pred)
        assert r["f1"] == pytest.approx(0.0)

    def test_partial_match(self):
        gold = [(0, 5, "Drug"), (10, 20, "Problem")]
        pred = [(0, 5, "Drug")]
        r = compute_span_f1(gold, pred)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(2/3, abs=0.01)
