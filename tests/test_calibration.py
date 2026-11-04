"""
tests/test_calibration.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from src.evaluation.calibration import CalibrationAnalyzer


class TestCalibrationAnalyzer:
    def test_perfect_calibration(self):
        """All predictions correct at high confidence → low ECE."""
        conf = np.linspace(0.9, 1.0, 100)
        correct = np.ones(100)
        cal = CalibrationAnalyzer(n_bins=5).analyze(conf, correct)
        assert cal.ece < 0.1

    def test_overconfident(self):
        """High confidence but random correctness → overconfident."""
        rng = np.random.RandomState(0)
        conf = np.full(200, 0.9)
        correct = rng.randint(0, 2, 200).astype(float)
        cal = CalibrationAnalyzer(n_bins=10).analyze(conf, correct)
        assert cal.is_overconfident()

    def test_bin_counts_sum_to_n(self):
        rng = np.random.RandomState(42)
        conf = rng.uniform(0, 1, 500)
        correct = (conf > 0.5).astype(float)
        cal = CalibrationAnalyzer(n_bins=10).analyze(conf, correct)
        assert cal.bin_counts.sum() == 500

    def test_quantile_strategy(self):
        rng = np.random.RandomState(7)
        conf = rng.beta(2, 5, 300)  # skewed distribution
        correct = (conf > conf.mean()).astype(float)
        cal = CalibrationAnalyzer(n_bins=10, strategy="quantile").analyze(conf, correct)
        assert 0 <= cal.ece <= 1.0

    def test_plot_returns_figure(self, tmp_path):
        rng = np.random.RandomState(0)
        conf = rng.uniform(0.5, 1.0, 100)
        correct = rng.randint(0, 2, 100).astype(float)
        cal = CalibrationAnalyzer().analyze(conf, correct)
        analyzer = CalibrationAnalyzer()
        fig = analyzer.plot_reliability_diagram(
            cal,
            save_path=tmp_path / "reliability.png",
        )
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "reliability.png").exists()
