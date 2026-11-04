"""
tests/test_synthetic_generator.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.data.synthetic_generator import SyntheticClinicalDataset, ClinicalSample


def test_generates_correct_count():
    gen = SyntheticClinicalDataset(n_samples=50, seed=0)
    samples = gen.generate()
    assert len(samples) == 50


def test_all_samples_have_text_and_spans():
    gen = SyntheticClinicalDataset(n_samples=20, seed=1)
    for s in gen.generate():
        assert isinstance(s, ClinicalSample)
        assert len(s.text) > 5
        assert len(s.spans) >= 0


def test_span_char_offsets_valid():
    gen = SyntheticClinicalDataset(n_samples=100, seed=2)
    for s in gen.generate():
        for span in s.spans:
            assert span["start"] >= 0
            assert span["end"] <= len(s.text)
            assert s.text[span["start"]:span["end"]] == span["text"]


def test_save_and_load(tmp_path):
    gen = SyntheticClinicalDataset(n_samples=30, seed=3)
    path = tmp_path / "test.jsonl"
    gen.save(path)
    loaded = SyntheticClinicalDataset.load(path)
    assert len(loaded) == 30
    assert all(isinstance(s, ClinicalSample) for s in loaded)


def test_custom_entity_types():
    gen = SyntheticClinicalDataset(entity_types=["Drug", "Dosage"], n_samples=20, seed=4)
    for s in gen.generate():
        for span in s.spans:
            assert span["label"] in ("Drug", "Dosage")


def test_invalid_entity_type_raises():
    with pytest.raises(ValueError, match="Unknown entity type"):
        SyntheticClinicalDataset(entity_types=["InvalidType"])
