"""
scripts/generate_synthetic_data.py
────────────────────────────────────
Standalone script to generate and inspect synthetic clinical data.

Usage:
    python scripts/generate_synthetic_data.py --n_samples 5000 --output data/synthetic/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_generator import SyntheticClinicalDataset, _ENTITIES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic clinical NER data.")
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--output", type=str, default="data/synthetic/clinical_samples.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview", action="store_true", help="Print 5 samples and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gen = SyntheticClinicalDataset(n_samples=args.n_samples, seed=args.seed)

    if args.preview:
        for s in gen.generate()[:5]:
            print(f"Text : {s.text}")
            print(f"Spans: {s.spans}")
            print()
        return

    gen.save(args.output)

    # Print statistics
    samples = SyntheticClinicalDataset.load(args.output)
    from collections import Counter

    label_counts: Counter = Counter()
    for s in samples:
        for span in s.spans:
            label_counts[span["label"]] += 1

    print(f"\n{'Entity Type':<25} {'Count':>8}")
    print("-" * 35)
    for label, count in sorted(label_counts.items()):
        print(f"{label:<25} {count:>8}")
    print(f"{'TOTAL':<25} {sum(label_counts.values()):>8}")


if __name__ == "__main__":
    main()
