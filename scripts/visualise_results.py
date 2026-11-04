"""
scripts/visualise_results.py
──────────────────────────────
Load saved results and produce publication-quality comparison plots.

Usage:
    python scripts/visualise_results.py --results results/metrics/all_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_f1_vs_shots(results: dict, save_dir: Path) -> None:
    """Bar chart: F1 vs shot size, grouped by method."""
    rows = []
    for shot_key, res in results.items():
        n = res["n_shots"]
        for method_key, method_name in [
            ("point_prototype", "Point Prototype"),
            ("gaussian_prototype", "Gaussian Prototype"),
        ]:
            m = res[method_key]["metrics"]
            rows.append(
                {
                    "Shot Size": n,
                    "Method": method_name,
                    "F1": m["f1"]["mean"],
                    "F1 std": m["f1"]["std"],
                    "Precision": m["precision"]["mean"],
                    "Recall": m["recall"]["mean"],
                }
            )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["Precision", "Recall", "F1"]
    colors = ["#4C72B0", "#DD8452"]

    for ax, metric in zip(axes, metrics):
        for i, (method, color) in enumerate(
            zip(["Point Prototype", "Gaussian Prototype"], colors)
        ):
            sub = df[df["Method"] == method].sort_values("Shot Size")
            x = np.arange(len(sub))
            offset = (i - 0.5) * 0.35
            ax.bar(
                x + offset,
                sub[metric],
                width=0.35,
                label=method,
                color=color,
                alpha=0.85,
            )
            if metric == "F1":
                ax.errorbar(
                    x + offset,
                    sub[metric],
                    yerr=sub["F1 std"],
                    fmt="none",
                    color="black",
                    capsize=4,
                )

        ax.set_xticks(np.arange(len(df["Shot Size"].unique())))
        ax.set_xticklabels(
            [f"{k}-shot" for k in sorted(df["Shot Size"].unique())]
        )
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Shot Size")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Gold-span Classification: Point vs Gaussian Prototype", fontsize=14)
    plt.tight_layout()

    path = save_dir / "f1_vs_shots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def plot_calibration_comparison(results: dict, save_dir: Path) -> None:
    """Line plot: ECE vs shot size for both methods."""
    rows = []
    for shot_key, res in results.items():
        n = res["n_shots"]
        rows.append(
            {
                "Shot Size": n,
                "Method": "Point Prototype",
                "ECE": res["point_prototype"]["calibration"]["ece"],
            }
        )
        rows.append(
            {
                "Shot Size": n,
                "Method": "Gaussian Prototype",
                "ECE": res["gaussian_prototype"]["calibration"]["ece"],
            }
        )

    df = pd.DataFrame(rows).sort_values("Shot Size")

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, color, marker in [
        ("Point Prototype", "#4C72B0", "o"),
        ("Gaussian Prototype", "#DD8452", "s"),
    ]:
        sub = df[df["Method"] == method]
        ax.plot(sub["Shot Size"], sub["ECE"], marker=marker, label=method, color=color)

    ax.set_xlabel("Shot Size (K)")
    ax.set_ylabel("Expected Calibration Error (ECE ↓)")
    ax.set_title("Calibration: ECE vs Shot Size")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = save_dir / "ece_vs_shots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        default="results/metrics/all_results.json",
        help="Path to all_results.json",
    )
    p.add_argument("--save_dir", default="results/plots", help="Output directory")
    args = p.parse_args()

    results = load_results(args.results)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_f1_vs_shots(results, save_dir)
    plot_calibration_comparison(results, save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
