"""Gold-span classification with a frozen GLiNER backbone and held-out notes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

from src.data.synthetic_generator import SyntheticClinicalDataset
from src.data.span_dataset import SpanDataset, EpisodeBuilder
from src.models import get_encoder as _get_gliner_encoder
from src.models.point_prototype import PointPrototype
from src.models.gaussian_prototype import GaussianPrototype
from src.evaluation.metrics import compute_ner_metrics, aggregate_episode_metrics
from src.evaluation.calibration import CalibrationAnalyzer
from src.evaluation.error_analysis import ErrorAnalyzer
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed
from src.utils.io_utils import save_json

logger = get_logger(__name__)


# ── Hydra entry point ──────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("=" * 60)
    logger.info("GLiNER Few-Shot Clinical NER Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.data.seed)

    from src.data.validated import load_validated, split_notes
    from src.data.heldout_episodes import HeldoutEpisodeBuilder
    samples, audit = load_validated(cfg.data.raw_dir)
    split = split_notes(samples, cfg.data.seed)
    output_dir = Path(cfg.output.results_dir)
    if (output_dir / "COMPLETE.json").exists():
        raise FileExistsError("Completed run exists. Set output.results_dir to a fresh directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(audit, output_dir / "data_audit.json")
    save_json({k:[x.note_id for x in v] for k,v in split.items()}, output_dir / "splits.json")
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    episode_builder = HeldoutEpisodeBuilder(
        SpanDataset(split['support'], cfg.data.entity_types),
        SpanDataset(split[cfg.evaluation.split], cfg.data.entity_types),
        cfg.data.entity_types, cfg.data.query_examples_per_type, cfg.data.seed)

    # ── 3. Load frozen encoder ─────────────────────────────────────────────────
    logger.info(f"Loading GLiNER encoder: {cfg.model.gliner_id}")
    encoder = _get_gliner_encoder(
        model_id=cfg.model.gliner_id,
        device=cfg.model.device,
        max_span_len=cfg.model.span_max_len,
        batch_size=cfg.model.batch_size,
        context_tokens=cfg.model.context_tokens,
        cache_dir=cfg.model.cache_dir,
        revision=cfg.model.revision,
    )
    logger.info(f"Encoder ready: {encoder}")

    # ── 4. Few-shot evaluation loop ────────────────────────────────────────────
    all_results: dict = {}
    output_dir = Path(cfg.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_shots in cfg.data.shot_sizes:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Running {n_shots}-shot evaluation ({cfg.support.n_episodes} episodes)")
        logger.info(f"{'─'*50}")

        shot_results = _run_shot_evaluation(
            n_shots=n_shots,
            n_episodes=cfg.support.n_episodes,
            episode_builder=episode_builder,
            encoder=encoder,
            cfg=cfg,
        )
        all_results[f"{n_shots}_shot"] = shot_results

    # ── 5. Save results ────────────────────────────────────────────────────────
    results_path = output_dir / "metrics" / "all_results.json"
    save_json(all_results, results_path)
    logger.info(f"\n✓ Results saved → {results_path}")
    _print_summary(all_results)
    from scripts.visualise_results import plot_f1_vs_shots, plot_calibration_comparison
    plot_f1_vs_shots(all_results, Path(cfg.output.plots_dir))
    plot_calibration_comparison(all_results, Path(cfg.output.plots_dir))
    import importlib.metadata
    save_json({"task": "gold_span_classification", "split": cfg.evaluation.split,
        "encoder_fingerprint": encoder.fingerprint,
        "versions": {p: importlib.metadata.version(p) for p in ['gliner','torch','transformers','numpy']},
        "data_sha256": audit['sha256']}, output_dir / "COMPLETE.json")


def _run_shot_evaluation(
    n_shots: int,
    n_episodes: int,
    episode_builder: EpisodeBuilder,
    encoder: GLiNEREncoder,
    cfg: DictConfig,
) -> dict:
    """Run n_episodes of K-shot evaluation and return aggregated results."""
    plots_dir = Path(cfg.output.plots_dir)
    metrics_dir = Path(cfg.output.metrics_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialise classifiers ────────────────────────────────────────────────
    point_clf = PointPrototype(
        distance_metric=cfg.prototypes.point.distance_metric,
    )
    gauss_clf = GaussianPrototype(
        distance_metric=cfg.prototypes.gaussian.distance_metric,
        covariance_type=cfg.prototypes.gaussian.covariance_type,
        regularisation=cfg.prototypes.gaussian.regularisation,
        shrinkage=cfg.prototypes.gaussian.shrinkage,
    )

    # ── Calibration & error trackers ──────────────────────────────────────────
    cal_analyzer = CalibrationAnalyzer(
        n_bins=cfg.evaluation.calibration.n_bins,
        strategy=cfg.evaluation.calibration.strategy,
    )
    point_error_analyzer = ErrorAnalyzer(entity_types=list(cfg.data.entity_types))
    gauss_error_analyzer = ErrorAnalyzer(entity_types=list(cfg.data.entity_types))

    # ── Per-episode metrics ────────────────────────────────────────────────────
    point_episode_metrics = []
    gauss_episode_metrics = []

    # Confidence + correctness accumulators for calibration
    point_confs, point_correct = [], []
    gauss_confs, gauss_correct = [], []

    episodes = episode_builder.build_episodes(n_shots=n_shots, n_episodes=n_episodes)

    for ep_idx, episode in enumerate(episodes):
        if (ep_idx + 1) % 10 == 0:
            logger.info(f"  Episode {ep_idx + 1}/{n_episodes}")

        # Collect support spans (all entity types)
        support_items = []
        support_labels = []
        for et, items in episode.support.items():
            support_items.extend(items)
            support_labels.extend([et] * len(items))

        # Encode support
        support_embs = encoder.encode_span_items(support_items)

        # Fit prototypes
        point_clf.fit(support_embs, support_labels)
        gauss_clf.fit(support_embs, support_labels)

        # Encode query
        query_items = episode.query
        query_embs = encoder.encode_span_items(query_items)
        query_labels_true = [item.label for item in query_items]

        # Point prototype predictions
        point_preds, point_conf = point_clf.predict_with_confidence(query_embs)
        gauss_preds, gauss_conf = gauss_clf.predict_with_confidence(query_embs)

        # Metrics
        point_metrics = compute_ner_metrics(
            query_labels_true, point_preds, entity_types=episode.entity_types
        )
        gauss_metrics = compute_ner_metrics(
            query_labels_true, gauss_preds, entity_types=episode.entity_types
        )
        point_episode_metrics.append(point_metrics)
        gauss_episode_metrics.append(gauss_metrics)

        # Calibration data
        p_correct = [t == p for t, p in zip(query_labels_true, point_preds)]
        g_correct = [t == p for t, p in zip(query_labels_true, gauss_preds)]
        point_confs.extend(point_conf.tolist())
        point_correct.extend(p_correct)
        gauss_confs.extend(gauss_conf.tolist())
        gauss_correct.extend(g_correct)

        # Error analysis
        point_error_analyzer.add_batch(query_items, point_preds, point_conf)
        gauss_error_analyzer.add_batch(query_items, gauss_preds, gauss_conf)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    point_agg = aggregate_episode_metrics(point_episode_metrics)
    gauss_agg = aggregate_episode_metrics(gauss_episode_metrics)
    for aggregate, episodes_metrics in [(point_agg, point_episode_metrics), (gauss_agg, gauss_episode_metrics)]:
        macro = [np.mean([v['f1'] for v in m.per_type.values()]) for m in episodes_metrics]
        aggregate['macro_f1'] = {'mean': float(np.mean(macro)), 'std': float(np.std(macro))}
    deltas = np.array([g.f1-p.f1 for p,g in zip(point_episode_metrics,gauss_episode_metrics)])
    save_json({'mean_gaussian_minus_point_f1': float(deltas.mean()),
               'std_episode_delta': float(deltas.std()),
               'gaussian_win_fraction': float(np.mean(deltas > 0)),
               'note': 'Paired episode differences; overlapping episodes are not independent.'},
              metrics_dir / f'paired_comparison_{n_shots}shot.json')


    logger.info(
        f"  PointPrototype  F1: {point_agg['f1']['mean']:.4f} ± {point_agg['f1']['std']:.4f}"
    )
    logger.info(
        f"  GaussPrototype  F1: {gauss_agg['f1']['mean']:.4f} ± {gauss_agg['f1']['std']:.4f}"
    )

    # ── Calibration analysis ───────────────────────────────────────────────────
    point_cal = cal_analyzer.analyze(
        np.array(point_confs), np.array(point_correct)
    )
    gauss_cal = cal_analyzer.analyze(
        np.array(gauss_confs), np.array(gauss_correct)
    )
    logger.info(f"  Point calibration: {point_cal}")
    logger.info(f"  Gauss calibration: {gauss_cal}")

    # ── Plot reliability diagrams ──────────────────────────────────────────────
    cal_analyzer.plot_reliability_diagram(
        point_cal,
        title=f"Point Prototype – {n_shots}-shot Calibration",
        save_path=plots_dir / f"calibration_point_{n_shots}shot.png",
    )
    cal_analyzer.plot_reliability_diagram(
        gauss_cal,
        title=f"Gaussian Prototype – {n_shots}-shot Calibration",
        save_path=plots_dir / f"calibration_gauss_{n_shots}shot.png",
    )

    # ── Confusion matrices ─────────────────────────────────────────────────────
    point_error_analyzer.plot_confusion_matrix(
        save_path=plots_dir / f"confusion_point_{n_shots}shot.png",
        title=f"Point Prototype – {n_shots}-shot Confusion Matrix",
    )
    gauss_error_analyzer.plot_confusion_matrix(
        save_path=plots_dir / f"confusion_gauss_{n_shots}shot.png",
        title=f"Gaussian Prototype – {n_shots}-shot Confusion Matrix",
    )

    # ── Confidence distributions ───────────────────────────────────────────────
    point_error_analyzer.plot_confidence_distribution(
        save_path=plots_dir / f"confidence_dist_point_{n_shots}shot.png"
    )
    gauss_error_analyzer.plot_confidence_distribution(
        save_path=plots_dir / f"confidence_dist_gauss_{n_shots}shot.png"
    )

    # ── Save error CSVs ────────────────────────────────────────────────────────
    if cfg.evaluation.error_analysis.save_fp_fn:
        point_error_analyzer.save_errors(
            metrics_dir / f"errors_point_{n_shots}shot.csv"
        )
        gauss_error_analyzer.save_errors(
            metrics_dir / f"errors_gauss_{n_shots}shot.csv"
        )

    save_json({"point": [m.to_dict() for m in point_episode_metrics], "gaussian": [m.to_dict() for m in gauss_episode_metrics]},
              metrics_dir / f"episodes_{n_shots}shot.json")
    return {
        "n_shots": n_shots,
        "point_prototype": {
            "metrics": point_agg,
            "calibration": {
                "ece": point_cal.ece,
                "mce": point_cal.mce,
            },
        },
        "gaussian_prototype": {
            "metrics": gauss_agg,
            "calibration": {
                "ece": gauss_cal.ece,
                "mce": gauss_cal.mce,
            },
        },
    }


def _print_summary(all_results: dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 75)
    print(f"{'RESULTS SUMMARY':^75}")
    print("=" * 75)
    header = f"{'K-Shot':<10} {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ECE':>8}"
    print(header)
    print("-" * 75)
    for shot_key, res in all_results.items():
        n = res["n_shots"]
        for method_key, method_name in [
            ("point_prototype", "Point Prototype"),
            ("gaussian_prototype", "Gaussian Prototype"),
        ]:
            m = res[method_key]["metrics"]
            c = res[method_key]["calibration"]
            print(
                f"{f'{n}-shot':<10} {method_name:<22} "
                f"{m['precision']['mean']:>9.4f} "
                f"{m['recall']['mean']:>9.4f} "
                f"{m['f1']['mean']:>9.4f} "
                f"{c['ece']:>7.4f}"
            )
        print("-" * 75)
    print("=" * 75)


if __name__ == "__main__":
    main()
