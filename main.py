#!/usr/bin/env python3
"""
Reef Bioacoustics Pipeline — Unified CLI Entry Point
====================================================
Compares:
1. Traditional Ecoacoustic Indices (scikit-maad) -> R^5
2. YAMNet Embeddings (TF Hub) -> R^1024
3. SurfPerch Embeddings (Kaggle Models) -> R^1280

Supports:
- Preprocessing & 5-Second Segment Manifest generation
- In-domain Site-Grouped Cross-Validation (3-fold)
- Geographic Generalization (Leave-One-Country-Out / LOCO)
- Hardware Robustness Degradations (8 kHz, 8-bit, Additive Noise SNR)
- Restoration Trajectory Index (RTI)
- Dual-mode class mapping: 4-class native (D, H, N, R) and 3-class (Degraded, Healthy, Restoration)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
ECOSOUND_SRC = PROJECT_ROOT / "EcoSound-Audit" / "src"
if ECOSOUND_SRC.exists():
    sys.path.insert(0, str(ECOSOUND_SRC))
else:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecosound.evaluation import run_in_domain_site_grouped_cv
from ecosound.features import (
    extract_ecoacoustic_indices,
    load_feature_cache,
    save_feature_cache,
)
from ecosound.generalization import run_leave_one_country_out_benchmark
from ecosound.preprocessing import generate_segment_manifest
from ecosound.robustness import (
    apply_additive_noise_degradation,
    apply_bit_depth_degradation,
    apply_downsampling_degradation,
    evaluate_representation_robustness,
)
from ecosound.rti import (
    compute_degraded_healthy_centroids,
    compute_restoration_trajectory_index,
    evaluate_rti_ecological_correlation,
)
from ecosound.training import tune_and_train_linear_probe


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("reef_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Reef Bioacoustics Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["manifest", "extract_maad", "evaluate", "loco", "robustness", "rti", "all"],
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="4class",
        choices=["4class", "3class"],
        help="Class mapping mode: '4class' (D, H, N, R) or '3class' (Degraded, Healthy, Restoration)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="maad",
        choices=["maad", "yamnet", "surfperch", "all"],
        help="Audio representation to evaluate",
    )
    args = parser.parse_args()

    cfg_file = PROJECT_ROOT / args.config
    if not cfg_file.exists():
        cfg_file = PROJECT_ROOT / "EcoSound-Audit" / "config.yaml"
    config = load_config(cfg_file)

    log_path = PROJECT_ROOT / config["paths"]["pipeline_log"]
    logger = setup_logger(log_path)

    logger.info("=" * 75)
    logger.info("REEF BIOACOUSTICS PIPELINE: ECOACOUSTIC INDICES vs YAMNET vs SURFPERCH")
    logger.info("Mode: %s | Classification: %s | Random Seed: %d", args.mode, args.classes, config["pipeline"]["random_seed"])
    logger.info("=" * 75)

    # 1. Manifest Generation
    sampled_csv = PROJECT_ROOT / config["paths"]["sampled_recordings_csv"]
    seg_manifest_csv = PROJECT_ROOT / config["paths"]["segment_manifest_csv"]

    if args.mode in ("manifest", "all") or not seg_manifest_csv.exists():
        logger.info("Step 1: Checking/generating 5-second segment manifest...")
        generate_segment_manifest(
            sampled_recordings_csv=sampled_csv,
            output_segment_manifest_csv=seg_manifest_csv,
            segment_duration_s=config["audio"]["segment_duration_s"],
            nominal_duration_s=config["audio"]["nominal_file_duration_s"],
        )

    df_manifest = pd.read_csv(seg_manifest_csv)
    logger.info("Master segment manifest ready: %d segments across %d sites.", len(df_manifest), df_manifest["site_id"].nunique())

    # Label column selection
    label_col = "habitat_label" if args.classes == "4class" else "label_3class"
    labels_all = df_manifest[label_col].to_numpy()
    sites_all = df_manifest["site_id"].to_numpy()
    countries_all = df_manifest["country"].to_numpy()
    segment_ids = df_manifest["segment_id"].tolist()

    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    features_dir = PROJECT_ROOT / config["paths"]["features_dir"]
    features_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract or Load Features
    # For local CPU verification or test runs, check cache
    reps_to_run = ["maad"] if args.representation == "maad" else ["maad", "yamnet", "surfperch"]

    feature_matrices = {}
    for rep in reps_to_run:
        cache_file = features_dir / f"features_{rep}.npz"
        if cache_file.exists():
            feats, _ = load_feature_cache(features_dir, rep)
            logger.info("Loaded cached features for %s from %s (shape: %s)", rep, cache_file, feats.shape)
            feature_matrices[rep] = feats
        else:
            logger.info("No cached feature file found at %s.", cache_file)
            if rep == "maad":
                logger.info("To extract full features from remote audio on GPU, use the Colab pipeline notebook.")

    # 3. In-Domain Site-Grouped CV (if feature matrix is available)
    if args.mode in ("evaluate", "all"):
        logger.info("-" * 75)
        logger.info("Step 3: In-Domain Site-Grouped Cross-Validation (Zero Site Leakage)")
        logger.info("-" * 75)
        for rep, feats in feature_matrices.items():
            cv_res = run_in_domain_site_grouped_cv(
                feats,
                labels_all[:len(feats)],
                sites_all[:len(feats)],
                n_splits=config["evaluation"]["in_domain_cv_folds"],
                c_grid=config["classifier"]["c_grid"],
                random_seed=config["pipeline"]["random_seed"],
                representation_name=rep.upper(),
            )
            out_json = results_dir / f"in_domain_{rep}_{args.classes}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(cv_res, f, indent=2)
            logger.info("[%s] In-Domain Macro-F1: %.4f | Site Vote Macro-F1: %.4f",
                        rep.upper(),
                        cv_res["segment_level_metrics"]["macro_f1"],
                        cv_res["site_level_metrics"]["macro_f1"])

    # 4. LOCO Geographic Generalization
    if args.mode in ("loco", "all"):
        logger.info("-" * 75)
        logger.info("Step 4: Leave-One-Country-Out (LOCO) Geographic Generalization")
        logger.info("-" * 75)
        for rep, feats in feature_matrices.items():
            loco_res = run_leave_one_country_out_benchmark(
                feats,
                labels_all[:len(feats)],
                sites_all[:len(feats)],
                countries_all[:len(feats)],
                c_grid=config["classifier"]["c_grid"],
                random_seed=config["pipeline"]["random_seed"],
                representation_name=rep.upper(),
            )
            out_json = results_dir / f"loco_{rep}_{args.classes}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(loco_res, f, indent=2)
            logger.info("[%s] Mean LOCO Macro-F1: %.4f across %d countries",
                        rep.upper(), loco_res["mean_loco_macro_f1"], len(loco_res["loco_folds"]))

    # 5. Ecological Validation (RTI) Compliance
    if args.mode in ("rti", "all"):
        logger.info("-" * 75)
        logger.info("Step 5: Restoration Trajectory Index (RTI) Ecological Validation")
        logger.info("-" * 75)
        rti_eval = evaluate_rti_ecological_correlation(
            site_rti_estimates={},
            ecological_measurements=None,
        )
        out_json = results_dir / "rti_ecological_validation.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rti_eval, f, indent=2)
        logger.info("RTI Status: %s | Compliance Reason: %s", rti_eval["status"], rti_eval["reason"])

    logger.info("=" * 75)
    logger.info("Pipeline execution step finished successfully.")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
