"""
EcoSound-Audit — Geographic Generalization (LOCO / LORO)
=========================================================
Implements:
1. Leave-One-Country-Out (LOCO) / Leave-One-Region-Out (LORO) Evaluation.
2. For each held-out country:
   - Train probe on all other 4 countries.
   - Evaluate on held-out country.
   - Explicitly document evaluable classes and support per country.
   - Compute performance drop vs in-domain (Delta F1).
   - Cluster bootstrap confidence intervals over site clusters.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ecosound.evaluation import (
    aggregate_site_level_predictions,
    compute_comprehensive_metrics,
    site_cluster_bootstrap_ci,
)
from ecosound.training import TrainedProbe, predict_probe, tune_and_train_linear_probe

logger = logging.getLogger("reef_pipeline")


def run_leave_one_country_out_benchmark(
    features: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    countries: np.ndarray,
    in_domain_baseline_f1: Optional[float] = None,
    c_grid: Optional[List[float]] = None,
    random_seed: int = 42,
    representation_name: str = "Representation",
) -> Dict[str, Any]:
    """
    Run Leave-One-Country-Out (LOCO) evaluation across all present countries.
    """
    unique_countries = sorted(np.unique(countries).tolist())
    loco_results = {}

    for held_out in unique_countries:
        logger.info("[%s] Running LOCO fold: Held-out country = %s", representation_name, held_out)

        train_mask = (countries != held_out)
        test_mask = (countries == held_out)

        X_tr, y_tr, grp_tr = features[train_mask], labels[train_mask], site_ids[train_mask]
        X_te, y_te, grp_te = features[test_mask], labels[test_mask], site_ids[test_mask]

        # Document classes in test country vs training
        test_classes = sorted(np.unique(y_te).tolist())
        train_classes = sorted(np.unique(y_tr).tolist())

        # Fit probe on training countries
        probe = tune_and_train_linear_probe(
            X_tr,
            y_tr,
            grp_tr,
            c_grid=c_grid,
            inner_cv_splits=3,
            random_seed=random_seed,
            representation_name=f"{representation_name}_LOCO_{held_out}",
        )

        y_pred, y_prob = predict_probe(probe, X_te)

        # Macro-F1 computed on test set
        test_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))

        # Metrics
        eval_metrics = compute_comprehensive_metrics(y_te, y_pred, y_prob, probe.classes_)
        site_metrics = aggregate_site_level_predictions(y_te, y_prob, grp_te, probe.classes_)
        ci_stats = site_cluster_bootstrap_ci(y_te, y_pred, grp_te, n_iterations=1000, random_seed=random_seed)

        # Compute delta F1 if baseline supplied
        delta_f1 = (in_domain_baseline_f1 - test_f1) if in_domain_baseline_f1 is not None else None

        loco_results[held_out] = {
            "held_out_country": held_out,
            "train_countries": [c for c in unique_countries if c != held_out],
            "train_classes": train_classes,
            "test_classes": test_classes,
            "test_sites_count": len(np.unique(grp_te)),
            "test_samples_count": len(y_te),
            "macro_f1": test_f1,
            "delta_f1_drop": delta_f1,
            "segment_level_metrics": eval_metrics,
            "site_level_metrics": site_metrics,
            "cluster_bootstrap_ci": ci_stats,
            "best_c": probe.best_c,
        }

    # Summary table across countries
    summary_rows = []
    for c, res in loco_results.items():
        summary_rows.append({
            "held_out_country": c,
            "macro_f1": res["macro_f1"],
            "delta_f1_drop": res["delta_f1_drop"],
            "ci_lower": res["cluster_bootstrap_ci"]["ci_lower"],
            "ci_upper": res["cluster_bootstrap_ci"]["ci_upper"],
            "test_sites": res["test_sites_count"],
            "test_classes": ",".join(res["test_classes"]),
        })

    return {
        "representation": representation_name,
        "loco_folds": loco_results,
        "summary_table": summary_rows,
        "mean_loco_macro_f1": float(np.mean([r["macro_f1"] for r in summary_rows])),
    }
