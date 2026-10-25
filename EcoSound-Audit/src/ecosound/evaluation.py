"""
EcoSound-Audit — Standardized Evaluation Protocols
===================================================
Implements:
1. In-Domain Site-Grouped Cross-Validation (StratifiedGroupKFold on site_id).
2. Standardized Metrics:
   - Macro-F1
   - Balanced Accuracy
   - AUROC (one-vs-rest, macro)
   - Per-class Precision, Recall, F1
   - Normalized Confusion Matrix
3. Site-Level Diagnostic Aggregation with prior tie-breaking.
4. Site-Level Cluster Bootstrap 95% Confidence Intervals.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from ecosound.training import TrainedProbe, predict_probe, tune_and_train_linear_probe

logger = logging.getLogger("reef_pipeline")


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute Macro-F1, Balanced Accuracy, AUROC (OvR macro),
    per-class precision/recall/F1, and confusion matrix.
    """
    classes_list = list(classes)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Per-class precision, recall, f1, support
    prec, rec, f1s, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class = {}
    for idx, c in enumerate(classes):
        per_class[str(c)] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1s[idx]),
            "support": int(sup[idx]),
        }

    # AUROC calculation
    auroc = None
    present_classes = np.unique(y_true)
    if len(present_classes) >= 2 and y_prob is not None:
        try:
            if len(classes) == 2:
                # Binary AUROC
                pos_idx = 1
                auroc = float(roc_auc_score(y_true == classes[pos_idx], y_prob[:, pos_idx]))
            else:
                # Multiclass OvR macro
                auroc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                        labels=classes,
                    )
                )
        except Exception as e:
            logger.debug("Could not compute AUROC (likely missing classes in test fold): %s", e)
            auroc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "auroc_ovr_macro": auroc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.nan_to_num(cm_norm).tolist(),
        "classes": [str(c) for c in classes],
        "sample_count": len(y_true),
    }


def aggregate_site_level_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    site_ids: np.ndarray,
    classes: np.ndarray,
    class_priors: Optional[Dict[Any, float]] = None,
) -> Dict[str, Any]:
    """
    Aggregate segment-level predictions to site-level votes:
    Averages predicted probabilities across all segments belonging to each site.
    Resolves ties using class training priors.
    """
    unique_sites = np.unique(site_ids)
    site_true = []
    site_preds = []
    site_probs = []

    for s in unique_sites:
        mask = (site_ids == s)
        s_prob = np.mean(y_prob[mask], axis=0)
        s_true = y_true[mask][0]

        # Argmax with tie break
        max_val = np.max(s_prob)
        best_indices = np.where(np.isclose(s_prob, max_val))[0]
        if len(best_indices) == 1:
            best_idx = best_indices[0]
        else:
            # Tie breaker: highest prior
            if class_priors:
                best_idx = max(best_indices, key=lambda idx: class_priors.get(classes[idx], 0.0))
            else:
                best_idx = best_indices[0]

        site_true.append(s_true)
        site_preds.append(classes[best_idx])
        site_probs.append(s_prob)

    site_true_arr = np.array(site_true)
    site_pred_arr = np.array(site_preds)
    site_prob_arr = np.array(site_probs)

    metrics = compute_comprehensive_metrics(site_true_arr, site_pred_arr, site_prob_arr, classes)
    metrics["num_sites"] = len(unique_sites)
    return metrics


def site_cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    site_ids: np.ndarray,
    n_iterations: int = 1000,
    random_seed: int = 42,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compute cluster bootstrap confidence intervals resampling entire sites (clusters),
    preventing artificial deflation of standard error.
    """
    unique_sites = np.unique(site_ids)
    n_sites = len(unique_sites)
    if n_sites < 2:
        return {"ci_lower": 0.0, "ci_upper": 0.0, "mean": 0.0}

    rng = np.random.RandomState(random_seed)
    boot_f1s = []

    site_to_indices = {s: np.where(site_ids == s)[0] for s in unique_sites}

    for _ in range(n_iterations):
        sampled_sites = rng.choice(unique_sites, size=n_sites, replace=True)
        sampled_indices = np.concatenate([site_to_indices[s] for s in sampled_sites])
        b_true = y_true[sampled_indices]
        b_pred = y_pred[sampled_indices]
        f1 = f1_score(b_true, b_pred, average="macro", zero_division=0)
        boot_f1s.append(f1)

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_f1s, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_f1s, 100 * (1 - alpha / 2)))
    mean_val = float(np.mean(boot_f1s))

    return {
        "ci_lower": lower,
        "ci_upper": upper,
        "mean": mean_val,
        "std": float(np.std(boot_f1s)),
    }


def run_in_domain_site_grouped_cv(
    features: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    n_splits: int = 3,
    c_grid: Optional[List[float]] = None,
    random_seed: int = 42,
    representation_name: str = "Representation",
) -> Dict[str, Any]:
    """
    Run 3-fold Site-Grouped Cross-Validation.
    All segments from any physical site remain strictly in the same fold.
    """
    unique_sites = np.unique(site_ids)
    n_splits_actual = min(n_splits, len(unique_sites))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_seed)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_sites = []
    fold_summaries = []

    classes = np.unique(labels)
    # Estimate class priors
    priors = {c: float(np.mean(labels == c)) for c in classes}

    for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(features, labels, groups=site_ids)):
        X_tr, y_tr, grp_tr = features[tr_idx], labels[tr_idx], site_ids[tr_idx]
        X_val, y_val, grp_val = features[val_idx], labels[val_idx], site_ids[val_idx]

        probe = tune_and_train_linear_probe(
            X_tr,
            y_tr,
            grp_tr,
            c_grid=c_grid,
            inner_cv_splits=2,
            random_seed=random_seed + fold_idx,
            representation_name=f"{representation_name}_fold{fold_idx+1}",
        )

        y_p, y_pb = predict_probe(probe, X_val)

        # Align probability columns if needed
        aligned_probs = np.zeros((len(y_val), len(classes)), dtype=np.float32)
        for i, c in enumerate(probe.classes_):
            if c in classes:
                target_idx = np.where(classes == c)[0][0]
                aligned_probs[:, target_idx] = y_pb[:, i]

        all_y_true.extend(y_val)
        all_y_pred.extend(y_p)
        all_y_prob.extend(aligned_probs)
        all_sites.extend(grp_val)

        fold_f1 = f1_score(y_val, y_p, average="macro", zero_division=0)
        fold_summaries.append({
            "fold": fold_idx + 1,
            "train_sites": sorted(np.unique(grp_tr).tolist()),
            "val_sites": sorted(np.unique(grp_val).tolist()),
            "macro_f1": float(fold_f1),
            "best_c": probe.best_c,
        })

    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    y_prob_all = np.array(all_y_prob)
    sites_all = np.array(all_sites)

    segment_metrics = compute_comprehensive_metrics(y_true_all, y_pred_all, y_prob_all, classes)
    site_metrics = aggregate_site_level_predictions(y_true_all, y_prob_all, sites_all, classes, priors)
    bootstrap_ci = site_cluster_bootstrap_ci(y_true_all, y_pred_all, sites_all, n_iterations=1000, random_seed=random_seed)

    return {
        "representation": representation_name,
        "segment_level_metrics": segment_metrics,
        "site_level_metrics": site_metrics,
        "site_cluster_bootstrap_ci": bootstrap_ci,
        "folds": fold_summaries,
    }
