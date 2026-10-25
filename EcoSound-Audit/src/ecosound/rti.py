"""
EcoSound-Audit — Restoration Trajectory Index (RTI) & Ecological Validation
===========================================================================
Implements:
1. Mathematical definition of the Restoration Trajectory Index (RTI):
   Position of an audio embedding along the trajectory axis connecting
   the Degraded (cD) and Healthy (cH) class centroids (fit strictly on training data).
2. Site-level RTI aggregation.
3. Ecological validation check:
   Strict adherence to brief Section 4.4: if independent continuous ecological
   measurements (live coral cover %, fish biomass) are absent from the dataset,
   the correlation step is formally skipped with a cited audit finding,
   preventing fabricated proxies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("reef_pipeline")


def compute_degraded_healthy_centroids(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degraded_label: Any = "D",
    healthy_label: Any = "H",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Degraded (cD) and Healthy (cH) class centroids strictly from training data.
    """
    mask_d = (y_train == degraded_label)
    mask_h = (y_train == healthy_label)

    if not np.any(mask_d):
        raise ValueError(f"Degraded class '{degraded_label}' not present in training data")
    if not np.any(mask_h):
        raise ValueError(f"Healthy class '{healthy_label}' not present in training data")

    cD = np.mean(X_train[mask_d], axis=0)
    cH = np.mean(X_train[mask_h], axis=0)
    return cD, cH


def compute_restoration_trajectory_index(
    X: np.ndarray,
    cD: np.ndarray,
    cH: np.ndarray,
) -> np.ndarray:
    """
    Calculate the Restoration Trajectory Index (RTI) for arbitrary feature vectors:
    RTI(x) = ((x - cD) . (cH - cD)) / (||cH - cD||^2)

    Properties:
    - RTI(cD) = 0.0 (Degraded centroid baseline)
    - RTI(cH) = 1.0 (Healthy reference centroid)
    - Early/Mid restoration plots fall intermediate along the trajectory.
    """
    v = cH - cD
    v_norm_sq = float(np.dot(v, v))
    if v_norm_sq <= 1e-12:
        logger.warning("Degraded and Healthy centroids are identical; returning zero RTI")
        return np.zeros(len(X), dtype=np.float32)

    diff = X - cD
    projections = np.dot(diff, v) / v_norm_sq
    return projections.astype(np.float32)


def evaluate_rti_ecological_correlation(
    site_rti_estimates: Dict[str, float],
    ecological_measurements: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Correlate site-level RTI with independent continuous benthic measurements
    (e.g. coral cover %, fish counts) using Spearman rank correlation.

    Strict Rule (Brief Section 4.4):
    If no independent continuous measurements exist in the dataset, skip this step
    and state so clearly. Do not substitute synthetic proxies.
    """
    if not ecological_measurements:
        msg = (
            "No continuous benthic ecological measurements (e.g. live coral cover %, "
            "structural rugosity, fish biomass) exist in the Figshare dataset metadata "
            "(Williams & Jones 2025; DOI: 10.5522/04/29958062.v2). The dataset provides "
            "categorical habitat codes (D, H, N, R) only. Per Brief Section 4.4 hard rules, "
            "the correlation step is skipped without substituting synthetic or proxy data."
        )
        logger.info("[RTI Ecological Validation] %s", msg)
        return {
            "status": "SKIPPED_NO_BENTHIC_GROUND_TRUTH",
            "reason": msg,
            "spearman_rho": None,
            "p_value": None,
            "num_sites": len(site_rti_estimates),
            "site_rti_values": site_rti_estimates,
        }

    # If measurements are supplied:
    common_sites = sorted(set(site_rti_estimates.keys()) & set(ecological_measurements.keys()))
    if len(common_sites) < 5:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": f"Only {len(common_sites)} overlapping sites; minimum 5 required.",
            "spearman_rho": None,
            "p_value": None,
        }

    from scipy.stats import spearmanr
    r_vals = [site_rti_estimates[s] for s in common_sites]
    e_vals = [ecological_measurements[s] for s in common_sites]
    rho, pval = spearmanr(r_vals, e_vals)

    return {
        "status": "COMPLETED",
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "num_sites": len(common_sites),
        "common_sites": common_sites,
    }
