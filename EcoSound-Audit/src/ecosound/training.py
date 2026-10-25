"""
EcoSound-Audit — Standardized Probe Training & Hyperparameter Tuning
====================================================================
Implements:
1. Standardized Linear Probe (Logistic Regression L2) for all 3 representations.
2. StandardScaler fit strictly on training set data.
3. Inner Site-Grouped Cross-Validation on training folds to tune regularization C.
4. Balanced class weighting to prevent majority-class bias.
5. Deterministic random seeds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("reef_pipeline")

DEFAULT_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass
class TrainedProbe:
    model: LogisticRegression
    scaler: StandardScaler
    best_c: float
    classes_: np.ndarray
    feature_dim: int
    representation_name: str


def tune_and_train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    c_grid: Optional[List[float]] = None,
    inner_cv_splits: int = 3,
    random_seed: int = 42,
    representation_name: str = "Representation",
) -> TrainedProbe:
    """
    Standardize features and train an L2-regularized Logistic Regression probe.
    Tuning of C is performed strictly via inner site-grouped cross-validation
    on X_train / y_train.
    """
    if c_grid is None:
        c_grid = DEFAULT_C_GRID

    # Fit scaler strictly on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(f"Training set has fewer than 2 distinct classes: {unique_classes}")

    # Inner Site-Grouped CV to find best C
    best_c = 1.0
    best_inner_f1 = -1.0

    # Ensure inner CV splits do not exceed minimum class group count
    n_splits = min(inner_cv_splits, len(np.unique(groups_train)))
    if n_splits >= 2:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        
        for c_val in c_grid:
            fold_f1s = []
            try:
                for inner_tr_idx, inner_val_idx in sgkf.split(X_train_scaled, y_train, groups=groups_train):
                    X_tr_inner, y_tr_inner = X_train_scaled[inner_tr_idx], y_train[inner_tr_idx]
                    X_val_inner, y_val_inner = X_train_scaled[inner_val_idx], y_train[inner_val_idx]

                    # Scale inside fold
                    inner_scaler = StandardScaler()
                    X_tr_inner_s = inner_scaler.fit_transform(X_tr_inner)
                    X_val_inner_s = inner_scaler.transform(X_val_inner)

                    clf = LogisticRegression(
                        C=c_val,
                        solver="lbfgs",
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_seed,
                    )
                    clf.fit(X_tr_inner_s, y_tr_inner)
                    y_pred = clf.predict(X_val_inner_s)
                    f1 = f1_score(y_val_inner, y_pred, average="macro", zero_division=0)
                    fold_f1s.append(f1)

                mean_f1 = float(np.mean(fold_f1s))
            except Exception as e:
                logger.warning("Inner CV error for C=%s: %s; falling back to direct fit", c_val, e)
                mean_f1 = 0.0

            if mean_f1 > best_inner_f1:
                best_inner_f1 = mean_f1
                best_c = c_val
    else:
        best_c = 1.0

    logger.info(
        "[%s] Selected best regularization C=%.4f (inner CV Macro-F1: %.4f)",
        representation_name, best_c, best_inner_f1
    )

    # Final probe fit on full scaled training set with best C
    final_model = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_seed,
    )
    final_model.fit(X_train_scaled, y_train)

    return TrainedProbe(
        model=final_model,
        scaler=scaler,
        best_c=best_c,
        classes_=final_model.classes_,
        feature_dim=X_train.shape[1],
        representation_name=representation_name,
    )


def predict_probe(
    probe: TrainedProbe,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict class labels and probabilities for test samples using trained probe.
    Applies the frozen training scaler.
    """
    X_scaled = probe.scaler.transform(X)
    probs = probe.model.predict_proba(X_scaled)
    preds = probe.model.predict(X_scaled)
    return preds, probs
