"""
Tests for 3-Fold Site-Grouped Cross-Validation (EcoSound-Audit Phase 3.2 Gate)
=============================================================================
Verifies that:
1. The 25 training sites partition into exactly 3 folds.
2. Every fold contains all four required classes (D, H, N, R) in both train and validation splits.
3. There is strict ZERO site leakage across folds.
"""
from pathlib import Path
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_site_grouped_3fold_cross_validation_coverage_and_no_leakage():
    metadata_path = PROJECT_ROOT / "data" / "sampling" / "site_metadata.csv"
    assert metadata_path.exists(), f"site_metadata.csv not found at {metadata_path}"

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)

    # 1. Verify train site counts and class inventory
    assert len(train_df) == 25, f"Expected 25 training sites, found {len(train_df)}"
    class_counts = train_df["habitat_label"].value_counts().to_dict()
    assert class_counts == {"D": 8, "H": 8, "R": 6, "N": 3}, f"Unexpected train class counts: {class_counts}"

    # 2. Partition into 3 folds using StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(sgkf.split(train_df, train_df["habitat_label"], groups=train_df["site_id"]))

    assert len(splits) == 3, f"Expected 3 splits, got {len(splits)}"

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_train_sites = train_df.iloc[train_idx]
        fold_val_sites = train_df.iloc[val_idx]

        # Check zero site leakage
        train_sites_set = set(fold_train_sites["site_id"])
        val_sites_set = set(fold_val_sites["site_id"])
        overlap = train_sites_set.intersection(val_sites_set)
        assert len(overlap) == 0, f"Fold {fold_idx} has site leakage: {overlap}"

        # Check all 4 classes present in validation set
        val_classes = set(fold_val_sites["habitat_label"])
        assert val_classes == {"D", "H", "N", "R"}, (
            f"Fold {fold_idx} val is missing classes: {val_classes}"
        )

        # Check all 4 classes present in train set
        train_classes = set(fold_train_sites["habitat_label"])
        assert train_classes == {"D", "H", "N", "R"}, (
            f"Fold {fold_idx} train is missing classes: {train_classes}"
        )

        # Verify class N has exactly 1 site in val in each fold
        assert fold_val_sites["habitat_label"].value_counts().get("N", 0) == 1, (
            f"Fold {fold_idx} should have exactly 1 site for class N"
        )
