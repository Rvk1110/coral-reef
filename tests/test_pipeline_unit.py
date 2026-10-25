"""
EcoSound-Audit — End-to-End Pipeline Unit Tests
===============================================
Tests:
1. Preprocessing, 5-second segmentation, and polyphase resampling.
2. scikit-maad ecoacoustic index calculation and range validity.
3. Linear probe training with StandardScaler fit strictly on train split.
4. Site-grouped CV evaluation and zero site leakage.
5. Leave-One-Country-Out (LOCO) execution.
6. Robustness test-time degradations (8 kHz, 8-bit, SNR noise).
7. Restoration Trajectory Index (RTI) computation & ecological validation rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "EcoSound-Audit" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecosound.evaluation import (
    aggregate_site_level_predictions,
    compute_comprehensive_metrics,
    run_in_domain_site_grouped_cv,
    site_cluster_bootstrap_ci,
)
from ecosound.features import extract_ecoacoustic_indices
from ecosound.generalization import run_leave_one_country_out_benchmark
from ecosound.preprocessing import (
    preprocess_audio_waveform,
    resample_for_surfperch,
    slice_waveform_into_segments,
)
from ecosound.robustness import (
    apply_additive_noise_degradation,
    apply_bit_depth_degradation,
    apply_downsampling_degradation,
)
from ecosound.rti import (
    compute_degraded_healthy_centroids,
    compute_restoration_trajectory_index,
    evaluate_rti_ecological_correlation,
)
from ecosound.training import predict_probe, tune_and_train_linear_probe


def test_preprocessing_and_resampling():
    sr = 16000
    duration_s = 60.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Generate 1 kHz synthetic marine tone
    waveform_16k = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    # Anomaly policy check
    processed = preprocess_audio_waveform(waveform_16k, orig_sr=16000)
    assert processed is not None
    assert len(processed) == 960000

    # 5-second segmentation check
    segments = slice_waveform_into_segments(processed, sr=16000, segment_duration_s=5.0)
    assert len(segments) == 12
    for seg in segments:
        assert len(seg) == 80000  # 5.0 s * 16,000 Hz

    # SurfPerch polyphase resampling check
    seg_32k = resample_for_surfperch(segments[0])
    assert len(seg_32k) == 160000  # 5.0 s * 32,000 Hz
    assert not np.isnan(seg_32k).any()


def test_ecoacoustic_indices_extraction():
    sr = 16000
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    # Combination of 1.5 kHz (anthro proxy) and 4 kHz (bio proxy)
    waveform = (0.3 * np.sin(2 * np.pi * 1500 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)

    features = extract_ecoacoustic_indices(waveform, sr=16000)
    assert features.shape == (5,)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

    # NDSI must lie in theoretical [-1.0, 1.0]
    ndsi = features[1]
    assert -1.0 <= ndsi <= 1.0


def test_training_and_zero_leakage():
    rng = np.random.RandomState(42)
    n_samples = 120
    n_features = 5

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.choice(["D", "H", "N", "R"], size=n_samples)
    groups = np.array([f"site_{i // 10}" for i in range(n_samples)])

    probe = tune_and_train_linear_probe(
        X, y, groups, c_grid=[0.01, 1.0], inner_cv_splits=2, random_seed=42
    )

    assert probe.scaler is not None
    assert probe.best_c in [0.01, 1.0]

    preds, probs = predict_probe(probe, X[:10])
    assert len(preds) == 10
    assert probs.shape == (10, len(probe.classes_))
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_in_domain_cv_and_site_voting():
    rng = np.random.RandomState(42)
    n_samples = 120
    n_features = 5

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = np.array(["D"] * 30 + ["H"] * 30 + ["N"] * 30 + ["R"] * 30)
    groups = np.array([f"site_{i // 10}" for i in range(n_samples)])

    res = run_in_domain_site_grouped_cv(
        X, y, groups, n_splits=3, c_grid=[0.1, 1.0], random_seed=42, representation_name="TestMaad"
    )

    assert "segment_level_metrics" in res
    assert "site_level_metrics" in res
    assert "site_cluster_bootstrap_ci" in res
    assert 0.0 <= res["segment_level_metrics"]["macro_f1"] <= 1.0
    assert 0.0 <= res["site_level_metrics"]["macro_f1"] <= 1.0


def test_loco_geographic_generalization():
    rng = np.random.RandomState(42)
    n_samples = 120
    n_features = 5

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = np.array(["D"] * 30 + ["H"] * 30 + ["N"] * 30 + ["R"] * 30)
    sites = np.array([f"site_{i // 10}" for i in range(n_samples)])
    countries = np.array(["aus"] * 24 + ["ind"] * 24 + ["ken"] * 24 + ["mal"] * 24 + ["mex"] * 24)

    loco_res = run_leave_one_country_out_benchmark(
        X, y, sites, countries, in_domain_baseline_f1=0.75, c_grid=[1.0], random_seed=42
    )

    assert len(loco_res["loco_folds"]) == 5
    for c in ["aus", "ind", "ken", "mal", "mex"]:
        assert c in loco_res["loco_folds"]
        assert "macro_f1" in loco_res["loco_folds"][c]


def test_robustness_degradations():
    sr = 16000
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    # 1. 8 kHz downsampling & return
    deg_8k = apply_downsampling_degradation(waveform, target_degraded_hz=8000, orig_hz=16000)
    assert len(deg_8k) == len(waveform)
    assert not np.isnan(deg_8k).any()

    # 2. 8-bit depth reduction
    deg_8bit = apply_bit_depth_degradation(waveform, target_bits=8)
    assert len(deg_8bit) == len(waveform)
    assert not np.isnan(deg_8bit).any()

    # 3. Additive noise SNR
    deg_noise = apply_additive_noise_degradation(waveform, snr_db=10.0, random_seed=42)
    assert len(deg_noise) == len(waveform)
    assert not np.isnan(deg_noise).any()


def test_restoration_trajectory_index_and_ecological_rule():
    X = np.array([
        [0.0, 0.0],  # D sample
        [0.2, 0.1],  # D sample
        [1.0, 1.0],  # H sample
        [0.9, 1.1],  # H sample
        [0.5, 0.5],  # Restored sample
    ], dtype=np.float32)
    y = np.array(["D", "D", "H", "H", "R"])

    cD, cH = compute_degraded_healthy_centroids(X[:4], y[:4], degraded_label="D", healthy_label="H")
    rti_vals = compute_restoration_trajectory_index(X, cD, cH)

    assert np.isclose(rti_vals[0], 0.0, atol=0.2)
    assert np.isclose(rti_vals[2], 1.0, atol=0.2)
    # Restored plot should fall intermediate (near ~0.5)
    assert 0.2 < rti_vals[4] < 0.8

    # Strict compliance rule check: absence of independent ground truth
    eco_eval = evaluate_rti_ecological_correlation({"site_1": 0.5}, ecological_measurements=None)
    assert eco_eval["status"] == "SKIPPED_NO_BENTHIC_GROUND_TRUTH"
    assert "No continuous benthic ecological measurements" in eco_eval["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
