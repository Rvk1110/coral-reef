"""
EcoSound-Audit — Hardware Degradation & Robustness Evaluation
=============================================================
Implements test-time hardware and channel degradations (models trained on clean data):
1. Sampling rate reduction: 16 kHz -> 8 kHz -> back to model's expected rate.
2. Bit-depth reduction: 16-bit PCM -> 8-bit quantization.
3. Additive Gaussian noise across multiple SNR levels (+20 dB, +10 dB, 0 dB, -10 dB).
4. Tracks Macro-F1 change and degradation curves across representations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import f1_score

from ecosound.training import TrainedProbe, predict_probe

logger = logging.getLogger("reef_pipeline")


def apply_downsampling_degradation(
    waveform_16k: np.ndarray,
    target_degraded_hz: int = 8000,
    orig_hz: int = 16000,
) -> np.ndarray:
    """
    Simulate bandwidth-limited low-cost hydrophone by downsampling to 8 kHz,
    then upsampling back to original 16 kHz.
    """
    # Downsample 16k -> 8k (up=1, down=2)
    down = resample_poly(waveform_16k, 1, 2)
    # Upsample 8k -> 16k (up=2, down=1)
    up = resample_poly(down, 2, 1).astype(np.float32)

    # Ensure length matches original exactly
    if len(up) < len(waveform_16k):
        up = np.pad(up, (0, len(waveform_16k) - len(up)), mode="constant")
    elif len(up) > len(waveform_16k):
        up = up[:len(waveform_16k)]
    return np.clip(up, -1.0, 1.0)


def apply_bit_depth_degradation(
    waveform_float32: np.ndarray,
    target_bits: int = 8,
) -> np.ndarray:
    """
    Simulate low-resolution ADC quantization to 8-bit precision.
    Uniform quantization with 256 discrete bins.
    """
    levels = 2 ** target_bits
    half_levels = levels / 2.0
    # Map [-1.0, 1.0] to integer steps [-128, 127]
    quantized_int = np.round(waveform_float32 * (half_levels - 1))
    quantized_int = np.clip(quantized_int, -half_levels, half_levels - 1)
    # Rescale back to float32 [-1.0, 1.0]
    dequantized = (quantized_int / (half_levels - 1)).astype(np.float32)
    return dequantized


def apply_additive_noise_degradation(
    waveform_float32: np.ndarray,
    snr_db: float,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Inject zero-mean additive Gaussian white noise at specified SNR (dB).
    SNR is defined as: SNR_dB = 10 * log10(P_signal / P_noise)
    """
    sig_power = float(np.mean(waveform_float32 ** 2))
    if sig_power <= 1e-12:
        return waveform_float32

    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)

    rng = np.random.RandomState(random_seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(waveform_float32)).astype(np.float32)

    noisy_waveform = waveform_float32 + noise
    return np.clip(noisy_waveform, -1.0, 1.0)


def evaluate_representation_robustness(
    probe: TrainedProbe,
    clean_features_test: np.ndarray,
    degraded_features_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate trained probe across multiple degraded test feature sets.
    Reports clean Macro-F1, degraded Macro-F1, and Delta F1 drop.
    """
    # Baseline clean performance
    y_pred_clean, _ = predict_probe(probe, clean_features_test)
    clean_f1 = float(f1_score(y_test, y_pred_clean, average="macro", zero_division=0))

    results = {
        "clean_baseline_macro_f1": clean_f1,
        "degradations": {},
    }

    for deg_name, X_deg in degraded_features_dict.items():
        y_pred_deg, _ = predict_probe(probe, X_deg)
        deg_f1 = float(f1_score(y_test, y_pred_deg, average="macro", zero_division=0))
        delta_f1 = clean_f1 - deg_f1
        retention = (deg_f1 / clean_f1 * 100.0) if clean_f1 > 0 else 0.0

        results["degradations"][deg_name] = {
            "macro_f1": deg_f1,
            "delta_f1_drop": delta_f1,
            "f1_retention_pct": round(retention, 2),
        }
        logger.info(
            "[%s] Robustness %s: Macro-F1 = %.4f (Delta = %+.4f, Retention = %.1f%%)",
            probe.representation_name, deg_name, deg_f1, delta_f1, retention
        )

    return results
