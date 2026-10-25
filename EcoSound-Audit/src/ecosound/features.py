"""
EcoSound-Audit — Standardized Feature Extractors
================================================
Implements:
1. Pipeline A: Traditional Ecoacoustic Indices (scikit-maad) -> R^5
   - ACI, NDSI (marine adapted: 1-2 kHz anthrophony, 2-8 kHz biophony), ADI, BI, AEI.
2. Pipeline B: YAMNet Foundation Embeddings (TF Hub) -> R^1024
   - 16 kHz mono input, temporal mean pooling over 5-second segment.
3. Pipeline C: SurfPerch Marine Bioacoustic Embeddings (Kaggle Models) -> R^1280
   - 32 kHz mono input (160,000 samples), domain-adapted EfficientNet-B0 backbone.
4. Persistent Caching Layer: Saves/loads matrices to/from disk (.npz / .parquet).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("reef_pipeline")

ECOACOUSTIC_FEATURE_NAMES = ["ACI", "NDSI", "ADI", "BI", "AEI"]


def extract_ecoacoustic_indices(
    audio_segment_16k: np.ndarray,
    sr: int = 16000,
    nperseg: int = 1024,
    noverlap: int = 512,
    flim_antroPh: Tuple[int, int] = (1000, 2000),
    flim_bioPh: Tuple[int, int] = (2000, 8000),
    fmax: int = 8000,
    bin_step: int = 1000,
) -> np.ndarray:
    """
    Extract 5 traditional ecoacoustic indices for a single 5-second segment.
    Uses verified scikit-maad syntax with marine-adapted frequency bands.
    """
    try:
        import maad
        from maad import features, sound

        # Compute power spectrogram
        Sxx_power, tn, fn, ext = sound.spectrogram(
            audio_segment_16k,
            sr,
            nperseg=nperseg,
            noverlap=noverlap,
            mode="psd",
        )

        # 1. ACI
        _, _, aci_sum = features.acoustic_complexity_index(Sxx_power)

        # 2. NDSI (Soundscape Index)
        NDSI, _, _, _ = features.soundscape_index(
            Sxx_power,
            fn,
            flim_bioPh=flim_bioPh,
            flim_antroPh=flim_antroPh,
            R_compatible="soundecology",
        )

        # 3. ADI
        ADI = features.acoustic_diversity_index(
            Sxx_power,
            fn,
            fmin=0,
            fmax=fmax,
            bin_step=bin_step,
            index="shannon",
        )

        # 4. BI
        BI = features.bioacoustics_index(
            Sxx_power,
            fn,
            flim=flim_bioPh,
            R_compatible="soundecology",
        )

        # 5. AEI
        AEI = features.acoustic_eveness_index(
            Sxx_power,
            fn,
            fmin=0,
            fmax=fmax,
            bin_step=bin_step,
        )

        vec = np.array([aci_sum, NDSI, ADI, BI, AEI], dtype=np.float32)

    except ImportError:
        logger.warning("scikit-maad not installed, computing standard spectral acoustic indices via scipy")
        from scipy import signal

        freqs, psd = signal.welch(audio_segment_16k, fs=sr, nperseg=nperseg)
        # 1. Spectral Complexity (std/mean across frequency)
        aci_sim = float(np.std(psd) / (np.mean(psd) + 1e-12))

        # 2. NDSI: bio (2-8kHz) vs anthro (1-2kHz)
        mask_antro = (freqs >= flim_antroPh[0]) & (freqs < flim_antroPh[1])
        mask_bio = (freqs >= flim_bioPh[0]) & (freqs <= flim_bioPh[1])
        e_antro = float(np.sum(psd[mask_antro]))
        e_bio = float(np.sum(psd[mask_bio]))
        denom = e_bio + e_antro + 1e-12
        ndsi_sim = float((e_bio - e_antro) / denom)

        # 3. ADI: Shannon diversity across 1kHz bins
        bin_energies = []
        for b_start in range(0, fmax, bin_step):
            b_mask = (freqs >= b_start) & (freqs < b_start + bin_step)
            bin_energies.append(float(np.sum(psd[b_mask])))
        total_e = sum(bin_energies) + 1e-12
        p_bins = [e / total_e for e in bin_energies if e > 0]
        adi_sim = float(-sum(p * np.log(p) for p in p_bins))

        # 4. BI: Log energy in bio band
        bi_sim = float(np.sum(np.log10(psd[mask_bio] + 1e-12)))

        # 5. AEI: Gini evenness across bins
        p_sorted = sorted(p_bins)
        n = len(p_sorted)
        if n > 0:
            gini = 2 * sum((i + 1) * p for i, p in enumerate(p_sorted)) / (n * sum(p_sorted)) - (n + 1) / n
            aei_sim = float(gini)
        else:
            aei_sim = 0.0

        vec = np.array([aci_sim, ndsi_sim, adi_sim, bi_sim, aei_sim], dtype=np.float32)

    # Sanity checks: NaN / Inf replacement with 0.0
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
    return vec


class YAMNetFeatureExtractor:
    """Wrapper for TF Hub YAMNet model."""

    def __init__(self, hub_url: str = "https://tfhub.dev/google/yamnet/1"):
        self.hub_url = hub_url
        self.model = None

    def load(self):
        if self.model is None:
            import tensorflow_hub as hub
            logger.info("Loading YAMNet model from %s...", self.hub_url)
            self.model = hub.load(self.hub_url)
            logger.info("YAMNet model loaded successfully.")

    def extract(self, audio_segment_16k: np.ndarray) -> np.ndarray:
        """
        Extract 1024-d mean-pooled embedding for a 5.0-second 16 kHz segment.
        """
        if self.model is None:
            self.load()

        import tensorflow as tf

        waveform_tf = tf.convert_to_tensor(audio_segment_16k, dtype=tf.float32)
        scores, embeddings, spectrogram = self.model(waveform_tf)
        pooled = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)

        pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled


class SurfPerchFeatureExtractor:
    """Wrapper for Kaggle Models SurfPerch model."""

    def __init__(self, model_handle: str = "google/surfperch/tensorFlow2/default"):
        self.model_handle = model_handle
        self.model = None

    def load(self):
        if self.model is None:
            import kagglehub
            import tensorflow_hub as hub

            logger.info("Downloading/loading SurfPerch from Kaggle (%s)...", self.model_handle)
            model_path = kagglehub.model_download(self.model_handle)
            self.model = hub.load(model_path)
            logger.info("SurfPerch model loaded successfully from %s", model_path)

    def extract(self, audio_segment_32k: np.ndarray) -> np.ndarray:
        """
        Extract 1280-d embedding for a 5.0-second 32 kHz segment (160,000 samples).
        """
        if self.model is None:
            self.load()

        import numpy as np

        # Ensure exact shape (1, 160000)
        target_len = 160000
        if len(audio_segment_32k) < target_len:
            audio_segment_32k = np.pad(
                audio_segment_32k, (0, target_len - len(audio_segment_32k)), mode="constant"
            )
        elif len(audio_segment_32k) > target_len:
            audio_segment_32k = audio_segment_32k[:target_len]

        batch = np.expand_dims(audio_segment_32k.astype(np.float32), axis=0)
        outputs = self.model(batch)

        if isinstance(outputs, dict):
            emb_key = "embedding" if "embedding" in outputs else list(outputs.keys())[0]
            emb = outputs[emb_key].numpy()
        else:
            emb = outputs.numpy()

        # Squeeze batch dimension -> (1280,)
        vector = np.squeeze(emb).astype(np.float32)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        return vector


def save_feature_cache(
    features_matrix: np.ndarray,
    segment_ids: List[str],
    representation_name: str,
    cache_dir: Path,
) -> Path:
    """
    Save extracted feature matrix and segment IDs to disk in compressed NPZ format.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / f"features_{representation_name.lower()}.npz"
    np.savez_compressed(
        out_file,
        features=features_matrix,
        segment_ids=np.array(segment_ids),
        representation=representation_name,
    )
    logger.info("Cached %s features to %s (shape: %s)", representation_name, out_file, features_matrix.shape)
    return out_file


def load_feature_cache(
    cache_dir: Path,
    representation_name: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load cached feature matrix and segment IDs from disk.
    """
    cache_file = cache_dir / f"features_{representation_name.lower()}.npz"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cached features not found at {cache_file}")

    data = np.load(cache_file, allow_pickle=True)
    return data["features"], data["segment_ids"].tolist()
