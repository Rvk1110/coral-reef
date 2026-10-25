"""
EcoSound-Audit — Audio Preprocessing and 5-Second Segmentation
==============================================================
Implements:
1. Manifest generation for 5-second non-overlapping audio segments.
2. Anomaly handling (duration < 50s dropped, 50-60s padded).
3. Dual-mode label mapping: 4-class native (D/H/N/R) and 3-class (Degraded/Healthy/Restoration).
4. Model-specific resampling: 16 kHz native (Maad & YAMNet) and 32 kHz polyphase (SurfPerch).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("reef_pipeline")

RECORDER_TYPE = "AudioMoth/HydroMoth"

LABEL_MAPPING_3CLASS = {
    "D": "Degraded",
    "H": "Healthy",
    "N": "Restoration",
    "R": "Restoration",
}

LABEL_NAMES_4CLASS = {
    "D": "Degraded",
    "H": "Healthy",
    "N": "Early-stage Restored (<3mo)",
    "R": "Mid-stage Restored (32-53mo)",
}


@dataclass(frozen=True)
class AudioSegmentMetadata:
    segment_id: str
    source_file: str
    source_archive: str
    segment_index: int
    start_time_s: float
    end_time_s: float
    country: str
    country_code: str
    site_id: str
    site_code: str
    habitat_label: str       # Native 4-class: D, H, N, R
    label_3class: str        # 3-class: Degraded, Healthy, Restoration
    split: str               # train, val, test
    recorder_type: str = RECORDER_TYPE
    sample_rate_native_hz: int = 16000
    duration_s: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_segment_manifest(
    sampled_recordings_csv: Path,
    output_segment_manifest_csv: Path,
    segment_duration_s: float = 5.0,
    nominal_duration_s: float = 60.0,
) -> List[AudioSegmentMetadata]:
    """
    Generate the master segment manifest from sampled recordings.
    Each 60.0s recording yields 12 non-overlapping 5.0s segments.
    """
    segments: List[AudioSegmentMetadata] = []
    num_segments_per_file = int(nominal_duration_s // segment_duration_s)

    with open(sampled_recordings_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            stem = Path(filename).stem
            archive = row.get("source_archive", "")
            country = row["country"]
            country_code = row["country_code"]
            site_id = row["site_id"]
            site_code = row["site_code"]
            hab_label = row["habitat_label"]
            split = row.get("split", "train")
            hab_3class = LABEL_MAPPING_3CLASS.get(hab_label, "Unknown")

            for seg_idx in range(num_segments_per_file):
                t_start = seg_idx * segment_duration_s
                t_end = (seg_idx + 1) * segment_duration_s
                seg_id = f"{stem}_seg{seg_idx:02d}"

                seg_meta = AudioSegmentMetadata(
                    segment_id=seg_id,
                    source_file=filename,
                    source_archive=archive,
                    segment_index=seg_idx,
                    start_time_s=t_start,
                    end_time_s=t_end,
                    country=country,
                    country_code=country_code,
                    site_id=site_id,
                    site_code=site_code,
                    habitat_label=hab_label,
                    label_3class=hab_3class,
                    split=split,
                    recorder_type=RECORDER_TYPE,
                    sample_rate_native_hz=16000,
                    duration_s=segment_duration_s,
                )
                segments.append(seg_meta)

    # Export to CSV
    output_segment_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    if segments:
        fieldnames = list(segments[0].to_dict().keys())
        with open(output_segment_manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in segments:
                writer.writerow(s.to_dict())

    logger.info(
        "Generated segment manifest: %d segments from %s -> %s",
        len(segments), sampled_recordings_csv, output_segment_manifest_csv
    )
    return segments


def preprocess_audio_waveform(
    waveform: np.ndarray,
    orig_sr: int = 16000,
    target_duration_s: float = 60.0,
    min_duration_s: float = 50.0,
) -> Optional[np.ndarray]:
    """
    Apply standardized anomaly policy to audio waveform:
    - If length < min_duration_s (50.0s), discard (returns None).
    - If min_duration_s <= length < target_duration_s, zero-pad with silence to 60.0s.
    - If length > target_duration_s, trim to exactly 60.0s.
    - Ensure mono float32 normalized in [-1.0, 1.0].
    """
    if waveform.ndim > 1:
        # Convert multi-channel to mono
        waveform = np.mean(waveform, axis=-1)

    # Normalize dtype
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

    actual_duration = len(waveform) / float(orig_sr)
    target_samples = int(target_duration_s * orig_sr)
    min_samples = int(min_duration_s * orig_sr)

    if len(waveform) < min_samples:
        logger.warning(
            "Audio duration %.2fs is below minimum threshold %.2fs. Dropping file.",
            actual_duration, min_duration_s
        )
        return None

    if len(waveform) < target_samples:
        pad_width = target_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    elif len(waveform) > target_samples:
        waveform = waveform[:target_samples]

    # Clip to valid amplitude range
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform


def slice_waveform_into_segments(
    waveform: np.ndarray,
    sr: int = 16000,
    segment_duration_s: float = 5.0,
) -> List[np.ndarray]:
    """
    Slice 60-second mono waveform into non-overlapping 5-second chunks.
    For sr=16000, each chunk is 80,000 samples.
    """
    samples_per_seg = int(sr * segment_duration_s)
    num_segs = len(waveform) // samples_per_seg
    chunks = [
        waveform[i * samples_per_seg : (i + 1) * samples_per_seg]
        for i in range(num_segs)
    ]
    return chunks


def resample_for_surfperch(
    audio_16k_segment: np.ndarray,
) -> np.ndarray:
    """
    Polyphase resample 16 kHz audio segment to 32 kHz (up=2, down=1).
    Yields exactly 160,000 samples for a 5.0-second segment.
    """
    try:
        from scipy.signal import resample_poly
        resampled = resample_poly(audio_16k_segment, 2, 1).astype(np.float32)
        target_len = 32000 * 5  # 160,000
        if len(resampled) > target_len:
            resampled = resampled[:target_len]
        elif len(resampled) < target_len:
            resampled = np.pad(resampled, (0, target_len - len(resampled)), mode="constant")
        return resampled
    except ImportError:
        # Fallback linear interpolation
        x_orig = np.linspace(0, 1, len(audio_16k_segment))
        x_new = np.linspace(0, 1, len(audio_16k_segment) * 2)
        return np.interp(x_new, x_orig, audio_16k_segment).astype(np.float32)
