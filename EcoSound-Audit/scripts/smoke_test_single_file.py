"""
EcoSound-Audit — Phase 3.2 Single-File Smoke Test
=================================================
Validates Gate 1 to Gate 5 on a single 1-minute audio recording:
1. Streams 1 real WAV recording from Figshare remote archive via HTTP Range.
2. Decompresses raw 16,000 Hz, 1-ch, 16-bit PCM WAV in memory without saving to disk.
3. Extracts and verifies Pipeline A: Traditional Ecoacoustic Indices (scikit-maad):
   - Confirms mode='psd' in sound.spectrogram
   - Confirms exact parameter syntax: flim_bioPh=(2000, 8000), flim_antroPh=(1000, 2000)
   - Unpacks ACI, NDSI, ADI, BI, AEI into feature vector of shape (5,)
4. Reports output shapes, numerical validity (no NaNs/Infs), extraction latency, and NDSI bounds.
"""
from __future__ import annotations

import io
import json
import logging
import struct
import time
import wave
import zlib
from pathlib import Path
from typing import Dict, Any
from urllib.request import Request, urlopen

import numpy as np
from maad import sound, features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ecosound_smoke_test")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def stream_single_wav_from_figshare(
    file_id: int = 57335960,  # ken_H1.zip
    archive_name: str = "ken_H1.zip",
    user_agent: str = "EcoSoundAudit/1.0",
) -> Dict[str, Any]:
    """Stream a single 1-minute WAV from Figshare archive via HTTP Range requests."""
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    t0 = time.time()

    # 1. HEAD request
    req = Request(url, method="HEAD", headers={"User-Agent": user_agent})
    with urlopen(req, timeout=30) as resp:
        content_len = resp.headers.get("Content-Length")
        if not content_len:
            raise RuntimeError(f"Could not get Content-Length for {archive_name}")
        file_size = int(content_len)

    # 2. EOCD fetch
    tail_req = Request(
        url,
        headers={"Range": f"bytes={file_size - 65536}-{file_size - 1}", "User-Agent": user_agent},
    )
    with urlopen(tail_req, timeout=30) as resp:
        tail = resp.read()

    eocd_pos = tail.rfind(b"PK\x05\x06")
    cd_entries = struct.unpack_from("<H", tail, eocd_pos + 10)[0]
    cd_size = struct.unpack_from("<I", tail, eocd_pos + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, eocd_pos + 16)[0]

    # 3. CD fetch
    cd_req = Request(
        url,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}", "User-Agent": user_agent},
    )
    with urlopen(cd_req, timeout=30) as resp:
        cd_data = resp.read()

    # 4. Parse first WAV entry
    pos = 0
    target_entry = None
    while pos + 46 <= len(cd_data):
        if cd_data[pos:pos + 4] != b"PK\x01\x02":
            break
        comp_size = struct.unpack_from("<I", cd_data, pos + 20)[0]
        uncomp_size = struct.unpack_from("<I", cd_data, pos + 24)[0]
        fname_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
        extra_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
        comment_len = struct.unpack_from("<H", cd_data, pos + 32)[0]
        local_offset = struct.unpack_from("<I", cd_data, pos + 42)[0]
        compression = struct.unpack_from("<H", cd_data, pos + 10)[0]
        fname = cd_data[pos + 46:pos + 46 + fname_len].decode("utf-8", errors="replace")
        if fname.lower().endswith(".wav"):
            target_entry = {
                "filename": fname,
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "local_offset": local_offset,
                "compression": compression,
            }
            break
        pos += 46 + fname_len + extra_len + comment_len

    if not target_entry:
        raise RuntimeError("No WAV file found in Central Directory")

    # 5. Fetch local file header to compute exact data start
    local_off = target_entry["local_offset"]
    lh_req = Request(url, headers={"Range": f"bytes={local_off}-{local_off + 100}", "User-Agent": user_agent})
    with urlopen(lh_req, timeout=30) as resp:
        lh_buf = resp.read()

    lh_fname_len = struct.unpack_from("<H", lh_buf, 26)[0]
    lh_extra_len = struct.unpack_from("<H", lh_buf, 28)[0]
    data_start = 30 + lh_fname_len + lh_extra_len

    # 6. Fetch compressed audio payload
    comp_size = target_entry["comp_size"]
    comp_req = Request(
        url,
        headers={"Range": f"bytes={local_off + data_start}-{local_off + data_start + comp_size - 1}", "User-Agent": user_agent},
    )
    with urlopen(comp_req, timeout=30) as resp:
        comp_bytes = resp.read()

    t_fetch = time.time() - t0

    # 7. Decompress raw WAV
    raw_wav = zlib.decompress(comp_bytes, -15)

    # 8. Parse WAV
    with wave.open(io.BytesIO(raw_wav), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        pcm_data = wf.readframes(n_frames)

    duration_s = n_frames / sr
    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

    return {
        "filename": target_entry["filename"],
        "archive_name": archive_name,
        "compressed_bytes": len(comp_bytes),
        "uncompressed_bytes": len(raw_wav),
        "channels": channels,
        "sample_rate_hz": sr,
        "n_frames": n_frames,
        "duration_s": duration_s,
        "waveform": audio,
        "io_time_s": round(t_fetch, 3),
    }


def run_smoke_test() -> Dict[str, Any]:
    """Execute the single-file smoke test and verify ecoacoustic indices."""
    logger.info("Starting Phase 3.2 Single-File Smoke Test...")

    # Stream 1 pilot recording
    audio_info = stream_single_wav_from_figshare()
    audio = audio_info["waveform"]
    sr = audio_info["sample_rate_hz"]

    logger.info("Streamed %s (%d bytes, duration=%.2fs) in %.2fs",
                audio_info["filename"], audio_info["compressed_bytes"], audio_info["duration_s"], audio_info["io_time_s"])

    t_feat0 = time.time()

    # 1. Power spectrogram: mode='psd' is the verified correct parameter in scikit-maad
    Sxx_power, tn, fn, ext = sound.spectrogram(audio, sr, nperseg=1024, noverlap=512, mode="psd")

    # 2. ACI: returns (aci_temporal, aci_per_bin, aci_sum)
    aci_temporal, aci_per_bin, aci_sum = features.acoustic_complexity_index(Sxx_power)

    # 3. NDSI: parameter names are flim_bioPh and flim_antroPh
    # Anthrophony band: 1000 - 2000 Hz, Biophony band: 2000 - 8000 Hz (under 8 kHz Nyquist)
    NDSI, ratioBA, anthroPh, bioPh = features.soundscape_index(
        Sxx_power,
        fn,
        flim_bioPh=(2000, 8000),
        flim_antroPh=(1000, 2000),
        R_compatible="soundecology",
    )

    # 4. ADI: Acoustic Diversity Index (Shannon diversity)
    ADI = features.acoustic_diversity_index(Sxx_power, fn, fmin=0, fmax=8000, bin_step=1000, index="shannon")

    # 5. BI: Bioacoustic Index
    BI = features.bioacoustics_index(Sxx_power, fn, flim=(2000, 8000), R_compatible="soundecology")

    # 6. AEI: Acoustic Evenness Index (spelled acoustic_eveness_index)
    AEI = features.acoustic_eveness_index(Sxx_power, fn, fmin=0, fmax=8000, bin_step=1000)

    t_feat = time.time() - t_feat0

    x_maad = np.array([aci_sum, NDSI, ADI, BI, AEI], dtype=np.float32)

    # Validations
    assert x_maad.shape == (5,), f"Expected shape (5,), got {x_maad.shape}"
    assert not np.isnan(x_maad).any(), f"NaNs detected in feature vector: {x_maad}"
    assert not np.isinf(x_maad).any(), f"Infs detected in feature vector: {x_maad}"
    assert -1.0 <= NDSI <= 1.0, f"NDSI value {NDSI} out of theoretical range [-1.0, 1.0]"

    summary = {
        "status": "PASSED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_file": audio_info["filename"],
        "archive": audio_info["archive_name"],
        "audio_specs": {
            "channels": audio_info["channels"],
            "sample_rate_hz": audio_info["sample_rate_hz"],
            "duration_s": audio_info["duration_s"],
            "compressed_bytes_fetched": audio_info["compressed_bytes"],
            "decompressed_pcm_bytes": audio_info["uncompressed_bytes"],
        },
        "pipeline_a_maad": {
            "status": "PASSED",
            "feature_vector_shape": list(x_maad.shape),
            "feature_values": {
                "ACI": float(round(aci_sum, 4)),
                "NDSI": float(round(NDSI, 4)),
                "ADI": float(round(ADI, 4)),
                "BI": float(round(BI, 4)),
                "AEI": float(round(AEI, 4)),
            },
            "ndsi_verification": {
                "verified_parameter_names": {
                    "biophony": "flim_bioPh=(2000, 8000)",
                    "anthrophony": "flim_antroPh=(1000, 2000)",
                    "r_compatible": "soundecology",
                },
                "biophony_energy": float(f"{bioPh:.4e}"),
                "anthrophony_energy": float(f"{anthroPh:.4e}"),
                "ratio_ba": float(round(ratioBA, 4)),
                "ndsi_value": float(round(NDSI, 4)),
                "within_theoretical_bounds": bool(-1.0 <= NDSI <= 1.0),
            },
            "spectrogram_mode_verified": "psd",
            "extraction_latency_seconds": round(t_feat, 4),
            "zero_nans_infs": True,
        },
        "total_remote_io_and_extraction_seconds": round(audio_info["io_time_s"] + t_feat, 3),
    }

    # Save summary report
    report_path = PROJECT_ROOT / "reports" / "smoke_test_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Smoke test summary saved to %s", report_path)
    return summary


if __name__ == "__main__":
    res = run_smoke_test()
    print(json.dumps(res, indent=2))
