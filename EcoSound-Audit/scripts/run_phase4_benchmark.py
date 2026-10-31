#!/usr/bin/env python3
"""
EcoSound-Audit — Phase 4: Full Benchmark Dataset Preparation
=============================================================
Generates the full 45-site benchmark sampling manifests using:
  - 100 recordings per site (Option A, TARGET_PER_SITE=100)
  - Frozen site-level train/val/test split from data/sampling/site_metadata.csv
  - 2-D temporal stratification (date × hour-of-day)
  - hourly_binning for Indonesia's 1-in-2 min duty cycle
  - 50 s minimum duration filter
  - 12 segments per recording (5 s each)

Outputs (all under data/sampling_full/ and reports/full/):
  - data/sampling_full/sampled_recordings.csv    — recording manifest
  - data/sampling_full/segment_manifest.csv      — segment manifest
  - data/sampling_full/shortfall_log.csv         — sites with < 100 eligible files
  - reports/full/sampling_summary.json
  - reports/full/benchmark_summary.json
  - reports/full/benchmark_summary.md

Pilot files are NEVER touched.

Usage:
------
    python scripts/run_phase4_benchmark.py [--config config/sampling_config_full.yaml]
    python scripts/run_phase4_benchmark.py --config config/sampling_config_full.yaml --max-archives 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecosound.models import RecordingMetadata, parse_recording_filename
from ecosound.remote_reader import fetch_archive_filenames
from ecosound.sampler import sample_recordings_uniform_temporal


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ecosound_phase4")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config & manifest loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest_csv(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load site archive inventory (manifest.csv) — skips detection archives."""
    sites = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zip_name = row["zip_name"].strip()
            if zip_name.startswith("detections_"):
                continue
            site_code = row["site_code"].strip()
            country = row["country"].strip()
            habitat_label = site_code[0]
            site_id = f"{country}_{site_code}"
            sites.append({
                "zip_name": zip_name,
                "country": country,
                "site_code": site_code,
                "site_id": site_id,
                "habitat_label": habitat_label,
                "file_count": int(row["file_count"]),
                "total_bytes": int(row["total_bytes"]),
                "sha256": row["sha256"].strip(),
            })
    return sites


def load_frozen_site_splits(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load the frozen site-level split from site_metadata.csv.
    Returns: {site_id: {"split": str, "country": str, "habitat_label": str, ...}}
    """
    splits = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            site_id = row["site_id"].strip()
            splits[site_id] = {
                "split": row["split"].strip(),
                "country": row["country"].strip(),
                "site_code": row["site_code"].strip(),
                "habitat_label": row["habitat_label"].strip(),
                "file_count": int(row["file_count"]),
                "sha256": row["sha256"].strip(),
                "zip_name": row["zip_name"].strip(),
            }
    return splits


def load_figshare_file_ids(json_path: Path) -> Dict[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["name"]: item["id"] for item in items}


def get_git_commit_hash() -> str:
    """Return the current HEAD git commit hash (short), or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Duration inference from filename
# ---------------------------------------------------------------------------

MIN_DURATION_S = 50.0


def infer_nominal_duration(rec: RecordingMetadata, known_short_sites: Optional[Dict[str, float]] = None) -> float:
    """
    Return inferred nominal duration for a recording.
    Known-short sites (e.g. ind_N1 with 25.6s files) can be supplied as a dict.
    For the standard dataset, all nominal durations are 60 s.
    The 50 s filter is applied AFTER this — short files are excluded at the
    sampling stage (we only have filenames from CD listings, not actual durations).

    NOTE: Without actual WAV headers, we cannot determine true duration from
    the ZIP CD listing alone. The sampler operates on filenames only.
    The 50 s filter therefore applies to sites documented in Phase 1 as having
    truncated files; all other files are assumed 60 s nominal.
    Files between 50–60 s are zero-padded during Phase 5 feature extraction.
    """
    if known_short_sites and rec.site_id in known_short_sites:
        return known_short_sites[rec.site_id]
    return rec.nominal_duration_s  # default: 60.0


# ---------------------------------------------------------------------------
# Segment manifest generation
# ---------------------------------------------------------------------------

def generate_segments(
    recordings: List[Dict[str, Any]],
    segments_per_recording: int = 12,
    segment_duration_s: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Generate segment manifest rows: one row per segment, 12 per recording.
    segment_id is constructed as: {recording_id}__seg{segment_index:02d}
    recording_id is: {site_id}__{filename_stem}
    """
    segments = []
    for rec in recordings:
        filename_stem = Path(rec["filename"]).stem
        recording_id = f"{rec['site_id']}__{filename_stem}"
        for seg_idx in range(segments_per_recording):
            segment_id = f"{recording_id}__seg{seg_idx:02d}"
            segments.append({
                "segment_id": segment_id,
                "recording_id": recording_id,
                "segment_index": seg_idx,
                "segment_start_s": round(seg_idx * segment_duration_s, 3),
                "segment_end_s": round((seg_idx + 1) * segment_duration_s, 3),
                "site_id": rec["site_id"],
                "site_code": rec["site_code"],
                "country": rec["country"],
                "habitat": rec["habitat_label"],
                "split": rec["split"],
            })
    return segments


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EcoSound-Audit Phase 4: Full Benchmark Sampling")
    parser.add_argument("--config", type=str, default="config/sampling_config_full.yaml")
    parser.add_argument("--max-archives", type=int, default=None,
                        help="Limit number of archives (for debugging)")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    log_file = PROJECT_ROOT / config["paths"]["pipeline_log"]
    logger = setup_logger(log_file)

    seed = config["pipeline"]["random_seed"]
    target_count = config["sampling"]["target_samples_per_site"]
    duty_cycle_strat = config["sampling"]["duty_cycle_handling"]
    min_dur = config["sampling"]["min_duration_seconds"]
    segs_per_rec = config["sampling"]["segments_per_recording"]
    seg_dur = config["sampling"]["segment_duration_seconds"]

    logger.info("=" * 60)
    logger.info("EcoSound-Audit Phase 4: Full Benchmark Sampling")
    logger.info("Config: %s", args.config)
    logger.info("Option A — TARGET_PER_SITE=%d, seed=%d, duty_cycle=%s, min_dur=%.0fs",
                target_count, seed, duty_cycle_strat, min_dur)
    logger.info("=" * 60)

    # Guard: verify pilot files are NOT in output paths
    pilot_sampled = PROJECT_ROOT / "data/sampling/sampled_recordings.csv"
    out_sampled = PROJECT_ROOT / config["paths"]["sampled_dataset_csv"]
    assert str(pilot_sampled.resolve()) != str(out_sampled.resolve()), \
        "FATAL: Output path would overwrite pilot sampled_recordings.csv!"
    logger.info("Guard check passed: output paths do not collide with pilot files.")

    # Create output directories
    out_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports/full"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest and Figshare file IDs
    manifest_csv_path = PROJECT_ROOT / config["paths"]["manifest_csv"]
    sites = load_manifest_csv(manifest_csv_path)
    logger.info("Loaded %d site archives from %s", len(sites), manifest_csv_path)

    figshare_json = PROJECT_ROOT / config["paths"]["figshare_files_json"]
    name_to_id = load_figshare_file_ids(figshare_json)
    logger.info("Loaded %d Figshare file IDs", len(name_to_id))

    # 2. Load FROZEN site split — do not re-derive splits
    frozen_split_path = PROJECT_ROOT / config["paths"]["site_metadata_csv"]
    frozen_splits = load_frozen_site_splits(frozen_split_path)
    logger.info("Loaded frozen split for %d sites from %s", len(frozen_splits), frozen_split_path)

    # Attach frozen split to each site in the manifest
    for s in sites:
        if s["site_id"] in frozen_splits:
            s["split"] = frozen_splits[s["site_id"]]["split"]
        else:
            logger.warning("Site %s not in frozen split CSV — assigning 'unknown'", s["site_id"])
            s["split"] = "unknown"

    # Limit archives if --max-archives is set (debug mode)
    selected_sites = sites[:args.max_archives] if args.max_archives else sites
    logger.info("Processing %d site archives...", len(selected_sites))

    # 3. Fetch remote ZIP central directories and sample
    sampled_records: List[Dict[str, Any]] = []
    archive_stats: Dict[str, Any] = {}
    shortfall_sites: List[Dict[str, Any]] = []

    # Known short-file sites from Phase 1 audit (ind_N1 has ~25.6 s files)
    # These are the ONLY sites where we know files are < min_duration_s.
    # All other sites are assumed to have 60 s nominal duration.
    KNOWN_SHORT_SITES = {"indonesia_N1"}  # files ~25.6 s — below 50 s threshold

    for s in selected_sites:
        zname = s["zip_name"]
        site_id = s["site_id"]
        fid = name_to_id.get(zname)

        if not fid:
            logger.warning("[%s] No Figshare file ID found — skipping", zname)
            continue

        # Flag ind_R1 — no MD5 on Figshare, SHA-256 to be verified before Phase 5
        if zname == "ind_R1.zip":
            logger.warning("[ind_R1.zip] WARNING: No MD5 checksum on Figshare. "
                           "SHA-256=%s (from manifest). To be verified before Phase 5.",
                           s.get("sha256", "N/A"))

        try:
            fnames = fetch_archive_filenames(fid, zname)
        except Exception as e:
            logger.error("[%s] Remote CD fetch failed: %s", zname, e)
            continue

        # Parse filenames into RecordingMetadata
        site_records: List[RecordingMetadata] = []
        for fn in fnames:
            try:
                rec = parse_recording_filename(fn, source_archive=zname)
                site_records.append(rec)
            except ValueError as ve:
                logger.debug("[%s] Skipping invalid filename %s: %s", zname, fn, ve)

        total_files = len(site_records)

        # Apply 50 s duration filter
        # For known short sites, all files are excluded. For all other sites,
        # files are assumed >= 60 s (no way to read durations from CD listing alone).
        if site_id in KNOWN_SHORT_SITES:
            eligible = []
            logger.warning("[%s] ALL %d files excluded: known short-duration site (<50 s)", site_id, total_files)
        else:
            eligible = site_records  # assume 60 s nominal for all other sites

        eligible_count = len(eligible)

        # Log shortfall if fewer than target_count eligible recordings
        if eligible_count < target_count:
            shortfall = {
                "site_id": site_id,
                "country": s["country"],
                "habitat_label": s["habitat_label"],
                "split": s["split"],
                "eligible_count": eligible_count,
                "target_count": target_count,
                "shortfall": target_count - eligible_count,
                "note": "Known short-duration site" if site_id in KNOWN_SHORT_SITES else "Insufficient recordings",
            }
            shortfall_sites.append(shortfall)
            logger.warning("[%s] SHORTFALL: %d eligible < %d target (shortfall=%d)",
                           site_id, eligible_count, target_count, target_count - eligible_count)

        # Temporal sampling — takes all available if eligible_count < target_count
        site_samples = sample_recordings_uniform_temporal(
            eligible,
            target_count=target_count,
            random_seed=seed,
            duty_cycle_handling=duty_cycle_strat,
        )

        # Attach split and site info to each sampled record
        for rec in site_samples:
            row = rec.to_dict()
            row["split"] = s["split"]
            sampled_records.append(row)

        archive_stats[site_id] = {
            "country": s["country"],
            "site_code": s["site_code"],
            "habitat_label": s["habitat_label"],
            "split": s["split"],
            "total_remote_files": total_files,
            "eligible_count": eligible_count,
            "sampled_count": len(site_samples),
            "shortfall": max(0, target_count - eligible_count),
            "sampling_ratio": round(len(site_samples) / max(1, eligible_count), 4),
            "ind_R1_integrity": (
                "WARNING: no MD5 on Figshare — SHA-256 pending Phase 5 verification"
                if site_id == "indonesia_R1" else "ok"
            ),
        }
        logger.info("[%s] %d total → %d eligible → %d sampled (%s split)",
                    site_id, total_files, eligible_count, len(site_samples), s["split"])

    # 4. Export recording manifest CSV
    rec_fieldnames = [
        "filename", "source_archive", "country", "country_code",
        "site_code", "site_id", "habitat_label", "split",
        "date_str", "time_str", "timestamp_iso", "hour_of_day",
        "minute_of_hour", "duty_cycle_fraction", "sample_rate_hz",
        "channels", "bits_per_sample", "nominal_duration_s",
    ]
    sampled_csv_path = PROJECT_ROOT / config["paths"]["sampled_dataset_csv"]
    with open(sampled_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rec_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampled_records)
    logger.info("Exported %d sampled recordings to %s", len(sampled_records), sampled_csv_path)

    # 5. Generate and export segment manifest
    segments = generate_segments(sampled_records, segments_per_recording=segs_per_rec, segment_duration_s=seg_dur)
    seg_csv_path = PROJECT_ROOT / config["paths"]["segment_manifest_csv"]
    seg_fieldnames = [
        "segment_id", "recording_id", "segment_index",
        "segment_start_s", "segment_end_s",
        "site_id", "site_code", "country", "habitat", "split",
    ]
    with open(seg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seg_fieldnames)
        writer.writeheader()
        writer.writerows(segments)
    logger.info("Exported %d segments to %s", len(segments), seg_csv_path)

    # 6. Export shortfall log
    shortfall_csv_path = PROJECT_ROOT / config["paths"]["shortfall_log_csv"]
    with open(shortfall_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "site_id", "country", "habitat_label", "split",
            "eligible_count", "target_count", "shortfall", "note",
        ])
        writer.writeheader()
        writer.writerows(shortfall_sites)
    logger.info("Exported shortfall log (%d sites) to %s", len(shortfall_sites), shortfall_csv_path)

    # 7. Compute summary statistics
    split_rec_counts: Dict[str, Counter] = defaultdict(Counter)
    split_site_counts: Dict[str, Counter] = defaultdict(Counter)
    country_rec_counts: Counter = Counter()
    hour_counts: Counter = Counter()

    for rec in sampled_records:
        split_rec_counts[rec["split"]][rec["habitat_label"]] += 1
        country_rec_counts[rec["country"]] += 1
        hour_counts[rec["hour_of_day"]] += 1

    for site_id, info in archive_stats.items():
        split_site_counts[info["split"]][info["habitat_label"]] += 1

    total_recordings = len(sampled_records)
    indonesia_count = country_rec_counts.get("indonesia", 0)
    indonesia_share = round(indonesia_count / max(1, total_recordings), 4)

    git_hash = get_git_commit_hash()
    pipeline_ts = datetime.now(timezone.utc).isoformat()

    # 8. Export sampling summary JSON
    summary_path = PROJECT_ROOT / config["paths"]["sampling_summary_json"]
    summary = {
        "pipeline_timestamp_utc": pipeline_ts,
        "git_commit_hash": git_hash,
        "config_file": str(PROJECT_ROOT / args.config),
        "random_seed": seed,
        "phase": "Phase 4 — Full Benchmark",
        "option": "A",
        "target_samples_per_site": target_count,
        "duty_cycle_handling": duty_cycle_strat,
        "min_duration_seconds": min_dur,
        "segments_per_recording": segs_per_rec,
        "segment_duration_seconds": seg_dur,
        "total_sites_in_dataset": len(sites),
        "total_sampled_recordings": total_recordings,
        "total_segments": len(segments),
        "indonesia_recordings": indonesia_count,
        "indonesia_share": indonesia_share,
        "shortfall_sites_count": len(shortfall_sites),
        "shortfall_sites": [s["site_id"] for s in shortfall_sites],
        "split_distribution": {
            sp: {
                "recordings": sum(split_rec_counts[sp].values()),
                "class_counts": dict(split_rec_counts[sp]),
                "site_counts": dict(split_site_counts[sp]),
            }
            for sp in ["train", "val", "test"]
        },
        "country_distribution": dict(country_rec_counts),
        "archive_breakdown": archive_stats,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Exported sampling summary to %s", summary_path)

    # 9. Export benchmark summary (JSON + Markdown)
    benchmark_json_path = PROJECT_ROOT / config["paths"]["benchmark_summary_json"]
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Exported benchmark summary JSON to %s", benchmark_json_path)

    benchmark_md_path = PROJECT_ROOT / config["paths"]["benchmark_summary_md"]
    with open(benchmark_md_path, "w", encoding="utf-8") as f:
        f.write(f"# EcoSound-Audit Phase 4 — Benchmark Summary\n\n")
        f.write(f"**Generated:** {pipeline_ts}  \n")
        f.write(f"**Git commit:** `{git_hash}`  \n")
        f.write(f"**Config:** `{args.config}`  \n")
        f.write(f"**Seed:** {seed}  \n\n")
        f.write(f"## Counts\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Total recordings | {total_recordings} |\n")
        f.write(f"| Total segments | {len(segments)} |\n")
        f.write(f"| Sites processed | {len(archive_stats)} |\n")
        f.write(f"| Sites with shortfall | {len(shortfall_sites)} |\n")
        f.write(f"| Indonesia share | {indonesia_share:.1%} |\n\n")

        f.write(f"## Recordings per Split & Class\n\n")
        f.write(f"| Split | D | H | N | R | Total |\n|---|---|---|---|---|---|\n")
        for sp in ["train", "val", "test"]:
            cc = split_rec_counts[sp]
            total_sp = sum(cc.values())
            f.write(f"| {sp} | {cc.get('D', 0)} | {cc.get('H', 0)} | "
                    f"{cc.get('N', 0)} | {cc.get('R', 0)} | {total_sp} |\n")

        f.write(f"\n## Shortfall Sites\n\n")
        if shortfall_sites:
            f.write("| site_id | eligible | target | shortfall | note |\n|---|---|---|---|---|\n")
            for s in shortfall_sites:
                f.write(f"| {s['site_id']} | {s['eligible_count']} | "
                        f"{s['target_count']} | {s['shortfall']} | {s.get('note', '')} |\n")
        else:
            f.write("_No shortfall sites — all 45 sites have ≥ 100 eligible recordings._\n")

        f.write(f"\n## ind_R1.zip Integrity\n\n")
        f.write("> **Status:** No MD5 checksum on Figshare for `ind_R1.zip`. "
                "SHA-256 from manifest: `faf51da77eb2343b5b2cd59d6f7b7d67474df10491ffa95a67e55497de4694d4`. "
                "Full verification (against downloaded bytes) is **pending Phase 5** before feature extraction.\n")

    logger.info("Exported benchmark summary Markdown to %s", benchmark_md_path)
    logger.info("=" * 60)
    logger.info("Phase 4 pipeline completed. Total: %d recordings, %d segments across %d sites.",
                total_recordings, len(segments), len(archive_stats))
    logger.info("REMINDER: Do NOT proceed to Phase 5 without user approval.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
