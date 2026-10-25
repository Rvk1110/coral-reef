#!/usr/bin/env python3
"""
EcoSound-Audit — Phase 2 Main Sampling Pipeline
================================================
Reproducible manifest generation, site-level split partitioning,
and uniform temporal sampling across all 45 sites in the 5 countries.

Usage:
------
    python scripts/run_sampling_pipeline.py [--config config/sampling_config.yaml] [--pilot]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecosound.models import RecordingMetadata, parse_recording_filename
from ecosound.remote_reader import fetch_archive_filenames
from ecosound.sampler import (
    create_loco_folds,
    create_site_level_splits,
    sample_recordings_uniform_temporal,
)


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ecosound_phase2")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    
    # Avoid duplicate handlers
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


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest_csv(manifest_path: Path) -> List[Dict[str, Any]]:
    sites = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zip_name = row["zip_name"].strip()
            # Skip detection archives
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


def load_figshare_file_ids(json_path: Path) -> Dict[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["name"]: item["id"] for item in items}


def main():
    parser = argparse.ArgumentParser(description="EcoSound-Audit Phase 2 Pipeline")
    parser.add_argument("--config", type=str, default="config/sampling_config.yaml", help="Path to config file")
    parser.add_argument("--pilot", action="store_true", help="Run pilot sampling (fewer archives/recordings)")
    parser.add_argument("--max-archives", type=int, default=None, help="Limit number of archives for remote fetch")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    log_file = PROJECT_ROOT / config["paths"]["pipeline_log"]
    logger = setup_logger(log_file)

    logger.info("============================================================")
    logger.info("Starting EcoSound-Audit Phase 2: Manifest & Sampling Pipeline")
    logger.info("Random Seed: %d", config["pipeline"]["random_seed"])
    logger.info("Duty Cycle Handling: %s", config["sampling"]["duty_cycle_handling"])
    logger.info("============================================================")

    out_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load site inventory from manifest.csv
    manifest_csv = PROJECT_ROOT / config["paths"]["manifest_csv"]
    sites = load_manifest_csv(manifest_csv)
    logger.info("Loaded %d site archives from %s", len(sites), manifest_csv)

    # Load Figshare file IDs
    figshare_json = PROJECT_ROOT / config["paths"]["figshare_files_json"]
    name_to_id = load_figshare_file_ids(figshare_json)

    # 2. Partition Sites into Strict Site-Level Splits (Train / Val / Test)
    split_info = create_site_level_splits(
        sites,
        train_ratio=config["splits"]["train_ratio"],
        val_ratio=config["splits"]["val_ratio"],
        test_ratio=config["splits"]["test_ratio"],
        random_seed=config["pipeline"]["random_seed"],
    )

    # Attach split to sites
    for s in sites:
        s["split"] = split_info[s["site_id"]]["split"]

    # Export site metadata and splits
    site_meta_csv = PROJECT_ROOT / config["paths"]["site_metadata_csv"]
    with open(site_meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "site_id", "country", "site_code", "habitat_label", "split",
            "file_count", "total_bytes", "zip_name", "sha256"
        ])
        writer.writeheader()
        for s in sites:
            writer.writerow(s)
    logger.info("Exported site metadata to %s", site_meta_csv)

    # Create LOCO folds
    loco_folds = create_loco_folds(sites)
    logger.info("Created %d Leave-One-Country-Out (LOCO) folds", len(loco_folds))

    # 3. Remote Archive Central Directory Reader for Representative Sample
    # In pilot mode or full mode, sample archives covering all 5 countries and all 4 labels
    target_count = config["sampling"]["pilot_samples_per_site"] if args.pilot else config["sampling"]["target_samples_per_site"]
    duty_cycle_strat = config["sampling"]["duty_cycle_handling"]
    seed = config["pipeline"]["random_seed"]

    # Select archives to fetch filenames from
    if args.max_archives:
        selected_sites = sites[:args.max_archives]
    elif args.pilot:
        # Pick 1 archive per country covering D, H, N, R
        pilot_zips = [
            "aus_D1.zip", "aus_H2.zip", "aus_R1.zip",
            "ken_H1.zip", "ken_N1.zip",
            "mal_D1.zip", "mal_N1.zip",
            "mex_N1.zip", "mex_R1.zip",
            "ind_H4.zip", "ind_N1.zip", "ind_R1.zip",
        ]
        selected_sites = [s for s in sites if s["zip_name"] in pilot_zips]
    else:
        # Full selection
        selected_sites = sites

    logger.info("Fetching remote Central Directory file listings for %d site archives...", len(selected_sites))

    sampled_records: List[RecordingMetadata] = []
    all_inspected_manifest_records: List[RecordingMetadata] = []
    archive_stats: Dict[str, Any] = {}

    for s in selected_sites:
        zname = s["zip_name"]
        fid = name_to_id.get(zname)
        if not fid:
            logger.warning("No Figshare file ID found for %s, skipping", zname)
            continue

        try:
            fnames = fetch_archive_filenames(fid, zname)
            site_records: List[RecordingMetadata] = []
            for fn in fnames:
                try:
                    rec = parse_recording_filename(fn, source_archive=zname)
                    site_records.append(rec)
                except ValueError as ve:
                    logger.warning("[%s] Skipping invalid filename %s: %s", zname, fn, ve)

            all_inspected_manifest_records.extend(site_records)

            # Perform uniform temporal sampling
            site_samples = sample_recordings_uniform_temporal(
                site_records,
                target_count=target_count,
                random_seed=seed,
                duty_cycle_handling=duty_cycle_strat,
            )
            sampled_records.extend(site_samples)

            archive_stats[s["site_id"]] = {
                "country": s["country"],
                "habitat_label": s["habitat_label"],
                "split": s["split"],
                "total_remote_files": len(site_records),
                "sampled_files": len(site_samples),
                "sampling_ratio": round(len(site_samples) / max(1, len(site_records)), 4),
            }
            logger.info("[%s] %d total files -> %d sampled (%s split)",
                        s["site_id"], len(site_records), len(site_samples), s["split"])

        except Exception as e:
            logger.error("[%s] Remote Central Directory fetch failed: %s", zname, e)

    # 4. Export Sampled Recordings CSV
    sampled_csv = PROJECT_ROOT / config["paths"]["sampled_dataset_csv"]
    with open(sampled_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename", "source_archive", "country", "country_code",
            "site_code", "site_id", "habitat_label", "split",
            "date_str", "time_str", "timestamp_iso", "hour_of_day",
            "minute_of_hour", "duty_cycle_fraction", "sample_rate_hz",
            "channels", "bits_per_sample", "nominal_duration_s"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sampled_records:
            row = r.to_dict()
            row["split"] = split_info[r.site_id]["split"]
            writer.writerow(row)
    logger.info("Exported %d sampled recordings to %s", len(sampled_records), sampled_csv)

    # 5. Export Summary JSON
    # Compute per-split habitat coverage for actual sampled recordings
    split_sampled_habs = defaultdict(Counter)
    for r in sampled_records:
        split_name = split_info[r.site_id]["split"]
        split_sampled_habs[split_name][r.habitat_label] += 1

    # Compute all-sites per-split habitat coverage
    all_sites_split_habs = defaultdict(Counter)
    for s in sites:
        all_sites_split_habs[s["split"]][s["habitat_label"]] += 1

    sampled_site_ids = set(archive_stats.keys())
    unsampled_sites = [
        {
            "site_id": s["site_id"],
            "country": s["country"],
            "habitat_label": s["habitat_label"],
            "split": s["split"],
            "file_count": s["file_count"],
        }
        for s in sites if s["site_id"] not in sampled_site_ids
    ]

    summary_path = PROJECT_ROOT / config["paths"]["sampling_summary_json"]
    summary = {
        "pipeline_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "pilot_mode": args.pilot,
        "target_samples_per_site": target_count,
        "duty_cycle_handling": duty_cycle_strat,
        "total_sites_in_dataset": len(sites),
        "sampled_sites_count": len(archive_stats),
        "unsampled_sites_count": len(unsampled_sites),
        "total_sampled_recordings": len(sampled_records),
        "all_45_sites_split_distribution": {
            "train_sites_count": sum(1 for s in sites if s["split"] == "train"),
            "val_sites_count": sum(1 for s in sites if s["split"] == "val"),
            "test_sites_count": sum(1 for s in sites if s["split"] == "test"),
            "habitat_coverage_all_sites": {
                sp: {
                    "classes_present": sorted(all_sites_split_habs[sp].keys()),
                    "counts_by_class": dict(all_sites_split_habs[sp]),
                    "all_four_classes_present": set(all_sites_split_habs[sp].keys()) == {"D", "H", "N", "R"},
                }
                for sp in ["train", "val", "test"]
            },
        },
        "sampled_recordings_split_distribution": {
            "train_recordings_count": sum(1 for r in sampled_records if split_info[r.site_id]["split"] == "train"),
            "val_recordings_count": sum(1 for r in sampled_records if split_info[r.site_id]["split"] == "val"),
            "test_recordings_count": sum(1 for r in sampled_records if split_info[r.site_id]["split"] == "test"),
            "habitat_coverage_sampled": {
                sp: {
                    "classes_present": sorted(split_sampled_habs[sp].keys()),
                    "counts_by_class": dict(split_sampled_habs[sp]),
                    "all_four_classes_present": set(split_sampled_habs[sp].keys()) == {"D", "H", "N", "R"},
                    "limitations": (
                        "All 4 classes present" if set(split_sampled_habs[sp].keys()) == {"D", "H", "N", "R"}
                        else f"Missing classes in {sp} sample: {sorted(set(['D', 'H', 'N', 'R']) - set(split_sampled_habs[sp].keys()))}"
                    ),
                }
                for sp in ["train", "val", "test"]
            },
        },
        "sampled_sites": sorted(list(sampled_site_ids)),
        "unsampled_sites": unsampled_sites,
        "archive_sampling_breakdown": archive_stats,
        "loco_folds": loco_folds,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Exported sampling summary to %s", summary_path)
    logger.info("Phase 2 pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
