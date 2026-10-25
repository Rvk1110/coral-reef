"""
EcoSound-Audit — Reproducible Sampling Pipeline
================================================
Implements:
1. Uniform temporal sampling across diurnal hours and dates.
2. Handling of Indonesia's 1-in-2 min duty cycle (balanced hourly coverage).
3. Preserving all 4 habitat labels (D, H, N, R).
4. Strict site-level separation for Train / Val / Test and Leave-One-Country-Out (LOCO).
"""
from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ecosound.models import RecordingMetadata, parse_recording_filename

logger = logging.getLogger("ecosound_phase2")


def sample_recordings_uniform_temporal(
    recordings: List[RecordingMetadata],
    target_count: int,
    random_seed: int = 42,
    duty_cycle_handling: str = "hourly_binning",
) -> List[RecordingMetadata]:
    """
    Sample recordings from a single site uniformly stratified across both
    recording deployment dates and hours of the day to avoid temporal clustering.

    Parameters:
    -----------
    recordings: List of RecordingMetadata for a single site.
    target_count: Maximum number of recordings to sample from this site.
    random_seed: Fixed seed for exact reproducibility.
    duty_cycle_handling:
        - "hourly_binning": Stratifies recordings across available deployment date bins
          and hour-of-day bins (0-23). Samples an equal quota of files per site,
          spreading selections across the recorded deployment days and diurnal hours.
          *Note*: This normalizes the sample count per site and prevents temporal clustering
          within the site's recorded window, but does NOT alter the original recording cadence
          or balance across macro-seasons (since each site was typically recorded for ~1 month).
        - "subsample_50pct": Specifically for Indonesia, discards alternate minutes
          (aligning to a 4-minute grid: minute % 4 == 0) prior to temporal sampling.
        - "direct_random": Stratified uniform random across the site.
    """
    if not recordings:
        return []

    rng = random.Random(random_seed)
    candidates = list(recordings)
    is_indonesia = candidates[0].country == "indonesia"

    # 1. Apply duty-cycle filtering BEFORE early-return checks
    if is_indonesia and duty_cycle_handling == "subsample_50pct":
        # Keep only recordings on 4-minute grid: e.g. minute % 4 == 0
        grid_aligned = [r for r in candidates if r.minute_of_hour % 4 == 0]
        if grid_aligned:
            candidates = grid_aligned

    # 2. Early-return check on filtered candidates
    if len(candidates) <= target_count:
        return sorted(candidates, key=lambda r: r.filename)

    # 3. 2D Stratified Temporal Sampling (Date & Hour)
    if duty_cycle_handling in ("hourly_binning", "subsample_50pct"):
        # Group recordings into a 2D grid: (date, hour)
        grid_bins: Dict[Tuple[str, int], List[RecordingMetadata]] = defaultdict(list)
        distinct_dates = sorted({r.date_str for r in candidates})
        distinct_hours = sorted({r.hour_of_day for r in candidates})

        for r in candidates:
            grid_bins[(r.date_str, r.hour_of_day)].append(r)

        # Deterministically shuffle files within each (date, hour) cell
        for cell in grid_bins:
            rng.shuffle(grid_bins[cell])

        # Generate ordered target tuples interleaving dates and diurnal hours
        # Date stepping and hour stepping ensure both deployment span and diurnal cycle are covered
        selected: List[RecordingMetadata] = []
        cell_indices: Dict[Tuple[str, int], int] = defaultdict(int)

        # Build prioritized sequence of (date, hour) combinations with seed-dependent offsets
        n_dates = len(distinct_dates)
        n_hours = len(distinct_hours)

        date_offset = rng.randint(0, n_dates - 1) if n_dates > 1 else 0
        hour_offset = rng.randint(0, n_hours - 1) if n_hours > 1 else 0

        # Step across dates and hours evenly
        date_step = n_dates / target_count if target_count < n_dates else 1.0
        hour_step = n_hours / target_count if target_count < n_hours else 1.0

        planned_pairs: List[Tuple[str, int]] = []
        for i in range(target_count):
            d_idx = int(date_offset + i * date_step) % n_dates
            h_idx = int(hour_offset + i * hour_step) % n_hours
            planned_pairs.append((distinct_dates[d_idx], distinct_hours[h_idx]))

        # First pass: try to sample the planned (date, hour) cells
        for d, h in planned_pairs:
            idx = cell_indices[(d, h)]
            if idx < len(grid_bins[(d, h)]):
                selected.append(grid_bins[(d, h)][idx])
                cell_indices[(d, h)] += 1

        # Second pass: if some planned cells were empty (e.g. recorder gap on that specific date/hour),
        # fill remaining slots in round-robin fashion across available (date, hour) cells
        if len(selected) < target_count:
            all_cells = sorted(grid_bins.keys())
            rng.shuffle(all_cells)
            while len(selected) < target_count:
                added = 0
                for cell in all_cells:
                    idx = cell_indices[cell]
                    if idx < len(grid_bins[cell]):
                        selected.append(grid_bins[cell][idx])
                        cell_indices[cell] += 1
                        added += 1
                        if len(selected) >= target_count:
                            break
                if added == 0:
                    break

        return sorted(selected, key=lambda r: r.filename)

    # Default direct random sampling
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected = shuffled[:target_count]
    return sorted(selected, key=lambda r: r.filename)


def create_site_level_splits(
    site_metadata_list: List[Dict[str, Any]],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Partition sites into Train, Validation, and Test sets, preserving:
    1. Strict site-level separation (zero site overlap between splits).
    2. Representation of all 4 habitat labels (D, H, N, R) in Train and Test where possible.
    3. Representation of countries across splits.

    Returns a mapping: site_id -> split_info dict.
    """
    rng = random.Random(random_seed)

    # Group sites by (country, habitat_label)
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for site in site_metadata_list:
        key = (site["country"], site["habitat_label"])
        strata[key].append(site)

    splits: Dict[str, Dict[str, Any]] = {}

    # For each stratum, deterministically allocate sites
    for (country, habitat), sites in sorted(strata.items()):
        shuffled = list(sites)
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n == 1:
            # If only 1 site exists in this stratum (e.g., aus_R1, mex_R1, mex_N1, etc.)
            # Deterministically assign: use hash of site_id with seed to distribute across splits
            h_val = rng.random()
            if h_val < train_ratio:
                split_name = "train"
            elif h_val < train_ratio + val_ratio:
                split_name = "val"
            else:
                split_name = "test"
            splits[shuffled[0]["site_id"]] = {
                "split": split_name,
                "country": country,
                "habitat_label": habitat,
                "reason": "single_site_stratum",
            }
        elif n == 2:
            # 1 to train, 1 to test (or val)
            splits[shuffled[0]["site_id"]] = {
                "split": "train",
                "country": country,
                "habitat_label": habitat,
                "reason": "two_site_stratum_1",
            }
            splits[shuffled[1]["site_id"]] = {
                "split": "test",
                "country": country,
                "habitat_label": habitat,
                "reason": "two_site_stratum_2",
            }
        else:
            # n >= 3
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            n_test = n - n_train - n_val
            if n_test < 1 and n >= 3:
                if n_train > 1:
                    n_train -= 1
                    n_test = 1
                else:
                    n_test = 1

            for s in shuffled[:n_train]:
                splits[s["site_id"]] = {
                    "split": "train",
                    "country": country,
                    "habitat_label": habitat,
                    "reason": "multi_site_stratum",
                }
            for s in shuffled[n_train:n_train + n_val]:
                splits[s["site_id"]] = {
                    "split": "val",
                    "country": country,
                    "habitat_label": habitat,
                    "reason": "multi_site_stratum",
                }
            for s in shuffled[n_train + n_val:]:
                splits[s["site_id"]] = {
                    "split": "test",
                    "country": country,
                    "habitat_label": habitat,
                    "reason": "multi_site_stratum",
                }

    return splits


def create_loco_folds(
    site_metadata_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Create Leave-One-Country-Out (LOCO) evaluation folds.
    For each held-out country, train set contains all sites from the other 4 countries,
    and test set contains all sites from the held-out country.
    """
    countries = sorted({s["country"] for s in site_metadata_list})
    folds: Dict[str, Dict[str, List[str]]] = {}

    for held_out in countries:
        train_sites = [s["site_id"] for s in site_metadata_list if s["country"] != held_out]
        test_sites = [s["site_id"] for s in site_metadata_list if s["country"] == held_out]
        folds[f"LOCO_{held_out}"] = {
            "held_out_country": held_out,
            "train_sites": sorted(train_sites),
            "test_sites": sorted(test_sites),
        }

    return folds
