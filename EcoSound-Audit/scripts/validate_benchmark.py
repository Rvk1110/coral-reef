#!/usr/bin/env python3
"""
EcoSound-Audit — Phase 4 Benchmark Validation Script
=====================================================
Exits non-zero on any failure. Run after run_phase4_benchmark.py.

Checks:
  1.  45 sites present; each has TARGET_PER_SITE recordings (or documented shortfall).
  2.  No site_id appears in more than one split.
  3.  Class counts per split; all four classes (D, H, N, R) present in each split.
  4.  Country, hour-of-day, and date distributions; Indonesia's share.
  5.  No recording with nominal_duration_s < 50 s (excluding documented short sites).
  6.  No duplicate recording_id or segment_id.
  7.  Segment count == 12 × recording count.
  8.  Leave-One-Country-Out feasibility (class support per held-out country).

Usage:
------
    python scripts/validate_benchmark.py [--config config/sampling_config_full.yaml]
    python scripts/validate_benchmark.py --recordings data/sampling_full/sampled_recordings.csv \
                                         --segments   data/sampling_full/segment_manifest.csv   \
                                         --site-meta  data/sampling/site_metadata.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


PASS = green("  PASS")
FAIL = red("  FAIL")
WARN = yellow("  WARN")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def validate(
    recordings: List[Dict[str, str]],
    segments: List[Dict[str, str]],
    site_meta: List[Dict[str, str]],
    shortfall_csv: Path,
    target_per_site: int = 100,
    segments_per_recording: int = 12,
    min_duration_s: float = 50.0,
) -> int:
    """Run all validation checks. Returns number of failures."""
    failures = 0
    warnings = 0

    # Load shortfall log
    shortfall_sites: Set[str] = set()
    if shortfall_csv.exists():
        rows = load_csv(shortfall_csv)
        shortfall_sites = {r["site_id"] for r in rows}
        shortfall_map = {r["site_id"]: int(r["eligible_count"]) for r in rows}
    else:
        shortfall_map = {}

    print(f"\n{'='*60}")
    print(bold("EcoSound-Audit Phase 4 — Benchmark Validation"))
    print(f"{'='*60}")
    print(f"  Recordings CSV  : {len(recordings)} rows")
    print(f"  Segments CSV    : {len(segments)} rows")
    print(f"  Site metadata   : {len(site_meta)} sites")
    print(f"  Shortfall sites : {len(shortfall_sites)}")
    print(f"  TARGET_PER_SITE : {target_per_site}")
    print(f"{'='*60}\n")

    # ── Check 1: 45 sites, each with target recordings (or documented shortfall) ──
    print(bold("CHECK 1: 45 sites, each with TARGET_PER_SITE recordings (or documented shortfall)"))
    sites_in_recordings = defaultdict(int)
    for r in recordings:
        sites_in_recordings[r["site_id"]] += 1

    all_sites_in_meta = {row["site_id"] for row in site_meta}

    # Sites with 0 eligible recordings won't appear in the recording CSV at all,
    # but are valid if they are fully documented in the shortfall log.
    fully_documented_zero = {sid for sid in shortfall_sites if shortfall_map.get(sid, -1) == 0}
    # Effective sites = those in recording CSV + zero-eligible shortfall sites
    effective_sites = set(sites_in_recordings.keys()) | fully_documented_zero
    missing_sites = all_sites_in_meta - effective_sites
    extra_sites = effective_sites - all_sites_in_meta
    n_sites_found = len(effective_sites)

    if n_sites_found == 45 and not missing_sites and not extra_sites:
        print(f"{PASS}  {n_sites_found}/45 sites accounted for (recording manifest + shortfall log).")
    else:
        print(f"{FAIL}  Expected 45 sites, found {n_sites_found}. Missing: {sorted(missing_sites)}. Extra: {sorted(extra_sites)}")
        failures += 1

    # Report zero-eligible shortfall sites
    for sid in sorted(fully_documented_zero):
        print(f"  {yellow('DOCUMENTED SHORTFALL (0 eligible)')}: {sid} — 0/{target_per_site} recordings "
              f"(fully excluded: known short-duration site, shortfall logged)")

    sites_underfull = []
    for site_id, count in sorted(sites_in_recordings.items()):
        if count < target_per_site:
            if site_id in shortfall_sites:
                expected = shortfall_map.get(site_id, count)
                if count == expected:
                    print(f"  {yellow('DOCUMENTED SHORTFALL')}: {site_id} — {count}/{target_per_site} recordings (shortfall logged)")
                else:
                    print(f"{FAIL}  {site_id}: {count} recordings, shortfall logged as {expected} — count mismatch!")
                    failures += 1
            else:
                print(f"{FAIL}  {site_id}: {count} < {target_per_site} recordings — NOT in shortfall log!")
                failures += 1
                sites_underfull.append(site_id)
    if not sites_underfull:
        print(f"{PASS}  All sites meet target or have documented shortfalls.")

    # ── Check 2: No site_id in more than one split ──
    print(f"\n{bold('CHECK 2: No site_id appears in more than one split')}")
    site_splits: Dict[str, Set[str]] = defaultdict(set)
    for r in recordings:
        site_splits[r["site_id"]].add(r["split"])

    multi_split_sites = {sid: sp for sid, sp in site_splits.items() if len(sp) > 1}
    if not multi_split_sites:
        print(f"{PASS}  No site_id appears in more than one split.")
    else:
        for sid, sp in multi_split_sites.items():
            print(f"{FAIL}  {sid} appears in splits: {sp}")
        failures += 1

    # ── Check 3: Class counts per split; all 4 classes in each split ──
    print(f"\n{bold('CHECK 3: Class counts per split; all four classes in each split')}")
    split_class_counts: Dict[str, Counter] = defaultdict(Counter)
    for r in recordings:
        split_class_counts[r["split"]][r["habitat_label"]] += 1

    all_ok = True
    for sp in ["train", "val", "test"]:
        cc = split_class_counts[sp]
        total = sum(cc.values())
        classes_present = set(cc.keys())
        missing = {"D", "H", "N", "R"} - classes_present
        line = f"  {sp:6s}: D={cc.get('D', 0):4d}  H={cc.get('H', 0):4d}  N={cc.get('N', 0):4d}  R={cc.get('R', 0):4d}  total={total:5d}"
        if missing:
            print(f"{FAIL}{line}  ← MISSING classes: {missing}")
            failures += 1
            all_ok = False
        else:
            print(f"{PASS}{line}")
    if all_ok:
        print(f"{PASS}  All four classes present in all three splits.")

    # ── Check 4: Country, hour-of-day, date distributions; Indonesia share ──
    print(f"\n{bold('CHECK 4: Country, hour-of-day, and date distributions')}")
    country_counts: Counter = Counter()
    hour_counts: Counter = Counter()
    date_counts: Counter = Counter()
    for r in recordings:
        country_counts[r["country"]] += 1
        hour_counts[int(r["hour_of_day"])] += 1
        date_counts[r["date_str"][:7]] += 1  # year-month

    total_recs = len(recordings)
    print(f"  Country distribution (total={total_recs}):")
    for country, cnt in sorted(country_counts.items()):
        share = cnt / max(1, total_recs)
        print(f"    {country:12s}: {cnt:5d} ({share:.1%})")
    indonesia_count = country_counts.get("indonesia", 0)
    indonesia_share = indonesia_count / max(1, total_recs)
    print(f"\n  Indonesia share: {indonesia_count}/{total_recs} = {indonesia_share:.1%}")
    if indonesia_share > 0.7:
        print(f"{WARN}  Indonesia exceeds 70% of recordings — check duty-cycle handling.")
        warnings += 1
    else:
        print(f"{PASS}  Indonesia share ({indonesia_share:.1%}) is within expected bounds.")

    # Hour distribution: check all 24 hours represented
    hours_present = set(hour_counts.keys())
    missing_hours = set(range(24)) - hours_present
    if missing_hours:
        print(f"{WARN}  Hour-of-day bins missing: {sorted(missing_hours)}")
        warnings += 1
    else:
        print(f"{PASS}  All 24 hours-of-day represented across recordings.")

    # ── Check 5: No recording < 50 s nominal duration ──
    print(f"\n{bold('CHECK 5: No recording with nominal_duration_s < 50 s')}")
    short_files = [r for r in recordings if float(r["nominal_duration_s"]) < min_duration_s]
    if short_files:
        print(f"{FAIL}  {len(short_files)} recordings have nominal_duration_s < {min_duration_s}:")
        for r in short_files[:5]:
            print(f"    {r['filename']}  dur={r['nominal_duration_s']}")
        failures += 1
    else:
        print(f"{PASS}  No recordings with nominal_duration_s < {min_duration_s} s.")

    # ── Check 6: No duplicate recording_id or segment_id ──
    print(f"\n{bold('CHECK 6: No duplicate recording_id or segment_id')}")
    # Build recording_ids from filename+site_id
    rec_ids = []
    for r in recordings:
        stem = Path(r["filename"]).stem
        recording_id = f"{r['site_id']}__{stem}"
        rec_ids.append(recording_id)

    dup_recs = [rid for rid, cnt in Counter(rec_ids).items() if cnt > 1]
    if dup_recs:
        print(f"{FAIL}  {len(dup_recs)} duplicate recording_ids found:")
        for d in dup_recs[:5]:
            print(f"    {d}")
        failures += 1
    else:
        print(f"{PASS}  No duplicate recording_ids.")

    seg_ids = [s["segment_id"] for s in segments]
    dup_segs = [sid for sid, cnt in Counter(seg_ids).items() if cnt > 1]
    if dup_segs:
        print(f"{FAIL}  {len(dup_segs)} duplicate segment_ids found:")
        for d in dup_segs[:5]:
            print(f"    {d}")
        failures += 1
    else:
        print(f"{PASS}  No duplicate segment_ids ({len(seg_ids)} total).")

    # ── Check 7: Segment count == 12 × recording count ──
    print(f"\n{bold('CHECK 7: Segment count == segments_per_recording × recording count')}")
    expected_segments = len(recordings) * segments_per_recording
    actual_segments = len(segments)
    if actual_segments == expected_segments:
        print(f"{PASS}  {actual_segments} segments == {len(recordings)} recordings × {segments_per_recording}.")
    else:
        print(f"{FAIL}  Expected {expected_segments} segments, found {actual_segments}.")
        failures += 1

    # ── Check 8: Leave-One-Country-Out feasibility ──
    print(f"\n{bold('CHECK 8: Leave-One-Country-Out feasibility — class support per held-out country')}")
    # Expected gaps: Australia has no N; Kenya and Maldives have no R
    EXPECTED_MISSING = {
        "australia": {"N"},
        "kenya": {"R"},
        "maldives": {"R"},
    }
    country_class_counts: Dict[str, Counter] = defaultdict(Counter)
    for r in recordings:
        country_class_counts[r["country"]][r["habitat_label"]] += 1

    all_countries = sorted(country_class_counts.keys())
    loco_ok = True
    for held_out in all_countries:
        classes = set(country_class_counts[held_out].keys())
        missing_in_held_out = {"D", "H", "N", "R"} - classes
        expected_missing = EXPECTED_MISSING.get(held_out, set())
        unexpected_missing = missing_in_held_out - expected_missing
        expected_but_present = expected_missing - missing_in_held_out

        class_detail = "  ".join(
            f"{c}={country_class_counts[held_out].get(c, 0)}"
            for c in ["D", "H", "N", "R"]
        )
        print(f"\n  Held-out country: {held_out}")
        print(f"    Classes: {class_detail}")

        if missing_in_held_out:
            if missing_in_held_out == expected_missing:
                print(f"    {yellow('EXPECTED ABSENCE')}: classes {missing_in_held_out} absent — documented in PHASE3_PLAN.md")
            else:
                if unexpected_missing:
                    print(f"    {red('UNEXPECTED ABSENCE')}: classes {unexpected_missing} missing unexpectedly!")
                    failures += 1
                    loco_ok = False
                if expected_but_present:
                    print(f"    {yellow('NOTE')}: expected {expected_but_present} to be absent but they are present.")
        else:
            print(f"    {green('All 4 classes present.')}")

    if loco_ok:
        print(f"\n{PASS}  LOCO feasibility check passed (documented absences confirmed).")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(bold("VALIDATION SUMMARY"))
    print(f"{'='*60}")
    print(f"  Failures  : {failures}")
    print(f"  Warnings  : {warnings}")
    if failures == 0:
        print(green(f"  STATUS: ALL CHECKS PASSED"))
    else:
        print(red(f"  STATUS: {failures} CHECK(S) FAILED — review output above"))
    print(f"{'='*60}\n")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EcoSound-Audit Phase 4 Benchmark Validation")
    parser.add_argument("--config", type=str, default="config/sampling_config_full.yaml")
    parser.add_argument("--recordings", type=str, default="data/sampling_full/sampled_recordings.csv")
    parser.add_argument("--segments", type=str, default="data/sampling_full/segment_manifest.csv")
    parser.add_argument("--site-meta", type=str, default="data/sampling/site_metadata.csv")
    parser.add_argument("--shortfall", type=str, default="data/sampling_full/shortfall_log.csv")
    parser.add_argument("--target", type=int, default=100, help="TARGET_PER_SITE")
    parser.add_argument("--segments-per-rec", type=int, default=12)
    parser.add_argument("--min-duration", type=float, default=50.0)
    args = parser.parse_args()

    rec_path = PROJECT_ROOT / args.recordings
    seg_path = PROJECT_ROOT / args.segments
    meta_path = PROJECT_ROOT / args.site_meta
    shortfall_path = PROJECT_ROOT / args.shortfall

    for p, label in [(rec_path, "recordings"), (seg_path, "segments"), (meta_path, "site-meta")]:
        if not p.exists():
            print(red(f"ERROR: {label} file not found: {p}"))
            sys.exit(2)

    recordings = load_csv(rec_path)
    segments = load_csv(seg_path)
    site_meta = load_csv(meta_path)

    failures = validate(
        recordings, segments, site_meta,
        shortfall_csv=shortfall_path,
        target_per_site=args.target,
        segments_per_recording=args.segments_per_rec,
        min_duration_s=args.min_duration,
    )
    sys.exit(failures)


if __name__ == "__main__":
    main()
