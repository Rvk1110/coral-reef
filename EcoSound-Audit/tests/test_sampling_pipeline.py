"""
Test Suite for EcoSound-Audit Pipeline
======================================
Tests:
1. Strict filename parsing & regex validation.
2. Handling of invalid filenames.
3. Preserved habitat labels (D, H, N, R).
4. Uniform temporal sampling & Indonesia duty cycle handling.
5. Strict site-level separation (zero leakage between splits).
6. LOCO fold creation and country coverage.
"""
import pytest
from ecosound.models import RecordingMetadata, parse_recording_filename
from ecosound.sampler import (
    create_loco_folds,
    create_site_level_splits,
    sample_recordings_uniform_temporal,
)


def test_filename_parsing_valid():
    fname = "aus_D1_20230207_120400.WAV"
    meta = parse_recording_filename(fname, "aus_D1.zip")
    assert meta.filename == "aus_D1_20230207_120400.WAV"
    assert meta.country == "australia"
    assert meta.country_code == "aus"
    assert meta.site_code == "D1"
    assert meta.site_id == "australia_D1"
    assert meta.habitat_label == "D"
    assert meta.date_str == "2023-02-07"
    assert meta.time_str == "12:04:00"
    assert meta.timestamp_iso == "2023-02-07T12:04:00"
    assert meta.hour_of_day == 12
    assert meta.minute_of_hour == 4
    assert meta.duty_cycle_fraction == 0.25
    assert meta.sample_rate_hz == 16000


def test_indonesia_duty_cycle_metadata():
    fname = "ind_R1_20220829_092000.WAV"
    meta = parse_recording_filename(fname)
    assert meta.country == "indonesia"
    assert meta.habitat_label == "R"
    assert meta.duty_cycle_fraction == 0.5


def test_all_habitat_labels_parse():
    for label in ["D", "H", "N", "R"]:
        fname = f"ken_{label}1_20230101_100000.wav"
        meta = parse_recording_filename(fname)
        assert meta.habitat_label == label


def test_invalid_filename_raises():
    with pytest.raises(ValueError, match="Invalid recording filename format"):
        parse_recording_filename("corrupted_filename.wav")

    with pytest.raises(ValueError, match="Unknown country prefix"):
        parse_recording_filename("usa_D1_20230101_100000.wav")

    with pytest.raises(ValueError, match="Invalid habitat label"):
        parse_recording_filename("aus_X1_20230101_100000.wav")

    with pytest.raises(ValueError, match="Invalid date/time components"):
        parse_recording_filename("aus_D1_20231332_999999.wav")


def test_uniform_temporal_sampling_date_and_hour_coverage():
    recordings = []
    # Create recordings across 5 distinct dates and 24 hours per date
    dates = ["20230201", "20230205", "20230210", "20230215", "20230220"]
    for d in dates:
        for h in range(24):
            for m in [0, 12, 24, 36, 48]:
                fname = f"aus_D1_{d}_{h:02d}{m:02d}00.WAV"
                recordings.append(parse_recording_filename(fname))

    # Sample 10 recordings
    sampled = sample_recordings_uniform_temporal(
        recordings, target_count=10, random_seed=42, duty_cycle_handling="hourly_binning"
    )
    assert len(sampled) == 10

    # Verify both dates and hours are broadly represented
    distinct_dates_sampled = {r.date_str for r in sampled}
    distinct_hours_sampled = {r.hour_of_day for r in sampled}
    assert len(distinct_dates_sampled) >= 4, f"Expected at least 4 dates, got {distinct_dates_sampled}"
    assert len(distinct_hours_sampled) >= 8, f"Expected at least 8 hours, got {distinct_hours_sampled}"


def test_duty_cycle_early_return_filtering():
    # If a site has 10 recordings, but 5 are filtered by subsample_50pct,
    # target_count=8 should return only the 5 filtered recordings, not crash or return unfiltered.
    recordings = []
    for m in range(10):  # minutes 0 through 9
        fname = f"ind_R1_20220829_12{m:02d}00.WAV"
        recordings.append(parse_recording_filename(fname))

    # With subsample_50pct, only minute % 4 == 0 are kept: 0, 4, 8 (3 recordings)
    sampled = sample_recordings_uniform_temporal(
        recordings, target_count=8, random_seed=42, duty_cycle_handling="subsample_50pct"
    )
    assert len(sampled) == 3
    for r in sampled:
        assert r.minute_of_hour % 4 == 0


def test_exact_reproducibility_same_seed():
    recordings = []
    for d in ["20230101", "20230102", "20230103"]:
        for h in range(12):
            fname = f"ken_H1_{d}_{h:02d}0000.WAV"
            recordings.append(parse_recording_filename(fname))

    run1 = sample_recordings_uniform_temporal(recordings, target_count=6, random_seed=123)
    run2 = sample_recordings_uniform_temporal(recordings, target_count=6, random_seed=123)
    run_diff_seed = sample_recordings_uniform_temporal(recordings, target_count=6, random_seed=999)

    assert [r.filename for r in run1] == [r.filename for r in run2]
    # Verify different seed alters sampling order / choices
    assert [r.filename for r in run1] != [r.filename for r in run_diff_seed]


def test_site_level_separation_no_leakage():
    # Mock site list representing the 45 sites
    site_list = []
    countries = ["australia", "indonesia", "kenya", "maldives", "mexico"]
    habitats = ["D", "H", "N", "R"]
    for c in countries:
        for h in habitats:
            # 2 sites per stratum
            for i in [1, 2]:
                site_id = f"{c}_{h}{i}"
                site_list.append({
                    "site_id": site_id,
                    "country": c,
                    "habitat_label": h,
                })

    splits = create_site_level_splits(site_list, random_seed=42)

    # 1. Zero site overlap
    train_sites = {s for s, v in splits.items() if v["split"] == "train"}
    val_sites = {s for s, v in splits.items() if v["split"] == "val"}
    test_sites = {s for s, v in splits.items() if v["split"] == "test"}

    assert len(train_sites.intersection(val_sites)) == 0
    assert len(train_sites.intersection(test_sites)) == 0
    assert len(val_sites.intersection(test_sites)) == 0

    # 2. All 4 habitat labels present in train and test
    train_habs = {splits[s]["habitat_label"] for s in train_sites}
    test_habs = {splits[s]["habitat_label"] for s in test_sites}
    assert train_habs == {"D", "H", "N", "R"}
    assert test_habs == {"D", "H", "N", "R"}


def test_loco_folds():
    site_list = [
        {"site_id": "australia_D1", "country": "australia"},
        {"site_id": "indonesia_D1", "country": "indonesia"},
        {"site_id": "kenya_D1", "country": "kenya"},
        {"site_id": "maldives_D1", "country": "maldives"},
        {"site_id": "mexico_D1", "country": "mexico"},
    ]
    folds = create_loco_folds(site_list)
    assert len(folds) == 5
    assert "LOCO_australia" in folds
    assert "australia_D1" in folds["LOCO_australia"]["test_sites"]
    assert "australia_D1" not in folds["LOCO_australia"]["train_sites"]
    assert len(folds["LOCO_australia"]["train_sites"]) == 4


# ===========================================================================
# Manifest & Artifact Integration Tests (against actual generated files)
# ===========================================================================
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_META_CSV = PROJECT_ROOT / "data" / "sampling" / "site_metadata.csv"
SAMPLED_CSV = PROJECT_ROOT / "data" / "sampling" / "sampled_recordings.csv"
SUMMARY_JSON = PROJECT_ROOT / "reports" / "sampling_summary.json"


def test_site_metadata_csv_structure_and_no_leakage():
    assert SITE_META_CSV.exists(), "site_metadata.csv does not exist"
    with open(SITE_META_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 45, f"Expected 45 sites, got {len(rows)}"

    # Check required columns
    expected_cols = {"site_id", "country", "site_code", "habitat_label", "split", "file_count", "total_bytes", "zip_name", "sha256"}
    assert expected_cols.issubset(rows[0].keys())

    # Check zero leakage across split sets
    split_site_map = {}
    for r in rows:
        sid = r["site_id"]
        assert sid not in split_site_map, f"Duplicate site_id {sid}"
        split_site_map[sid] = r["split"]

    train_sites = {s for s, sp in split_site_map.items() if sp == "train"}
    val_sites = {s for s, sp in split_site_map.items() if sp == "val"}
    test_sites = {s for s, sp in split_site_map.items() if sp == "test"}

    assert len(train_sites.intersection(val_sites)) == 0, "Train-Val leakage detected!"
    assert len(train_sites.intersection(test_sites)) == 0, "Train-Test leakage detected!"
    assert len(val_sites.intersection(test_sites)) == 0, "Val-Test leakage detected!"


def test_all_sites_habitat_coverage_in_manifest():
    with open(SITE_META_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # All 4 habitat labels must exist in the overall dataset
    all_habs = {r["habitat_label"] for r in rows}
    assert all_habs == {"D", "H", "N", "R"}

    # Check split habitat coverage across all 45 sites
    split_habs = {}
    for r in rows:
        split_habs.setdefault(r["split"], set()).add(r["habitat_label"])

    assert split_habs["train"] == {"D", "H", "N", "R"}
    assert split_habs["val"] == {"D", "H", "N", "R"}
    assert split_habs["test"] == {"D", "H", "N", "R"}


def test_sampled_recordings_integrity_and_timestamps():
    assert SAMPLED_CSV.exists(), "sampled_recordings.csv does not exist"
    with open(SAMPLED_CSV, "r", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))

    assert len(recs) == 120, f"Expected 120 pilot recordings, got {len(recs)}"

    # Load site metadata to verify split consistency
    with open(SITE_META_CSV, "r", encoding="utf-8") as f:
        site_splits = {r["site_id"]: r["split"] for r in csv.DictReader(f)}

    for r in recs:
        # Check split consistency
        assert r["split"] == site_splits[r["site_id"]], f"Split mismatch for {r['filename']}"

        # Check timestamp parsing validity
        hour = int(r["hour_of_day"])
        minute = int(r["minute_of_hour"])
        assert 0 <= hour <= 23, f"Invalid hour {hour}"
        assert 0 <= minute <= 59, f"Invalid minute {minute}"

        # Check duty cycle
        if r["country"] == "indonesia":
            assert float(r["duty_cycle_fraction"]) == 0.5
        else:
            assert float(r["duty_cycle_fraction"]) == 0.25

        # Check nominal audio props
        assert int(r["sample_rate_hz"]) == 16000
        assert int(r["channels"]) == 1
        assert int(r["bits_per_sample"]) == 16


def test_pilot_sampled_recordings_limitations_explicit():
    """
    Validates that the pilot sampled recordings reflect true class limitations:
    In the 12-site pilot, the test split contains ONLY 2 sites (kenya_H1, maldives_D1).
    Therefore, the test split in the pilot ONLY contains classes D and H, NOT all four!
    This test verifies that the pipeline and summary accurately document this limitation.
    """
    with open(SAMPLED_CSV, "r", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))

    test_recs = [r for r in recs if r["split"] == "test"]
    test_habs = {r["habitat_label"] for r in test_recs}
    assert test_habs == {"D", "H"}, f"Expected test habs in pilot to be D, H; got {test_habs}"

    # Verify summary JSON explicitly records this limitation
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            summary = json.load(f)
        cov = summary["sampled_recordings_split_distribution"]["habitat_coverage_sampled"]
        assert cov["test"]["all_four_classes_present"] is False
        assert "Missing classes in test sample: ['N', 'R']" in cov["test"]["limitations"]


def test_sampled_recordings_date_and_hour_distribution():
    """
    Validates that the 120 pilot recordings actually span multiple dates and hours
    per site, rather than collapsing into a single day or hour.
    """
    with open(SAMPLED_CSV, "r", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))

    site_dates = {}
    site_hours = {}
    for r in recs:
        sid = r["site_id"]
        site_dates.setdefault(sid, set()).add(r["date_str"])
        site_hours.setdefault(sid, set()).add(int(r["hour_of_day"]))

    for sid in site_dates:
        # Every site in the pilot must span at least 4 distinct dates (except very short deployments like kenya_H1 which has 4 distinct dates in total)
        assert len(site_dates[sid]) >= 4, f"Site {sid} only sampled across {len(site_dates[sid])} dates"
        # Every site must span at least 8 distinct hours across the 24-hr cycle
        assert len(site_hours[sid]) >= 8, f"Site {sid} only sampled across {len(site_hours[sid])} hours"


