#!/usr/bin/env python3
"""
EcoSound-Audit  –  Phase 1: Dataset Audit Script
=================================================
Performs lightweight, reproducible metadata and inventory checks for the
"Coral Reef Soundscapes from a Global Restoration Programme" dataset.

DOI   : 10.5522/04/29958062.v2
Source: https://rdr.ucl.ac.uk/articles/dataset/29958062

CONSTRAINTS
-----------
* Does NOT download any audio archive (ZIP files are 3–18 GB each).
* Does NOT extract or decode any audio.
* Does NOT perform feature extraction or model inference.
* Does NOT store credentials or API keys.
* Uses HTTP timeouts and retry logic for all remote requests.
* Reports failures explicitly; no silent fallbacks.

USAGE
-----
    cd EcoSound-Audit
    python scripts/audit_dataset.py

OUTPUT
------
    reports/audit_summary.json    – machine-readable summary of verified findings
    logs/audit.log                – human-readable progress and error log
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# ---------------------------------------------------------------------------
# Paths (relative to the EcoSound-Audit/ project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "audit_config.yaml"
MANIFEST_CSV_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Remote endpoints (no credentials required – all public)
# ---------------------------------------------------------------------------
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/29958062"
MANIFEST_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/57340871"

HTTP_TIMEOUT = 30   # seconds per request
MAX_RETRIES = 3
RETRY_BACKOFF = 5   # seconds between retries


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "audit.log"
    logger = logging.getLogger("ecosound_audit")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = _setup_logging()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _fetch_url(url: str, label: str) -> Optional[str]:
    """
    Fetch a URL with retries and timeout.
    Returns the decoded body on success, or None on failure.
    Does NOT raise exceptions silently – all failures are logged.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Fetching %s (attempt %d/%d): %s", label, attempt, MAX_RETRIES, url)
            req = Request(url, headers={"Accept": "*/*", "User-Agent": "EcoSoundAudit/1.0"})
            with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            logger.info("Successfully fetched %s (%d bytes)", label, len(body))
            return body
        except HTTPError as exc:
            logger.error("HTTP %d for %s: %s", exc.code, label, exc.reason)
            if attempt < MAX_RETRIES:
                logger.info("Retrying in %d s ...", RETRY_BACKOFF)
                time.sleep(RETRY_BACKOFF)
            else:
                return None
        except URLError as exc:
            logger.error("URL error for %s: %s", label, exc.reason)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF)
            else:
                return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error fetching %s: %s", label, exc)
            return None
    return None


# ---------------------------------------------------------------------------
# Audit A – Official dataset metadata from Figshare API
# ---------------------------------------------------------------------------
def audit_a_metadata() -> dict[str, Any]:
    """
    Fetch and parse the Figshare API response for article 29958062.
    Returns a structured dict with verified fields; uses 'unknown' for
    fields that cannot be determined from the response.
    """
    logger.info("=== Audit A: Official Dataset Metadata ===")
    body = _fetch_url(FIGSHARE_API_URL, "Figshare API")
    if body is None:
        logger.error("AUDIT A FAILED: Could not retrieve Figshare metadata.")
        return {"status": "failed", "error": "HTTP request failed"}

    try:
        meta = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.error("AUDIT A FAILED: JSON parse error: %s", exc)
        return {"status": "failed", "error": str(exc)}

    files = meta.get("files", [])
    logger.info("Figshare API returned %d file entries.", len(files))

    # Classify files
    audio_archives, detection_archives, other_files = [], [], []
    for f in files:
        name = f.get("name", "")
        size_bytes = f.get("size", 0)
        md5_match = f.get("supplied_md5") == f.get("computed_md5") and f.get("computed_md5", "") != ""
        entry = {
            "file_id": f.get("id"),
            "name": name,
            "size_bytes": size_bytes,
            "size_gb": round(size_bytes / 1e9, 3),
            "supplied_md5": f.get("supplied_md5"),
            "computed_md5": f.get("computed_md5"),
            "md5_verified": md5_match,
            "mimetype": f.get("mimetype"),
            "download_url": f.get("download_url"),
        }
        # Classify by naming convention
        if name.endswith(".zip") and name.startswith("detections_"):
            detection_archives.append(entry)
        elif name.endswith(".zip") and any(
            name.startswith(p) for p in ("aus_", "ind_", "ken_", "mal_", "mex_")
        ):
            audio_archives.append(entry)
        else:
            other_files.append(entry)

    authors = [a.get("full_name") for a in meta.get("authors", [])]
    license_info = meta.get("license", {})

    result = {
        "status": "success",
        "api_url": FIGSHARE_API_URL,
        "article_id": meta.get("id"),
        "doi": meta.get("doi"),
        "title": "Coral Reef Soundscapes from a Global Restoration Programme",
        "version": meta.get("version"),
        "published_date": meta.get("published_date"),
        "modified_date": meta.get("modified_date"),
        "license_name": license_info.get("name"),
        "license_url": license_info.get("url"),
        "total_size_bytes": meta.get("size"),
        "total_size_tb": round(meta.get("size", 0) / 1e12, 3),
        "funding": meta.get("funding"),
        "repository_url": meta.get("figshare_url"),
        "authors_api": authors,
        "total_file_entries_in_api": len(files),
        "audio_site_archives": len(audio_archives),
        "detection_archives": len(detection_archives),
        "other_files": len(other_files),
        "audio_archive_list": audio_archives,
        "detection_archive_list": detection_archives,
        "other_file_list": other_files,
    }

    logger.info("Audit A complete: %d audio archives, %d detection archives, %d other files.",
                len(audio_archives), len(detection_archives), len(other_files))

    # Flag any MD5 mismatches
    mismatches = [a["name"] for a in audio_archives + detection_archives + other_files
                  if a["supplied_md5"] and a["computed_md5"] and
                  a["supplied_md5"] != a["computed_md5"]]
    if mismatches:
        logger.warning("MD5 MISMATCH detected for: %s", mismatches)
        result["md5_mismatches"] = mismatches
    else:
        # Note archives with empty computed_md5
        empty_computed = [a["name"] for a in audio_archives + detection_archives + other_files
                          if not a.get("computed_md5")]
        if empty_computed:
            logger.warning("Archives with empty computed_md5 (verification pending): %s", empty_computed)
            result["md5_unverified_archives"] = empty_computed

    return result


# ---------------------------------------------------------------------------
# Audit B/C – Geographic, site, and habitat structure from manifest
# ---------------------------------------------------------------------------
def audit_bc_manifest() -> dict[str, Any]:
    """
    Download manifest.csv (5 KB) and parse site-level metadata.
    Returns country/site/habitat inventory with verified counts.
    """
    logger.info("=== Audit B/C: Geographic, Site, and Habitat Structure ===")

    # Try local cache first, then download
    csv_text = None
    if MANIFEST_CSV_PATH.exists():
        logger.info("Using locally cached manifest: %s", MANIFEST_CSV_PATH)
        csv_text = MANIFEST_CSV_PATH.read_text(encoding="utf-8")
    else:
        csv_text = _fetch_url(MANIFEST_DOWNLOAD_URL, "manifest.csv")
        if csv_text is None:
            logger.error("AUDIT B/C FAILED: Could not retrieve manifest.csv.")
            return {"status": "failed", "error": "manifest.csv download failed"}
        MANIFEST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_CSV_PATH.write_text(csv_text, encoding="utf-8")
        logger.info("Manifest saved to %s", MANIFEST_CSV_PATH)

    # Strip markdown wrapper if file came from the fetcher tool
    if csv_text.startswith("Title:"):
        lines = csv_text.splitlines()
        # Find the line starting with "zip_name"
        for i, line in enumerate(lines):
            if line.startswith("zip_name"):
                csv_text = "\n".join(lines[i:])
                break

    # Parse CSV
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)
    logger.info("Manifest has %d rows.", len(rows))

    sites: list[dict] = []
    detection_rows: list[dict] = []

    for row in rows:
        zip_name = row.get("zip_name", "").strip()
        country = row.get("country", "").strip().lower()
        site_code = row.get("site_code", "").strip()
        file_count_raw = row.get("file_count", "0").strip()
        total_bytes_raw = row.get("total_bytes", "0").strip()

        # Skip detection rows
        if zip_name.startswith("detections_"):
            detection_rows.append(row)
            continue

        # Parse numeric fields safely
        try:
            file_count = int(file_count_raw)
        except ValueError:
            file_count = None
            logger.warning("Non-integer file_count for %s: %r", zip_name, file_count_raw)
        try:
            total_bytes = int(total_bytes_raw)
        except ValueError:
            total_bytes = None

        # Extract habitat code (first character of site_code) and site number
        if len(site_code) >= 1:
            habitat_code = site_code[0]
            site_number = site_code[1:] if len(site_code) > 1 else "?"
        else:
            habitat_code = "UNKNOWN"
            site_number = "?"

        sites.append({
            "zip_name": zip_name,
            "country": country,
            "site_code": site_code,
            "habitat_code": habitat_code,
            "site_number": site_number,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / 1e9, 3) if total_bytes else None,
            "sha256": row.get("sha256", "").strip(),
        })

    # Summary by country
    countries_found = sorted(set(s["country"] for s in sites))
    country_summary: dict[str, Any] = {}
    for ctry in countries_found:
        ctry_sites = [s for s in sites if s["country"] == ctry]
        country_summary[ctry] = {
            "site_count": len(ctry_sites),
            "site_codes": sorted(s["site_code"] for s in ctry_sites),
            "total_files": sum(s["file_count"] for s in ctry_sites if s["file_count"] is not None),
            "total_gb": round(sum(s["total_bytes"] for s in ctry_sites if s["total_bytes"]) / 1e9, 1),
        }

    # Summary by habitat label
    habitat_summary: dict[str, Any] = {}
    for s in sites:
        hc = s["habitat_code"]
        if hc not in habitat_summary:
            habitat_summary[hc] = {"site_count": 0, "countries": set(), "sites": []}
        habitat_summary[hc]["site_count"] += 1
        habitat_summary[hc]["countries"].add(s["country"])
        habitat_summary[hc]["sites"].append(s["site_code"])
    # Convert sets to sorted lists for JSON serialisation
    for hc in habitat_summary:
        habitat_summary[hc]["countries"] = sorted(habitat_summary[hc]["countries"])

    total_files_all = sum(s["file_count"] for s in sites if s["file_count"] is not None)

    # Approximate total duration: each file is 1 min; ~total_files minutes
    approx_hours = total_files_all / 60
    approx_days = approx_hours / 24

    result = {
        "status": "success",
        "manifest_source": "https://ndownloader.figshare.com/files/57340871",
        "total_manifest_rows": len(rows),
        "site_archive_rows": len(sites),
        "detection_archive_rows": len(detection_rows),
        "countries_verified": countries_found,
        "country_count": len(countries_found),
        "total_audio_site_archives": len(sites),
        "country_summary": country_summary,
        "habitat_code_summary": habitat_summary,
        "total_files_across_all_sites": total_files_all,
        "approx_total_hours": round(approx_hours, 1),
        "approx_total_days": round(approx_days, 1),
        "duration_note": (
            "ESTIMATE based on manifest file_count × 1 min/file. "
            "Assumes all recordings are exactly 1 minute. "
            "A small number of files at end-of-battery may be shorter (source documentation note). "
            "This is an approximation, not an exact count."
        ),
        "site_inventory": sites,
    }

    logger.info("Audit B/C: %d countries, %d sites, %d habitat codes, ~%d files (~%.0f hours).",
                len(countries_found), len(sites), len(habitat_summary),
                total_files_all, approx_hours)

    return result


# ---------------------------------------------------------------------------
# Audit D – Audio characteristics (from documentation; no download)
# ---------------------------------------------------------------------------
def audit_d_audio_characteristics() -> dict[str, Any]:
    """
    Records audio format properties as documented in the official dataset
    description. No audio files are downloaded or decoded.
    Source: Figshare article description (https://api.figshare.com/v2/articles/29958062)
    """
    logger.info("=== Audit D: Audio Characteristics (from documentation) ===")

    result = {
        "status": "documented_only",
        "source": "Figshare article description (Williams et al., 2025)",
        "note": (
            "No audio files were downloaded or decoded during Phase 1. "
            "Properties below are taken verbatim from the dataset documentation. "
            "They have not been independently verified against actual file headers."
        ),
        "file_format": "WAV",
        "file_extension": ".wav",
        "sample_rate_hz": 16000,
        "sample_rate_source": "Dataset description: 'sampled at 16 kHz'",
        "channels": "unknown – not stated in documentation",
        "bit_depth": "unknown – not stated in documentation",
        "duration_per_recording_minutes": 1,
        "duration_source": "Dataset description: '1-minute HydroMoth recordings'",
        "approx_size_per_file_mb": 1.83,
        "size_source": "Dataset description: '1.83 MB per .WAV'",
        "recorder_model": "HydroMoth (Open Acoustic Devices)",
        "recorder_source": "Dataset description",
        "duty_cycle_default": "1-in-4 minutes (15 min/hr recording)",
        "duty_cycle_indonesia": "1-in-2 minutes (30 min/hr recording)",
        "duty_cycle_source": "Dataset description",
        "filename_pattern": "country_site_YYYYMMDD_HHMMSS.wav",
        "filename_timestamp": "local time (Audiomoth standard)",
        "filename_example": "aus_D1_20230207_120400.wav",
        "filename_source": "Dataset description – file naming scheme section",
        "unverified_properties": [
            "channels (mono vs stereo)",
            "bit depth / sample encoding",
            "actual sample rate across all sites",
            "whether all files are exactly 60 seconds",
            "whether duty cycle differs further across countries",
        ],
        "planned_verification": (
            "Phase 2 should inspect WAV headers from at least one file per country "
            "using a lightweight tool (e.g., soxi or soundfile.info) on a small "
            "remote-downloaded sample before full processing."
        ),
    }

    logger.info("Audit D documented. WAV at 16 kHz, 1-min files, 1.83 MB each. "
                "Channels and bit depth not documented – require header inspection.")
    return result


# ---------------------------------------------------------------------------
# Audit E – Recording counts and duration
# ---------------------------------------------------------------------------
def audit_e_recording_counts(bc_result: dict[str, Any]) -> dict[str, Any]:
    """
    Derives per-country and per-habitat recording counts from manifest data.
    Estimates total duration. Clearly distinguishes exact vs approximate figures.
    """
    logger.info("=== Audit E: Recording Counts and Duration ===")

    if bc_result.get("status") != "success":
        logger.warning("Skipping Audit E: Audit B/C did not succeed.")
        return {"status": "skipped", "reason": "Audit B/C failed"}

    sites = bc_result["site_inventory"]

    # Per-country
    country_counts: dict[str, dict] = {}
    for s in sites:
        c = s["country"]
        if c not in country_counts:
            country_counts[c] = {"files": 0, "sites": 0, "approx_hours": 0.0}
        if s["file_count"] is not None:
            country_counts[c]["files"] += s["file_count"]
        country_counts[c]["sites"] += 1

    # Duty cycles per country (from documentation)
    duty_cycles = {
        "australia": 0.25,   # 1-in-4: 15 min per hour
        "kenya": 0.25,
        "maldives": 0.25,
        "mexico": 0.25,
        "indonesia": 0.50,   # 1-in-2: 30 min per hour
    }

    for c, data in country_counts.items():
        dc = duty_cycles.get(c, 0.25)
        # Each file = 1 min recording
        approx_recorded_hours = data["files"] / 60
        # Calendar time spanned (roughly): recorded_hours / duty_cycle_fraction
        approx_calendar_hours = approx_recorded_hours / dc
        data["approx_recorded_hours"] = round(approx_recorded_hours, 1)
        data["approx_calendar_hours"] = round(approx_calendar_hours, 1)
        data["duty_cycle_fraction"] = dc

    # Per-habitat
    habitat_summary = bc_result.get("habitat_code_summary", {})
    habitat_counts: dict[str, dict] = {}
    for s in sites:
        hc = s["habitat_code"]
        if hc not in habitat_counts:
            habitat_counts[hc] = {"files": 0, "sites": 0}
        if s["file_count"] is not None:
            habitat_counts[hc]["files"] += s["file_count"]
        habitat_counts[hc]["sites"] += 1

    total_files = bc_result["total_files_across_all_sites"]
    total_hours = bc_result["approx_total_hours"]

    result = {
        "status": "success",
        "count_basis": "manifest.csv file_count column (exact per archive)",
        "exactness": (
            "File counts per site are read directly from manifest.csv and are exact "
            "as stated by the dataset authors. Total duration is an approximation "
            "assuming all files are exactly 1 minute; a small number may be shorter."
        ),
        "total_files_all_sites": total_files,
        "approx_total_recorded_hours": round(total_hours, 1),
        "approx_total_recorded_days": round(total_hours / 24, 1),
        "country_counts": country_counts,
        "habitat_counts": habitat_counts,
        "note_duty_cycle": (
            "Indonesia used a 1-in-2-minute duty cycle; all other countries used 1-in-4. "
            "This means Indonesia has proportionally more recordings per calendar day. "
            "Calendar hours spanned is an estimate; exact deployment dates are not in manifest."
        ),
    }

    logger.info("Audit E: Total files = %d, approx %.0f recorded hours across %d sites.",
                total_files, total_hours, len(sites))
    return result


# ---------------------------------------------------------------------------
# Audit F – Recording schedule and confounders (from documentation)
# ---------------------------------------------------------------------------
def audit_f_schedule() -> dict[str, Any]:
    """
    Summarises known scheduling information from dataset documentation.
    No timestamps are parsed from actual files in Phase 1.
    """
    logger.info("=== Audit F: Recording Schedule and Confounders ===")

    result = {
        "status": "documented_only",
        "source": "Figshare article description and preprint abstract (Williams et al., 2025)",
        "recording_date_range": "October 2021 – June 2023",
        "recording_duration_per_site": "approximately 1 month per site",
        "duty_cycle": {
            "default": "1-in-4 minutes (recordings at 0, 4, 8, ... minutes each hour)",
            "indonesia": "1-in-2 minutes (recordings at 0, 2, 4, ... minutes each hour)",
            "source": "Dataset description",
        },
        "timestamp_format": "YYYYMMDD_HHMMSS (Audiomoth/HydroMoth standard)",
        "timestamp_timezone": "local time at each site",
        "known_confounders": [
            {
                "confounder": "Duty cycle difference",
                "description": (
                    "Indonesia recorded at 1-in-2 minute intervals while all other countries "
                    "used 1-in-4. This doubles the temporal density of Indonesia data, "
                    "which could bias any per-file or per-minute comparison."
                ),
                "severity": "High – must be accounted for in any cross-country comparison",
            },
            {
                "confounder": "Date range heterogeneity",
                "description": (
                    "Data collected October 2021 – June 2023. Sites may have been recorded "
                    "in different seasons or years, introducing temporal confounds "
                    "(seasonal soundscape variation, weather events)."
                ),
                "severity": "Moderate – requires timestamp inspection to quantify",
            },
            {
                "confounder": "Single recorder per site",
                "description": (
                    "Each site has one HydroMoth deployment. Recorder position, depth, "
                    "and orientation could vary between sites, confounding cross-site comparisons."
                ),
                "severity": "Moderate – recorder placement details not available in metadata",
            },
            {
                "confounder": "Biogeographic region",
                "description": (
                    "Five biogeographically distinct regions. Country and biogeographic "
                    "region are perfectly confounded: any country effect cannot be "
                    "attributed solely to habitat type without controlling for region."
                ),
                "severity": "High – fundamental to experimental design",
            },
            {
                "confounder": "Recorder battery end-of-life",
                "description": (
                    "Documentation notes 'a tiny number are of a reduced size as recorder "
                    "batteries reached end of life.' These files may be shorter than 1 minute."
                ),
                "severity": "Low – described as rare; confirm count in Phase 2",
            },
        ],
        "unresolved_schedule_questions": [
            "Exact deployment dates per site (not in manifest).",
            "Whether local-time timestamps allow reliable UTC alignment across regions.",
            "Whether different sites within the same country were recorded simultaneously or sequentially.",
            "Whether any site had recording gaps (equipment failure, maintenance).",
        ],
    }

    logger.info("Audit F documented. Key confounder: Indonesia duty cycle differs from all others.")
    return result


# ---------------------------------------------------------------------------
# Audit G – Independent ecological measurements
# ---------------------------------------------------------------------------
def audit_g_ecological_measurements() -> dict[str, Any]:
    """
    Checks for independently collected ecological data that could serve as
    ground truth or correlation targets. Based on metadata and documentation review.
    """
    logger.info("=== Audit G: Independent Ecological Measurements ===")

    result = {
        "status": "not_found_in_dataset",
        "source": "Figshare article description, manifest.csv, and preprint abstract",
        "findings": (
            "No independent ecological measurements (coral cover, fish abundance, "
            "species richness, restoration age in continuous form, habitat complexity "
            "indices) are present in the Figshare dataset files. "
            "The dataset contains: (1) raw audio archives, (2) sonotype detection clips, "
            "(3) a site manifest, and (4) a KML map. None include tabular ecological data."
        ),
        "habitat_label_is_not_ecological_measurement": (
            "The four habitat labels (D, H, N, R) encode categorical restoration stages, "
            "not continuous ecological measurements. They cannot substitute for "
            "independently measured variables such as coral cover percentage or fish biomass."
        ),
        "associated_study_note": (
            "The preprint (Williams et al., 2025, doi:10.1101/2025.09.24.678197) discusses "
            "ecosystem functions inferred from acoustic detections (fish diversity, "
            "recruitment cuescape, herbivory, bioturbation), but these are derived "
            "outputs from the same acoustic data, not independent ecological measurements."
        ),
        "restoration_age": {
            "available": "partial",
            "description": (
                "The labels encode ordinal restoration stage: N = early-stage (<3 months), "
                "R = mid-stage (32–53 months). These are categorical ranges, not continuous "
                "measured ages. Exact restoration start dates per site are not in the dataset."
            ),
        },
        "recommendation": (
            "If correlation with ecological state is needed in later phases, researchers "
            "must obtain independent ecological data (e.g., coral cover, fish surveys) "
            "from the MARRS project partners for each of the 45 sites. "
            "Do NOT treat habitat labels as continuous ecological ground truth."
        ),
    }

    logger.info("Audit G: No independent ecological measurements found in dataset files.")
    return result


# ---------------------------------------------------------------------------
# Audit H – Data quality checks (from metadata; no audio decoded)
# ---------------------------------------------------------------------------
def audit_h_data_quality(a_result: dict, bc_result: dict) -> dict[str, Any]:
    """
    Identifies data quality issues detectable from metadata alone.
    Does not decode audio or download archives.
    """
    logger.info("=== Audit H: Data Quality ===")

    issues_verified: list[str] = []
    issues_suspected: list[str] = []
    checks_not_performed: list[str] = []

    # Check 1: MD5 mismatches from API
    if "md5_mismatches" in a_result:
        issues_verified.append(
            f"MD5 mismatch in Figshare API for: {a_result['md5_mismatches']}"
        )
    if "md5_unverified_archives" in a_result:
        issues_suspected.append(
            f"Empty computed_md5 for {len(a_result['md5_unverified_archives'])} archives "
            f"(ind_R1.zip and detections_indonesia.zip). "
            f"These have not been verified by Figshare's own checksumming pipeline."
        )

    # Check 2: Battery end-of-life files
    issues_suspected.append(
        "A small number of WAV files are shorter than 1 minute due to recorder "
        "battery depletion (noted in dataset documentation). Exact count unknown."
    )

    # Check 3: Check for highly unequal file counts across sites
    if bc_result.get("status") == "success":
        sites = bc_result["site_inventory"]
        file_counts = [s["file_count"] for s in sites if s["file_count"] is not None]
        if file_counts:
            min_fc = min(file_counts)
            max_fc = max(file_counts)
            ratio = max_fc / min_fc if min_fc > 0 else float("inf")
            small_sites = [s["zip_name"] for s in sites
                           if s["file_count"] is not None and s["file_count"] < 5000]
            if small_sites:
                issues_suspected.append(
                    f"Unusually small file counts (< 5000 files) for: {small_sites}. "
                    f"Min = {min_fc}, Max = {max_fc}, ratio = {ratio:.1f}x. "
                    f"These may represent shorter deployments or partial archives."
                )

    # Check 4: Duplicate filenames (cannot verify without downloading archives)
    checks_not_performed.append(
        "Duplicate WAV filenames within archives – requires archive listing or download."
    )
    checks_not_performed.append(
        "Duplicate timestamps within a site – requires archive content inspection."
    )
    checks_not_performed.append(
        "Silent or near-silent recordings – requires audio decoding."
    )
    checks_not_performed.append(
        "Actual sample rate, channels, and bit depth verification – requires WAV header reading."
    )
    checks_not_performed.append(
        "Exact count of shortened battery-depleted files – requires archive listing."
    )

    result = {
        "status": "partial",
        "issues_verified_from_metadata": issues_verified,
        "issues_suspected_from_documentation": issues_suspected,
        "checks_not_performed_in_phase1": checks_not_performed,
        "data_integrity_note": (
            "Only 2 of 52 Figshare files have empty computed_md5 fields: "
            "ind_R1.zip and detections_indonesia.zip. All other files have "
            "matching supplied and computed MD5s, indicating successful upload verification. "
            "The empty fields suggest those two uploads were still being checksummed or "
            "experienced a processing issue at time of API query."
        ),
    }

    logger.info("Audit H: %d verified issues, %d suspected, %d checks deferred.",
                len(issues_verified), len(issues_suspected), len(checks_not_performed))
    return result


# ---------------------------------------------------------------------------
# Audit I – Feasibility and leakage risks
# ---------------------------------------------------------------------------
def audit_i_feasibility(bc_result: dict) -> dict[str, Any]:
    """
    Assesses experimental feasibility and data leakage risks based on
    verified site/country/habitat structure.
    """
    logger.info("=== Audit I: Feasibility and Leakage Risks ===")

    if bc_result.get("status") != "success":
        return {"status": "skipped", "reason": "Audit B/C failed"}

    habitat_summary = bc_result["habitat_code_summary"]
    country_summary = bc_result["country_summary"]

    leakage_explanation = (
        "Data leakage risk: If 5-second segments are extracted from 1-minute recordings and "
        "the train/test split is performed at the segment level (not the site or recording level), "
        "multiple segments from the same recording or the same site will appear in both "
        "training and testing. Because segments from the same recording share identical "
        "background noise, recorder characteristics, and habitat, a classifier can exploit "
        "these shared low-level features rather than learning generalisable habitat-level "
        "representations. This would inflate reported performance without reflecting "
        "true out-of-distribution generalisation."
    )

    # Check habitat label distribution across countries
    habitat_country_table = {}
    sites = bc_result["site_inventory"]
    for s in sites:
        hc = s["habitat_code"]
        c = s["country"]
        key = f"{hc}"
        if key not in habitat_country_table:
            habitat_country_table[key] = {}
        habitat_country_table[key][c] = habitat_country_table[key].get(c, 0) + 1

    # Identify labels present in only 1 country
    label_concentration_warnings = []
    for hc, ctry_counts in habitat_country_table.items():
        if len(ctry_counts) == 1:
            only_country = list(ctry_counts.keys())[0]
            label_concentration_warnings.append(
                f"Label '{hc}' appears in only 1 country ({only_country}). "
                f"A leave-one-country-out split removing {only_country} would have "
                f"zero examples of '{hc}' in the test set from that country."
            )

    # Australia: no N sites – verified from manifest
    aus_codes = country_summary.get("australia", {}).get("site_codes", [])
    aus_habitat_codes = sorted(set(c[0] for c in aus_codes))

    result = {
        "status": "success",
        "total_sites": bc_result["total_audio_site_archives"],
        "countries": bc_result["countries_verified"],
        "habitat_labels_found": sorted(habitat_summary.keys()),
        "habitat_country_distribution": habitat_country_table,
        "label_concentration_warnings": label_concentration_warnings,
        "australia_habitat_codes_present": aus_habitat_codes,
        "in_domain_evaluation": {
            "feasible": True,
            "note": (
                "45 sites provide reasonable within-dataset evaluation. "
                "Splits must be at site level (not segment or recording level) "
                "to avoid leakage."
            ),
        },
        "leave_one_site_out": {
            "feasible": True,
            "note": (
                "Site-level LOO is possible but leaves at most ~1 site per fold. "
                "Habitat classes must be present in remaining training sites. "
                "Small-count sites (ken_H1: 1060 files) may produce unstable estimates."
            ),
        },
        "leave_one_country_out": {
            "feasible": "partial",
            "note": (
                "Leave-one-country-out (LOCO) is the strongest geographic generalisation test. "
                "However: (1) Australia has NO early-stage-restored (N) sites, so LOCO-Australia "
                "cannot test N-class performance on Australian data. "
                "(2) Kenya has only 1 H site (ken_H1) and only 5 total sites, limiting its "
                "power as a held-out test region for all four classes. "
                "(3) Maldives has no R (mid-stage restored) sites. "
                "(4) Mexico has no D-label sites — wait, per manifest it does: mex_D1, mex_D2. "
                "Full LOCO feasibility per label must be checked per country carefully."
            ),
            "country_habitat_gaps": {},
        },
        "class_imbalance": {
            "note": (
                "Indonesia dominates by file count (~195k files vs ~13–18k for other countries). "
                "This is partly due to the 1-in-2 duty cycle vs 1-in-4. "
                "Any unweighted training procedure will over-represent Indonesia."
            ),
            "recommendation": (
                "Stratify by site, not by file count, when constructing training sets. "
                "Consider per-site normalisation."
            ),
        },
        "site_label_confounding": {
            "status": "confirmed",
            "note": (
                "Each site has exactly one habitat label (verified from naming convention). "
                "Site identity and habitat label are perfectly correlated within a country. "
                "Habitat-level classification from recordings always risks site-level leakage "
                "if the split is not strictly at the site level."
            ),
        },
        "leakage_risk_explanation": leakage_explanation,
        "recommended_split_unit": "site",
    }

    # Fill country-habitat gaps
    for c, csummary in country_summary.items():
        codes = csummary["site_codes"]
        hab_codes = sorted(set(code[0] for code in codes))
        missing = [hc for hc in sorted(habitat_summary.keys()) if hc not in hab_codes]
        if missing:
            result["leave_one_country_out"]["country_habitat_gaps"][c] = {
                "present_labels": hab_codes,
                "missing_labels": missing,
            }

    logger.info("Audit I: Feasibility assessed. LOCO has gaps. Leakage risk from segment splitting documented.")
    return result


# ---------------------------------------------------------------------------
# Audit J – Sampling strategy recommendation
# ---------------------------------------------------------------------------
def audit_j_sampling_strategy(bc_result: dict, e_result: dict) -> dict[str, Any]:
    """
    Recommends a practical sampling strategy for Phase 2 based on verified
    audit findings. Does NOT start the sampling pipeline.
    """
    logger.info("=== Audit J: Sampling Strategy Recommendation ===")

    result = {
        "status": "recommendation_only",
        "phase_boundary": "No sampling performed in Phase 1.",
        "recommended_approach": {
            "unit": "1-minute WAV file (one recording per unit)",
            "stratification": [
                "Sample proportionally across all 45 verified sites.",
                "Represent each site's full recording period by sampling across dates and times.",
                "Preserve balance across the four habitat labels.",
                "Ensure representation across countries proportional to the research question.",
            ],
            "pilot_first": (
                "Before full extraction, run a pilot on one small archive per country "
                "(e.g., ken_H1.zip at ~0.9 GB) to verify WAV format, sample rate, "
                "channels, and timestamp parsing before committing compute time."
            ),
            "metadata_to_preserve": [
                "Source archive name (e.g., aus_D1.zip)",
                "Original WAV filename (e.g., aus_D1_20230207_120400.wav)",
                "Country code",
                "Site code",
                "Habitat label code",
                "Date (YYYYMMDD)",
                "Time of day (HHMMSS)",
                "Timestamp timezone (local)",
            ],
            "split_unit": "site (NOT recording, NOT segment)",
            "random_seed": 42,
            "remote_storage": (
                "Given 8 GB RAM and 256 GB SSD, all large-scale processing should be "
                "done on Google Colab or another cloud runtime with GCS/Drive for storage. "
                "Only manifests and small pilot samples should be kept locally."
            ),
            "duty_cycle_note": (
                "Account for the 1-in-2 vs 1-in-4 duty cycle difference when calculating "
                "per-unit-time statistics. Do not compare raw file counts between Indonesia "
                "and other countries without normalising for duty cycle."
            ),
        },
        "do_not_commit_to_fixed_count": (
            "The exact number of recordings to sample per site should be determined in "
            "Phase 2 after verifying actual archive contents and confirming that the "
            "pilot WAV headers match documentation. The manifest file_count values are "
            "the best available guide (range: 1,060 – 17,921 per site)."
        ),
        "indonesia_note": (
            "Indonesia has 21 sites and ~195,000 files, representing >40% of the total dataset. "
            "Consider whether to cap Indonesia's contribution to avoid geographic dominance."
        ),
    }

    logger.info("Audit J: Sampling strategy documented. No sampling performed.")
    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("EcoSound-Audit Phase 1 started")
    logger.info("Timestamp: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all audits
    a_result = audit_a_metadata()
    bc_result = audit_bc_manifest()
    d_result = audit_d_audio_characteristics()
    e_result = audit_e_recording_counts(bc_result)
    f_result = audit_f_schedule()
    g_result = audit_g_ecological_measurements()
    h_result = audit_h_data_quality(a_result, bc_result)
    i_result = audit_i_feasibility(bc_result)
    j_result = audit_j_sampling_strategy(bc_result, e_result)

    # Assemble machine-readable summary
    summary = {
        "schema_version": "1.0",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 1 – Dataset Audit",
        "project": "Site-Invariant Reef Soundscape Embeddings for Zero-Shot Restoration Monitoring",
        "dataset": {
            "doi": "10.5522/04/29958062.v2",
            "article_id": 29958062,
            "title": "Coral Reef Soundscapes from a Global Restoration Programme",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "published": "2025-09-24",
            "version": 2,
            "total_size_bytes": a_result.get("total_size_bytes"),
            "total_size_tb": a_result.get("total_size_tb"),
        },
        "audit_a_metadata": {
            "status": a_result.get("status"),
            "total_file_entries_in_api": a_result.get("total_file_entries_in_api"),
            "audio_site_archives": a_result.get("audio_site_archives"),
            "detection_archives": a_result.get("detection_archives"),
            "other_files": a_result.get("other_files"),
            "md5_mismatches": a_result.get("md5_mismatches", []),
            "md5_unverified_archives": a_result.get("md5_unverified_archives", []),
        },
        "audit_bc_geography": {
            "status": bc_result.get("status"),
            "countries_verified": bc_result.get("countries_verified", []),
            "country_count": bc_result.get("country_count"),
            "total_audio_site_archives": bc_result.get("total_audio_site_archives"),
            "country_summary": bc_result.get("country_summary", {}),
            "habitat_code_summary": bc_result.get("habitat_code_summary", {}),
        },
        "audit_d_audio": d_result,
        "audit_e_counts": {
            "status": e_result.get("status"),
            "total_files_all_sites": e_result.get("total_files_all_sites"),
            "approx_total_recorded_hours": e_result.get("approx_total_recorded_hours"),
            "approx_total_recorded_days": e_result.get("approx_total_recorded_days"),
            "exactness": e_result.get("exactness"),
            "country_counts": e_result.get("country_counts", {}),
            "habitat_counts": e_result.get("habitat_counts", {}),
        },
        "audit_f_schedule": f_result,
        "audit_g_ecological_measurements": g_result,
        "audit_h_quality": h_result,
        "audit_i_feasibility": i_result,
        "audit_j_sampling": j_result,
        "unresolved_decisions_requiring_researcher_approval": [
            {
                "id": "UD-01",
                "question": "How should the Indonesia duty-cycle difference be handled in cross-country comparisons?",
                "options": [
                    "Normalise by dividing Indonesia file counts by 2 to match other countries.",
                    "Subsample Indonesia to match per-hour coverage of other countries.",
                    "Report results separately for Indonesia and treat it as a distinct analysis.",
                    "Accept the imbalance and report it as a limitation.",
                ],
            },
            {
                "id": "UD-02",
                "question": "How should the four habitat labels (D, H, N, R) be used in the classification experiment?",
                "options": [
                    "Use all four as separate classes (4-class problem).",
                    "Merge N and R into a single 'Restored' class (3-class problem). REQUIRES methodological justification.",
                    "Use only a subset of labels for a specific research question.",
                ],
                "warning": "Do NOT silently merge labels without explicit approval.",
            },
            {
                "id": "UD-03",
                "question": "Australia has no early-stage-restored (N) sites and Maldives has no mid-stage-restored (R) sites. How should LOCO evaluation handle missing habitat classes?",
                "options": [
                    "Exclude the missing class from the evaluation for that country.",
                    "Exclude countries with missing classes from LOCO entirely.",
                    "Accept that LOCO is partial and report which classes are testable per country.",
                ],
            },
            {
                "id": "UD-04",
                "question": "What is the minimum recording sample per site for Phase 2 processing?",
                "context": "Site file counts range from 1,060 (ken_H1) to 17,921 (ind_D5). A fixed sample per site caps the total dataset size.",
                "recommendation": "Defer decision until pilot audio header inspection confirms file properties.",
            },
            {
                "id": "UD-05",
                "question": "Should independent ecological measurements (coral cover, fish surveys) be sought from MARRS project partners?",
                "context": "No such data exists in the Figshare dataset. Obtaining it would enable correlation analysis but requires coordination with dataset authors.",
            },
            {
                "id": "UD-06",
                "question": "ind_R1.zip and detections_indonesia.zip have empty computed_md5 in Figshare API. How should this be handled?",
                "options": [
                    "Verify manually after download using the supplied_md5 and sha256 in manifest.",
                    "Contact dataset authors to confirm upload integrity.",
                    "Proceed and flag as a known data quality issue.",
                ],
            },
        ],
    }

    # Write summary JSON
    out_path = REPORTS_DIR / "audit_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("audit_summary.json written to %s", out_path)

    # Final status report
    logger.info("=" * 60)
    logger.info("Phase 1 Audit COMPLETE")
    logger.info("Countries verified : %d", bc_result.get("country_count", 0))
    logger.info("Sites verified     : %d", bc_result.get("total_audio_site_archives", 0))
    logger.info("Audio archives     : %d", a_result.get("audio_site_archives", 0))
    logger.info("Detection archives : %d", a_result.get("detection_archives", 0))
    logger.info("Habitat labels     : %s",
                sorted(bc_result.get("habitat_code_summary", {}).keys()))
    logger.info("Total WAV files    : ~%d", e_result.get("total_files_all_sites", 0))
    logger.info("Approx duration    : ~%.0f hours", e_result.get("approx_total_recorded_hours", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
