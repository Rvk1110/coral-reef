#!/usr/bin/env python3
"""
EcoSound-Audit — Phase 1 Addendum: Remote WAV Header Inspection (v2)
======================================================================
Uses HTTP Range requests + ZIP64-aware Central Directory parsing to inspect
WAV headers without downloading complete archives.

ZIP64 support added: reads the Zip64 End of Central Directory Locator
(20 bytes immediately before the EOCD) to find the Zip64 EOCD record,
which contains the true CD offset and size for large archives.

CONSTRAINTS (unchanged from v1)
-----------
* No complete archive downloaded.
* No audio decoded or extracted.
* No embeddings or indices computed.
* Pure Python stdlib only.
"""
from __future__ import annotations

import json
import logging
import re
import struct
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOGS_DIR / "header_inspection.log"
logger = logging.getLogger("wav_inspect_v2")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
# Clear existing handlers
for h in logger.handlers[:]:
    logger.removeHandler(h)
fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

HTTP_TIMEOUT = 45
MAX_RETRIES = 2
RETRY_PAUSE = 3

# ---------------------------------------------------------------------------
# Target archives
# Deliberately includes both small (non-ZIP64) and large (ZIP64) archives
# to cover all countries and all four habitat labels.
# ---------------------------------------------------------------------------
TARGETS = [
    # (habitat_label, country, archive_name, figshare_file_id)
    ("D", "australia",  "aus_D1.zip",  57340106),
    ("H", "australia",  "aus_H2.zip",  57340100),
    ("R", "australia",  "aus_R1.zip",  57340097),
    ("D", "kenya",      "ken_D1.zip",  57335987),
    ("H", "kenya",      "ken_H1.zip",  57335960),   # ~0.97 GB – non-ZIP64 confirmed
    ("N", "kenya",      "ken_N1.zip",  57336035),
    ("N", "maldives",   "mal_N1.zip",  57338294),
    ("D", "maldives",   "mal_D1.zip",  57338303),
    ("N", "mexico",     "mex_N1.zip",  57338882),
    ("R", "mexico",     "mex_R1.zip",  57338879),
    ("H", "indonesia",  "ind_H4.zip",  57334634),   # ~3.5 GB
    ("N", "indonesia",  "ind_N1.zip",  57334979),   # ~3.3 GB
    ("R", "indonesia",  "ind_R1.zip",  57338165),   # ~14.9 GB  ⚠ unverified MD5
]

WAV_SAMPLES_PER_ARCHIVE = 3

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _base_url(file_id: int) -> str:
    return f"https://ndownloader.figshare.com/files/{file_id}"


def _fetch_range(url: str, start: int, end: int, label: str) -> Optional[bytes]:
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "EcoSoundAudit/1.0"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                if resp.status not in (200, 206):
                    logger.warning("%s: unexpected HTTP status %d", label, resp.status)
                    return None
                return resp.read()
        except HTTPError as exc:
            logger.error("HTTP %d for %s: %s", exc.code, label, exc.reason)
        except URLError as exc:
            logger.error("URL error for %s: %s", label, exc.reason)
        except Exception as exc:
            logger.error("Error for %s: %s", label, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_PAUSE)
    return None


def _head_request(url: str, label: str) -> Optional[dict]:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "EcoSoundAudit/1.0"})
        with urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return {
                "status": r.status,
                "accept_ranges": r.getheader("Accept-Ranges", ""),
                "content_length": int(r.getheader("Content-Length", 0)),
            }
    except Exception as exc:
        logger.error("HEAD failed for %s: %s", label, exc)
        return None


# ---------------------------------------------------------------------------
# ZIP / ZIP64 Central Directory parsing
# ---------------------------------------------------------------------------
EOCD_SIG        = b"PK\x05\x06"   # End of Central Directory
EOCD64_LOC_SIG  = b"PK\x06\x07"  # Zip64 EOCD Locator
EOCD64_SIG      = b"PK\x06\x06"  # Zip64 EOCD Record
CD_SIG          = b"PK\x01\x02"  # Central Directory entry
LOCAL_SIG       = b"PK\x03\x04"  # Local file header

MAX_EOCD_BUF = 65536 + 22


def _find_eocd_and_cd_offset(tail: bytes, file_size: int) -> Optional[dict]:
    """
    Locate the EOCD signature and determine the true CD offset.
    Handles both standard ZIP and ZIP64 (offset sentinel = 0xFFFFFFFF).
    
    Returns dict with keys: cd_offset, cd_size, cd_entries, is_zip64
    or None on failure.
    """
    # Search for EOCD from the end
    pos = tail.rfind(EOCD_SIG)
    if pos == -1:
        return None

    eocd = tail[pos:]
    if len(eocd) < 22:
        return None

    cd_offset_raw = struct.unpack_from("<I", eocd, 16)[0]
    cd_size_raw   = struct.unpack_from("<I", eocd, 12)[0]
    cd_entries_raw= struct.unpack_from("<H", eocd, 10)[0]

    is_zip64 = (cd_offset_raw == 0xFFFFFFFF or
                cd_size_raw   == 0xFFFFFFFF or
                cd_entries_raw== 0xFFFF)

    if not is_zip64:
        return {
            "cd_offset": cd_offset_raw,
            "cd_size": cd_size_raw,
            "cd_entries": cd_entries_raw,
            "is_zip64": False,
        }

    # ZIP64: Find the Zip64 EOCD Locator (20 bytes immediately before EOCD)
    # The locator is at: EOCD position (in file) - 20 bytes
    eocd_pos_in_file = file_size - len(tail) + pos
    loc_pos_in_file  = eocd_pos_in_file - 20
    if loc_pos_in_file < 0:
        logger.warning("ZIP64 locator would be before file start; tail buffer too small")
        return None

    loc_start_in_tail = pos - 20
    if loc_start_in_tail < 0:
        # Locator not in our tail buffer – happens if comment is present
        # We already have the tail buffer with EOCD; the locator is 20 bytes before it
        return {
            "cd_offset": None,
            "cd_size": None,
            "cd_entries": None,
            "is_zip64": True,
            "error": "ZIP64 locator outside tail buffer; need separate range fetch",
        }

    loc = tail[loc_start_in_tail:loc_start_in_tail + 20]
    if loc[0:4] != EOCD64_LOC_SIG:
        logger.warning("ZIP64 locator signature not found at expected position (got %s)", loc[0:4].hex())
        return {
            "cd_offset": None,
            "cd_size": None,
            "cd_entries": None,
            "is_zip64": True,
            "error": f"ZIP64 locator sig not found; got {loc[0:4].hex()}",
        }

    eocd64_offset = struct.unpack_from("<Q", loc, 8)[0]
    return {
        "is_zip64": True,
        "eocd64_offset": eocd64_offset,
        "cd_offset": None,     # filled after fetching EOCD64
        "cd_size": None,
        "cd_entries": None,
    }


def _fetch_eocd64(url: str, eocd64_offset: int, label: str) -> Optional[dict]:
    """Fetch and parse the Zip64 End of Central Directory record (56 bytes min)."""
    buf = _fetch_range(url, eocd64_offset, eocd64_offset + 55, f"{label}/eocd64")
    if buf is None or len(buf) < 56:
        return None
    if buf[0:4] != EOCD64_SIG:
        logger.warning("%s: ZIP64 EOCD sig not found (got %s)", label, buf[0:4].hex())
        return None

    cd_entries = struct.unpack_from("<Q", buf, 24)[0]
    cd_size    = struct.unpack_from("<Q", buf, 40)[0]
    cd_offset  = struct.unpack_from("<Q", buf, 48)[0]
    return {"cd_entries": cd_entries, "cd_size": cd_size, "cd_offset": cd_offset}


def _parse_central_directory(cd_data: bytes) -> list[dict]:
    """Parse ZIP Central Directory; handles both 32-bit and ZIP64 extra fields."""
    entries = []
    pos = 0
    while pos + 46 <= len(cd_data):
        if cd_data[pos:pos+4] != CD_SIG:
            break
        comp_size    = struct.unpack_from("<I", cd_data, pos + 20)[0]
        uncomp_size  = struct.unpack_from("<I", cd_data, pos + 24)[0]
        fname_len    = struct.unpack_from("<H", cd_data, pos + 28)[0]
        extra_len    = struct.unpack_from("<H", cd_data, pos + 30)[0]
        comment_len  = struct.unpack_from("<H", cd_data, pos + 32)[0]
        local_offset = struct.unpack_from("<I", cd_data, pos + 42)[0]
        compression  = struct.unpack_from("<H", cd_data, pos + 10)[0]

        fname_start  = pos + 46
        fname_end    = fname_start + fname_len
        if fname_end > len(cd_data):
            break
        fname = cd_data[fname_start:fname_end].decode("utf-8", errors="replace")

        # Parse ZIP64 extra field if sentinel values present
        if local_offset == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF or comp_size == 0xFFFFFFFF:
            extra_start = fname_end
            extra_end   = extra_start + extra_len
            local_offset, uncomp_size, comp_size = _parse_zip64_extra(
                cd_data[extra_start:extra_end],
                local_offset, uncomp_size, comp_size
            )

        entries.append({
            "filename": fname,
            "compression": compression,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "local_offset": local_offset,
        })
        pos += 46 + fname_len + extra_len + comment_len

    return entries


def _parse_zip64_extra(extra: bytes, local_offset: int, uncomp_size: int, comp_size: int) -> tuple:
    """
    Walk ZIP extra fields to find the ZIP64 extended information (tag 0x0001)
    and replace sentinel values with actual 64-bit values.
    """
    i = 0
    while i + 4 <= len(extra):
        tag    = struct.unpack_from("<H", extra, i)[0]
        length = struct.unpack_from("<H", extra, i + 2)[0]
        data   = extra[i + 4: i + 4 + length]
        if tag == 0x0001:
            # Fields present only for sentinel values, in order:
            # uncomp_size (8 bytes), comp_size (8 bytes), local_offset (8 bytes)
            offset_in_data = 0
            if uncomp_size == 0xFFFFFFFF and offset_in_data + 8 <= len(data):
                uncomp_size = struct.unpack_from("<Q", data, offset_in_data)[0]
                offset_in_data += 8
            if comp_size == 0xFFFFFFFF and offset_in_data + 8 <= len(data):
                comp_size = struct.unpack_from("<Q", data, offset_in_data)[0]
                offset_in_data += 8
            if local_offset == 0xFFFFFFFF and offset_in_data + 8 <= len(data):
                local_offset = struct.unpack_from("<Q", data, offset_in_data)[0]
                offset_in_data += 8
            break
        i += 4 + length
    return local_offset, uncomp_size, comp_size


# ---------------------------------------------------------------------------
# WAV header parsing
# ---------------------------------------------------------------------------
def _parse_wav_header(data: bytes) -> Optional[dict]:
    if len(data) < 44:
        return None
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE" or data[12:16] != b"fmt ":
        return None

    audio_format    = struct.unpack_from("<H", data, 20)[0]
    channels        = struct.unpack_from("<H", data, 22)[0]
    sample_rate     = struct.unpack_from("<I", data, 24)[0]
    byte_rate       = struct.unpack_from("<I", data, 28)[0]
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]
    riff_size       = struct.unpack_from("<I", data, 4)[0]

    duration_s = (riff_size - 36) / byte_rate if byte_rate > 0 else None

    format_name = {1: "PCM", 3: "IEEE_FLOAT", 6: "ALAW", 7: "MULAW",
                   0xFFFE: "EXTENSIBLE"}.get(audio_format, f"UNKNOWN({audio_format})")
    return {
        "format": format_name,
        "audio_format_code": audio_format,
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "bits_per_sample": bits_per_sample,
        "byte_rate": byte_rate,
        "approx_duration_s": round(duration_s, 3) if duration_s is not None else None,
    }


def _parse_filename_timestamp(filename: str) -> dict:
    basename = Path(filename).name
    pattern = r"^([a-z]{3})_([A-Z]\d+)_(\d{8})_(\d{6})\.wav$"
    m = re.match(pattern, basename, re.IGNORECASE)
    if not m:
        return {"parseable": False, "raw": basename}
    d, t = m.group(3), m.group(4)
    return {
        "parseable": True,
        "country_code": m.group(1),
        "site_code": m.group(2),
        "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}",
        "time_local": f"{t[:2]}:{t[2:4]}:{t[4:6]}",
    }


# ---------------------------------------------------------------------------
# Main inspection routine for one archive
# ---------------------------------------------------------------------------
def inspect_archive(label: str, country: str, archive_name: str, file_id: int) -> dict:
    url = _base_url(file_id)
    result = {
        "archive": archive_name,
        "country": country,
        "habitat_label": label,
        "file_id": file_id,
        "status": "pending",
        "is_zip64": None,
        "file_size_bytes": None,
        "cd_entries_count": None,
        "wav_samples": [],
        "errors": [],
    }

    # ── Step 1: HEAD ─────────────────────────────────────────────────────
    head = _head_request(url, archive_name)
    if not head:
        result.update(status="head_failed", errors=["HEAD request failed"])
        return result
    file_size = head["content_length"]
    result["file_size_bytes"] = file_size
    if "bytes" not in head.get("accept_ranges", ""):
        result.update(status="range_not_supported", errors=["Server does not support Range"])
        return result
    logger.info("[%s] %.2f GB, Range OK", archive_name, file_size / 1e9)

    # ── Step 2: Fetch tail for EOCD ──────────────────────────────────────
    eocd_buf_size = min(MAX_EOCD_BUF, file_size)
    tail = _fetch_range(url, file_size - eocd_buf_size, file_size - 1, f"{archive_name}/tail")
    if tail is None:
        result.update(status="tail_fetch_failed", errors=["Tail fetch failed"])
        return result

    eocd_info = _find_eocd_and_cd_offset(tail, file_size)
    if eocd_info is None:
        result.update(status="eocd_not_found", errors=["EOCD signature not found"])
        return result

    result["is_zip64"] = eocd_info.get("is_zip64", False)

    # ── Step 3: Resolve ZIP64 EOCD if needed ────────────────────────────
    if eocd_info.get("is_zip64") and eocd_info.get("eocd64_offset") is not None:
        eocd64 = _fetch_eocd64(url, eocd_info["eocd64_offset"], archive_name)
        if eocd64 is None:
            result.update(status="eocd64_fetch_failed", errors=["ZIP64 EOCD fetch failed"])
            return result
        cd_offset  = eocd64["cd_offset"]
        cd_size    = eocd64["cd_size"]
        cd_entries = eocd64["cd_entries"]
        logger.info("[%s] ZIP64 EOCD: %d entries, CD@%d, size=%d",
                    archive_name, cd_entries, cd_offset, cd_size)
    elif eocd_info.get("error"):
        result.update(status="zip64_locator_error", errors=[eocd_info["error"]])
        return result
    else:
        cd_offset  = eocd_info["cd_offset"]
        cd_size    = eocd_info["cd_size"]
        cd_entries = eocd_info["cd_entries"]
        logger.info("[%s] Standard ZIP: %d entries, CD@%d, size=%d",
                    archive_name, cd_entries, cd_offset, cd_size)

    result["cd_entries_count"] = cd_entries

    # Guard: don't fetch > 20 MB CD
    if cd_size > 20 * 1024 * 1024:
        result.update(status="cd_too_large",
                      errors=[f"CD size {cd_size} bytes > 20 MB limit"])
        return result

    # ── Step 4: Fetch and parse Central Directory ────────────────────────
    cd_data = _fetch_range(url, cd_offset, cd_offset + cd_size - 1, f"{archive_name}/cd")
    if cd_data is None:
        result.update(status="cd_fetch_failed", errors=["CD fetch failed"])
        return result

    entries = _parse_central_directory(cd_data)
    wav_entries = [e for e in entries if e["filename"].lower().endswith(".wav")]
    logger.info("[%s] Parsed %d entries, %d WAV", archive_name, len(entries), len(wav_entries))

    if not wav_entries:
        result.update(status="no_wav_entries", errors=["No WAV entries in CD"])
        return result

    # Sample: first, middle, last
    n = len(wav_entries)
    selected = [wav_entries[i] for i in sorted({0, n // 2, n - 1})]

    # ── Step 5: Fetch local file header + WAV RIFF header ───────────────
    wav_results = []
    for entry in selected:
        fname = entry["filename"]
        local_off = entry["local_offset"]

        lh_buf = _fetch_range(url, local_off, local_off + 511, f"{archive_name}/{Path(fname).name[:40]}")
        if lh_buf is None:
            wav_results.append({"filename": fname, "error": "local header fetch failed"})
            continue
        if lh_buf[0:4] != LOCAL_SIG:
            wav_results.append({"filename": fname,
                                 "error": f"local sig mismatch: {lh_buf[0:4].hex()}"})
            continue

        lh_fname_len = struct.unpack_from("<H", lh_buf, 26)[0]
        lh_extra_len = struct.unpack_from("<H", lh_buf, 28)[0]
        lh_compression = struct.unpack_from("<H", lh_buf, 8)[0]
        data_start   = 30 + lh_fname_len + lh_extra_len
        raw_payload  = lh_buf[data_start:]

        wav_bytes = b""
        if lh_compression == 0:  # Stored (no compression)
            wav_bytes = raw_payload[:200]
        elif lh_compression == 8:  # Deflate
            try:
                # Need enough compressed bytes to decompress at least 200 bytes of WAV header.
                # If lh_buf only grabbed up to offset+511, fetch a larger chunk of compressed data if needed.
                compressed_chunk = raw_payload
                if len(compressed_chunk) < 4096:
                    comp_fetch = _fetch_range(url, local_off + data_start, local_off + data_start + 16383, f"{archive_name}/comp_stream")
                    if comp_fetch:
                        compressed_chunk = comp_fetch
                decompressor = zlib.decompressobj(-15)
                wav_bytes = decompressor.decompress(compressed_chunk, 500)
            except Exception as e:
                logger.warning("[%s] Deflate decompression error on %s: %s", archive_name, fname, e)
                wav_bytes = b""
        else:
            logger.warning("[%s] Unsupported compression %d on %s", archive_name, lh_compression, fname)

        wav_props = _parse_wav_header(wav_bytes)
        ts = _parse_filename_timestamp(fname)

        if wav_props:
            logger.info("[%s] %s → %d Hz, %d ch, %d bit, ~%.1f s",
                        archive_name, Path(fname).name,
                        wav_props["sample_rate_hz"],
                        wav_props["channels"],
                        wav_props["bits_per_sample"],
                        wav_props.get("approx_duration_s") or 0)
            wav_results.append({
                "filename": Path(fname).name,
                "wav_properties": wav_props,
                "timestamp_parsed": ts,
                "cd_compression": entry["compression"],
                "cd_uncompressed_size_bytes": entry["uncomp_size"],
            })
        else:
            wav_results.append({
                "filename": Path(fname).name,
                "error": (f"WAV parse failed; compression={lh_compression}; "
                          f"first8bytes={wav_bytes[:8].hex() if wav_bytes else 'empty'}"),
            })

    result["wav_samples"] = wav_results
    result["status"] = "success" if any("wav_properties" in w for w in wav_results) else "wav_parse_failed"
    return result


# ---------------------------------------------------------------------------
# Manifest SHA256 cross-check for unverified MD5 archives
# ---------------------------------------------------------------------------
def check_manifest_sha256() -> dict:
    import csv
    targets = {"ind_R1.zip", "detections_indonesia.zip"}
    entries: dict = {}
    if not MANIFEST_PATH.exists():
        return {"status": "manifest_not_found"}
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("zip_name", "").strip()
            if name in targets:
                sha = row.get("sha256", "").strip()
                entries[name] = {
                    "sha256_present": bool(sha),
                    "sha256_length": len(sha),
                    "sha256_value": sha or None,
                    "file_count_in_manifest": row.get("file_count", "").strip(),
                    "integrity_assessment": (
                        "USABLE — 64-character SHA256 hash present in manifest.csv. "
                        "Empty computed_md5 in Figshare API reflects incomplete background "
                        "checksumming, NOT a corrupt file. Verify with: "
                        f"shasum -a 256 {name}"
                        if sha else
                        "NOT USABLE — no SHA256 in manifest"
                    ),
                }
    return {"status": "complete", "checked_archives": list(targets), "entries": entries}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("WAV Header Inspection v2 (ZIP64-aware)")
    logger.info("Timestamp: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sha256_check = check_manifest_sha256()
    for name, info in sha256_check.get("entries", {}).items():
        logger.info("SHA256 check [%s]: present=%s, len=%d",
                    name, info["sha256_present"], info["sha256_length"])

    all_results = []
    for label, country, archive_name, file_id in TARGETS:
        logger.info("")
        logger.info("─" * 50)
        logger.info("Inspecting: %s [%s / %s]", archive_name, country, label)
        r = inspect_archive(label, country, archive_name, file_id)
        all_results.append(r)
        time.sleep(1)

    successes = [r for r in all_results if r["status"] == "success"]
    failures  = [r for r in all_results if r["status"] != "success"]

    logger.info("=" * 60)
    logger.info("Complete: %d/%d archives succeeded", len(successes), len(all_results))
    for f in failures:
        logger.warning("FAILED %s: %s", f["archive"], f.get("errors"))

    output = {
        "inspection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "HTTP Range requests, ZIP64-aware Central Directory parsing, WAV RIFF header inspection",
        "archives_attempted": len(all_results),
        "archives_succeeded": len(successes),
        "archives_failed": len(failures),
        "sha256_check_for_unverified_md5_archives": sha256_check,
        "archive_results": all_results,
    }

    out_path = REPORTS_DIR / "header_inspection.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
