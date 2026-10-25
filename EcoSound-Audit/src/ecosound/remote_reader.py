"""
EcoSound-Audit — Remote Archive Central Directory Reader
=========================================================
Lightweight, pure-stdlib remote reader that fetches ZIP Central Directories
via HTTP Range requests without downloading full audio files.
"""
from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

from ecosound.models import RecordingMetadata, parse_recording_filename

logger = logging.getLogger("ecosound_phase2")

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CD_SIG = b"PK\x01\x02"


def fetch_archive_filenames(
    file_id: int,
    archive_name: str,
    user_agent: str = "EcoSoundAudit/1.0",
    max_cd_bytes: int = 25 * 1024 * 1024,
) -> List[str]:
    """
    Fetch the list of file names from a remote Figshare zip archive using HTTP Range requests.
    Supports standard 32-bit ZIP and ZIP64 archives.
    """
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    
    # 1. HEAD request to get file size
    head_req = Request(url, method="HEAD", headers={"User-Agent": user_agent})
    with urlopen(head_req, timeout=30) as resp:
        content_len = resp.headers.get("Content-Length")
        if not content_len:
            raise RuntimeError(f"Could not get Content-Length for {archive_name}")
        file_size = int(content_len)

    # 2. Fetch tail 65KB to locate EOCD
    buf_size = min(65536, file_size)
    range_header = f"bytes={file_size - buf_size}-{file_size - 1}"
    tail_req = Request(url, headers={"Range": range_header, "User-Agent": user_agent})
    with urlopen(tail_req, timeout=30) as resp:
        tail = resp.read()

    eocd_pos = tail.rfind(EOCD_SIG)
    if eocd_pos == -1:
        raise RuntimeError(f"EOCD signature not found in tail of {archive_name}")

    cd_entries = struct.unpack_from("<H", tail, eocd_pos + 10)[0]
    cd_size = struct.unpack_from("<I", tail, eocd_pos + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, eocd_pos + 16)[0]

    # Check for ZIP64
    if cd_offset == 0xFFFFFFFF or cd_entries == 0xFFFF:
        loc_pos = tail.rfind(EOCD64_LOC_SIG, 0, eocd_pos)
        if loc_pos == -1:
            raise RuntimeError(f"ZIP64 locator signature not found for {archive_name}")
        eocd64_offset = struct.unpack_from("<Q", tail, loc_pos + 8)[0]

        # Fetch EOCD64 record (56 bytes)
        req64 = Request(
            url,
            headers={"Range": f"bytes={eocd64_offset}-{eocd64_offset + 55}", "User-Agent": user_agent},
        )
        with urlopen(req64, timeout=30) as resp:
            buf64 = resp.read()

        if buf64[0:4] != EOCD64_SIG:
            raise RuntimeError(f"ZIP64 EOCD signature mismatch for {archive_name}")

        cd_entries = struct.unpack_from("<Q", buf64, 24)[0]
        cd_size = struct.unpack_from("<Q", buf64, 40)[0]
        cd_offset = struct.unpack_from("<Q", buf64, 48)[0]

    if cd_size > max_cd_bytes:
        raise RuntimeError(f"Central Directory size {cd_size} exceeds safety limit {max_cd_bytes}")

    logger.debug("[%s] Fetching CD: %d entries, size=%d bytes at offset %d",
                 archive_name, cd_entries, cd_size, cd_offset)

    # 3. Fetch Central Directory
    cd_req = Request(
        url,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}", "User-Agent": user_agent},
    )
    with urlopen(cd_req, timeout=45) as resp:
        cd_data = resp.read()

    # 4. Parse Central Directory entries
    filenames: List[str] = []
    pos = 0
    while pos + 46 <= len(cd_data):
        if cd_data[pos:pos + 4] != CD_SIG:
            break
        fn_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
        ex_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
        cm_len = struct.unpack_from("<H", cd_data, pos + 32)[0]
        
        fn_bytes = cd_data[pos + 46:pos + 46 + fn_len]
        filename = fn_bytes.decode("utf-8", errors="replace")
        
        # Only retain audio WAV entries
        if filename.lower().endswith(".wav"):
            filenames.append(Path(filename).name)

        pos += 46 + fn_len + ex_len + cm_len

    logger.info("[%s] Successfully retrieved %d WAV filenames from remote CD", archive_name, len(filenames))
    return filenames
