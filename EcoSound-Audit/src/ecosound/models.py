"""
EcoSound-Audit — Core Models and Filename Parsing
=================================================
Strict, validated data structures for soundscape recordings, sites,
and sampling splits across all five countries and four habitat labels (D/H/N/R).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Canonical country mapping
VALID_COUNTRIES = {
    "aus": "australia",
    "ind": "indonesia",
    "ken": "kenya",
    "mal": "maldives",
    "mex": "mexico",
}

COUNTRY_TO_PREFIX = {v: k for k, v in VALID_COUNTRIES.items()}

# Preserved habitat labels
VALID_HABITATS = {"D", "H", "N", "R"}

# Regex pattern for hydrophone filenames: e.g. aus_D1_20230207_120000.WAV
FILENAME_PATTERN = re.compile(
    r"^([a-z]{3})_([A-Z]\d+)_(\d{8})_(\d{6})\.wav$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecordingMetadata:
    """Represents a validated recording entry with extracted acoustic & temporal metadata."""
    filename: str
    source_archive: str
    country: str
    country_code: str
    site_code: str
    site_id: str          # Global unique identifier: e.g. "australia_D1"
    habitat_label: str    # "D", "H", "N", or "R"
    date_str: str         # "YYYY-MM-DD"
    time_str: str         # "HH:MM:SS" (local site time)
    timestamp_iso: str    # "YYYY-MM-DDTHH:MM:SS"
    hour_of_day: int      # 0-23
    minute_of_hour: int   # 0-59
    duty_cycle_fraction: float  # 0.5 for Indonesia (1-in-2 min), 0.25 elsewhere (1-in-4 min)
    sample_rate_hz: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    nominal_duration_s: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_recording_filename(
    filename: str,
    source_archive: Optional[str] = None,
) -> RecordingMetadata:
    """
    Parse and validate a recording filename.
    Raises ValueError if filename does not conform to the project specification.
    """
    clean_name = Path(filename).name.strip()
    match = FILENAME_PATTERN.match(clean_name)
    if not match:
        raise ValueError(f"Invalid recording filename format: {filename}")

    c_prefix, s_code, d_str, t_str = match.groups()
    c_prefix = c_prefix.lower()
    s_code = s_code.upper()

    if c_prefix not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country prefix '{c_prefix}' in filename {filename}")

    country = VALID_COUNTRIES[c_prefix]
    habitat_code = s_code[0]

    if habitat_code not in VALID_HABITATS:
        raise ValueError(f"Invalid habitat label '{habitat_code}' in filename {filename}")

    # Parse date and time
    try:
        dt = datetime.strptime(f"{d_str}_{t_str}", "%Y%m%d_%H%M%S")
    except ValueError as e:
        raise ValueError(f"Invalid date/time components in filename {filename}: {e}") from e

    iso_date = dt.strftime("%Y-%m-%d")
    iso_time = dt.strftime("%H:%M:%S")
    iso_ts = dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Duty cycle
    duty_cycle = 0.5 if country == "indonesia" else 0.25

    # Inferred source archive if not provided
    archive = source_archive or f"{c_prefix}_{s_code}.zip"
    site_id = f"{country}_{s_code}"

    return RecordingMetadata(
        filename=clean_name,
        source_archive=archive,
        country=country,
        country_code=c_prefix,
        site_code=s_code,
        site_id=site_id,
        habitat_label=habitat_code,
        date_str=iso_date,
        time_str=iso_time,
        timestamp_iso=iso_ts,
        hour_of_day=dt.hour,
        minute_of_hour=dt.minute,
        duty_cycle_fraction=duty_cycle,
    )
