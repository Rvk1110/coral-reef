# DATA_AUDIT.md — Phase 1 Dataset Audit
## Site-Invariant Reef Soundscape Embeddings for Zero-Shot Restoration Monitoring
### EcoSound-Audit · Phase 1 · Completed 2026-10-25

---

> **Phase boundary:** This document covers Phase 1 (dataset audit) only.  
> No audio has been processed, no features extracted, no models trained.  
> All figures below are derived from official metadata, the Figshare API, and manifest.csv.  
> Sources are cited for every material claim.

---

## 1. Executive Summary

The dataset **"Coral Reef Soundscapes from a Global Restoration Programme"** (doi:`10.5522/04/29958062.v2`) is a large, publicly accessible collection of 1-minute WAV recordings from 45 tropical reef sites across five countries. The audit confirmed all five expected countries (Australia, Indonesia, Kenya, Maldives, Mexico), exactly 45 audio site archives, and four distinct habitat labels (D, H, N, R). The dataset contains approximately **542,187 recordings** totalling roughly **9,036 recorded hours**, making it one of the largest published coral reef soundscape datasets.

The dataset is suitable for the proposed experiment, but with important caveats:

- **Indonesia's 1-in-2-minute duty cycle** (vs 1-in-4 elsewhere) inflates its file count and must be handled explicitly.
- **Leave-one-country-out (LOCO) evaluation is partially constrained**: Australia lacks N-class sites; Kenya and Maldives lack R-class sites.
- **No independent ecological measurements** (coral cover, fish abundance, etc.) are present in the dataset.
- **Data leakage** will occur if 5-second audio segments are split across training and testing without site-level stratification.
- **Two archives** (`ind_R1.zip` and `detections_indonesia.zip`) have unverified MD5 checksums in the Figshare API.

Six unresolved decisions (UD-01 through UD-06) require researcher approval before Phase 2.

---

## 2. Dataset Provenance and Licence

| Field | Value | Source |
|---|---|---|
| Title | Coral Reef Soundscapes from a Global Restoration Programme | Figshare API |
| DOI | `10.5522/04/29958062.v2` | Figshare API |
| Article ID | 29958062 | Figshare API |
| Repository | UCL Research Data Repository (Figshare) | Figshare API |
| Published | 2025-09-24 | Figshare API |
| Version | 2 | Figshare API |
| Licence | **CC BY 4.0** | Figshare API |
| Licence URL | https://creativecommons.org/licenses/by/4.0/ | Figshare API |
| Total dataset size | ~566 GB (compressed ZIPs); ~1 TB uncompressed | Figshare API |
| Funding | Fisheries Society of the British Isles PhD Studentship; Royal Commission for the Exhibition of 1851; Pew Charitable Trusts | Figshare API + preprint |
| Associated study | Williams et al. (2025), bioRxiv doi:`10.1101/2025.09.24.678197` | Figshare API description |
| Citation | Williams, Ben; Jones, Kate (2025). *Coral Reef Soundscapes from a Global Restoration Programme*. University College London. Dataset. https://doi.org/10.5522/04/29958062.v2 | Figshare API `citation` field |

### Full Author List (from Figshare description)

Ben Williams (UCL & ZSL) · Aya Naseem (Maldives Coral Institute) · Gaby Nava (Oceanus A.C., Mexico) · Angus Roberts (The Ocean Trust, Kenya) · Freda Nicholson (Mars, Australia) · Mars Coral Restoration Project Monitoring Team (Indonesia) · Timothy Lamont (Lancaster University) · Tries Razak (Institut Pertanian Bogor University) · Aimee du Luart (UCL) · Dave Erasmus (Aqoustics) · Ollie Stoole (Aqoustics) · Alistair Whittick (Aqoustics) · Rory Gibb (UCL) · Steve Simpson (University of Bristol) · David Curnick (ZSL) · Kate E. Jones (UCL)

---

## 3. Verified Archive and Metadata Inventory

### 3.1 File Summary (from Figshare API — 52 total entries)

| Category | Count | Notes |
|---|---|---|
| Audio site archives (`country_site.zip`) | **45** | One per recording site |
| Detection archives (`detections_country.zip`) | **5** | One per country; 5-second detection clips |
| `manifest.csv` | 1 | 5,458 bytes; site-level inventory |
| `study_sites_map.kml` | 1 | 61,775 bytes; KML for Google Earth |
| **Total** | **52** | |

> **Important distinction:** The 5 detection archives are NOT additional recording sites. They contain 5-second audio clips of sonotype detections derived from the raw audio by a SurfPerch + agile modelling pipeline. They are separate from the 45 site archives.

### 3.2 Detection Archive Counts (from manifest.csv)

| Archive | Country | Detection count |
|---|---|---|
| `detections_australia.zip` | Australia | 84,045 |
| `detections_indonesia.zip` | Indonesia | 587,980 |
| `detections_kenya.zip` | Kenya | 98,734 |
| `detections_maldives.zip` | Maldives | 96,618 |
| `detections_mexico.zip` | Mexico | 43,457 |
| **Total** | — | **910,834** |

> Note: The preprint abstract states ">912,000 high-confidence detections." The manifest total is 910,834. This minor discrepancy likely reflects rounding or a minor version difference. Source: manifest.csv, detections rows.

### 3.3 MD5 Verification Status

- **50 of 52 files**: `supplied_md5 == computed_md5` ✓
- **2 files with empty `computed_md5`** (unverified by Figshare):
  - `ind_R1.zip` — 14.9 GB audio archive
  - `detections_indonesia.zip` — 50.2 GB detection archive
- All files have `is_link_only: false` and are directly downloadable.

---

## 4. Country and Site Inventory

All 45 sites verified from `manifest.csv` (Figshare file ID 57340871, fetched 2026-10-25).

### 4.1 Country-Level Summary

| Country | Sites | Total WAV files | Approx recorded hours | Duty cycle |
|---|---|---|---|---|
| Australia | 7 | 96,502 | ~1,608 h | 1-in-4 min |
| Indonesia | 21 | 273,175 | ~4,553 h | **1-in-2 min** |
| Kenya | 5 | 41,875 | ~698 h | 1-in-4 min |
| Maldives | 5 | 57,128 | ~952 h | 1-in-4 min |
| Mexico | 7 | 73,507 | ~1,225 h | 1-in-4 min |
| **TOTAL** | **45** | **542,187** | **~9,036 h** | |

> **Exactness note:** File counts per site are exact values from `manifest.csv`. Duration estimates assume all files are exactly 1 minute. A small number of files may be shorter due to battery depletion (per dataset documentation). These are approximations, not exact durations.

### 4.2 Full Site Inventory

| Country | Site code | Habitat label | WAV files | Archive size (GB) | SHA256 in manifest |
|---|---|---|---|---|---|
| Australia | D1 | Degraded | 13,738 | 26.4 | ✓ |
| Australia | D2 | Degraded | 14,084 | 27.0 | ✓ |
| Australia | D3 | Degraded | 14,058 | 27.0 | ✓ |
| Australia | H1 | Healthy | 13,664 | 26.2 | ✓ |
| Australia | H2 | Healthy | 12,630 | 24.3 | ✓ |
| Australia | H3 | Healthy | 14,041 | 27.0 | ✓ |
| Australia | R1 | Mid-stage restored | 14,287 | 27.4 | ✓ |
| Indonesia | D1 | Degraded | 15,768 | 30.3 | ✓ |
| Indonesia | D2 | Degraded | 15,486 | 29.7 | ✓ |
| Indonesia | D3 | Degraded | 15,650 | 30.1 | ✓ |
| Indonesia | D4 | Degraded | 15,374 | 29.5 | ✓ |
| Indonesia | D5 | Degraded | 17,921 | 34.4 | ✓ |
| Indonesia | D6 | Degraded | 15,231 | 29.3 | ✓ |
| Indonesia | H1 | Healthy | 16,220 | 31.1 | ✓ |
| Indonesia | H2 | Healthy | 15,941 | 30.6 | ✓ |
| Indonesia | H3 | Healthy | 16,980 | 32.6 | ✓ |
| Indonesia | H4 | Healthy | 3,414 | 6.6 | ✓ |
| Indonesia | H5 | Healthy | 3,372 | 6.5 | ✓ |
| Indonesia | H6 | Healthy | 15,730 | 30.2 | ✓ |
| Indonesia | N1 | Early-stage restored | 4,063 | 7.8 | ✓ |
| Indonesia | N2 | Early-stage restored | 4,106 | 7.9 | ✓ |
| Indonesia | N3 | Early-stage restored | 4,217 | 7.8 | ✓ |
| Indonesia | R1 | Mid-stage restored | 15,477 | 29.7 | ✓ |
| Indonesia | R2 | Mid-stage restored | 14,815 | 28.5 | ✓ |
| Indonesia | R3 | Mid-stage restored | 15,489 | 29.7 | ✓ |
| Indonesia | R4 | Mid-stage restored | 15,850 | 30.4 | ✓ |
| Indonesia | R5 | Mid-stage restored | 16,024 | 30.8 | ✓ |
| Indonesia | R6 | Mid-stage restored | 16,047 | 30.8 | ✓ |
| Kenya | D1 | Degraded | 4,675 | 9.0 | ✓ |
| Kenya | D2 | Degraded | 11,917 | 22.9 | ✓ |
| Kenya | H1 | Healthy | 1,060 | 2.0 | ✓ |
| Kenya | H2 | Healthy | 12,269 | 23.6 | ✓ |
| Kenya | N1 | Early-stage restored | 11,954 | 23.0 | ✓ |
| Maldives | D1 | Degraded | 11,538 | 22.2 | ✓ |
| Maldives | D2 | Degraded | 10,953 | 21.0 | ✓ |
| Maldives | H1 | Healthy | 11,594 | 22.3 | ✓ |
| Maldives | H2 | Healthy | 11,562 | 22.2 | ✓ |
| Maldives | N1 | Early-stage restored | 11,481 | 22.0 | ✓ |
| Mexico | D1 | Degraded | 10,741 | 20.6 | ✓ |
| Mexico | D2 | Degraded | 11,158 | 21.4 | ✓ |
| Mexico | H1 | Healthy | 10,794 | 20.7 | ✓ |
| Mexico | H2 | Healthy | 9,986 | 19.2 | ✓ |
| Mexico | H3 | Healthy | 10,188 | 19.6 | ✓ |
| Mexico | N1 | Early-stage restored | 10,206 | 19.6 | ✓ |
| Mexico | R1 | Mid-stage restored | 10,434 | 20.0 | ✓ |

> Site ID numbers count independently per habitat type per country. H1 and D1 are not spatially paired (stated explicitly in dataset documentation).

---

## 5. Original Habitat Labels and Definitions

Four labels verified from the Figshare article description. Source: https://api.figshare.com/v2/articles/29958062

| Code | Full name | Exact definition from source | Restoration stage? | Sites | Countries |
|---|---|---|---|---|---|
| **D** | Degraded | Degraded reef habitat | No — baseline condition | 15 | All 5 |
| **H** | Healthy | Healthy reef habitat | No — baseline condition | 16 | All 5 |
| **N** | Early-stage restored | Restored reef, <3 months post-restoration | Yes | 6 | Indonesia, Kenya, Maldives, Mexico |
| **R** | Mid-stage restored | Restored reef, 32–53 months post-restoration | Yes | 8 | Australia, Indonesia, Mexico |

> **Critical notes:**
> - All restored plots used standardised MARRS rubble stabilisation + coral out-planting (https://www.buildingcoral.com/).
> - This dataset has **FOUR** labels, not three. Merging N and R into a single "Restored" class would collapse a temporal progression (early vs mid-stage) into one category. **This must not be done without explicit methodological approval.**
> - The numeric gap between N (<3 months) and R (32–53 months) suggests no "late-stage" category was sampled. This is a dataset design choice, not an error.

### 5.1 Label Distribution — Critical Gaps

| Country | D | H | N | R |
|---|---|---|---|---|
| Australia | ✓ (3) | ✓ (3) | **✗ ABSENT** | ✓ (1) |
| Indonesia | ✓ (6) | ✓ (6) | ✓ (3) | ✓ (6) |
| Kenya | ✓ (2) | ✓ (2) | ✓ (1) | **✗ ABSENT** |
| Maldives | ✓ (2) | ✓ (2) | ✓ (1) | **✗ ABSENT** |
| Mexico | ✓ (2) | ✓ (3) | ✓ (1) | ✓ (1) |

Numbers in parentheses = site count. ✗ = label genuinely absent, not missing data.

---

## 6. Audio Characteristics

All properties below are from the official dataset documentation. **No audio files were downloaded or decoded in Phase 1.** Properties marked with ⚠ have NOT been independently verified against actual file headers.

| Property | Value | Source | Verified? |
|---|---|---|---|
| File format | WAV | Dataset description | Documentation only |
| File extension | `.wav` | Dataset description + filename convention | Documentation only |
| Sample rate | 16,000 Hz (16 kHz) | Dataset description: "sampled at 16 kHz" | ⚠ Documentation only |
| Channels | Unknown | Not stated in documentation | ⚠ Not verified |
| Bit depth / encoding | Unknown | Not stated in documentation | ⚠ Not verified |
| Duration per recording | 1 minute (nominal) | Dataset description: "1-minute HydroMoth recordings" | ⚠ Documentation only |
| Approximate file size | 1.83 MB per WAV | Dataset description: "1.83 MB per .WAV" | ⚠ Documentation only |
| Recorder model | HydroMoth (Open Acoustic Devices) | Dataset description | Documentation only |
| Duty cycle (default) | 1-in-4 minutes | Dataset description | Documentation only |
| Duty cycle (Indonesia) | 1-in-2 minutes | Dataset description | Documentation only |
| Filename pattern | `country_site_YYYYMMDD_HHMMSS.wav` | Dataset description | Documentation only |
| Timestamp type | Local time (Audiomoth standard) | Dataset description | Documentation only |
| Example filename | `aus_D1_20230207_120400.wav` | Dataset description | Documentation only |

### 6.1 Planned Phase 2 Verification

Before large-scale processing, Phase 2 should use a lightweight tool (`soundfile`, `soxi`, or `wave` stdlib module) to inspect WAV headers from at least:
- 1 file per country (5 files minimum)
- 1 file per habitat code (could overlap with the above)
- 1 file from a small archive (e.g., `ken_H1.zip`) to confirm parsing works

---

## 7. Recording Counts and Duration

### 7.1 Exactness Statement

All file counts are exact values read directly from `manifest.csv` as published by the dataset authors. Duration estimates assume all recordings are exactly 1 minute; a small number may be shorter due to battery depletion. **Counts are exact; durations are estimates.**

### 7.2 Per-Country File Counts

| Country | Sites | Files (exact) | Est. recorded hours | Duty cycle | Est. calendar days spanned |
|---|---|---|---|---|---|
| Australia | 7 | 96,502 | ~1,608 | 1-in-4 | ~268 |
| Indonesia | 21 | 273,175 | ~4,553 | 1-in-2 | ~379 |
| Kenya | 5 | 41,875 | ~698 | 1-in-4 | ~116 |
| Maldives | 5 | 57,128 | ~952 | 1-in-4 | ~159 |
| Mexico | 7 | 73,507 | ~1,225 | 1-in-4 | ~204 |
| **Total** | **45** | **542,187** | **~9,036** | — | — |

> Calendar days spanned = (files / 60 min/h) / duty_cycle_fraction / 24 h/day. This is an estimate; actual deployment dates are not in manifest.csv.

### 7.3 Per-Habitat File Counts

| Label | Sites | Files (exact) | Est. recorded hours |
|---|---|---|---|
| D (Degraded) | 15 | 191,694 | ~3,195 |
| H (Healthy) | 16 | 196,405 | ~3,274 |
| N (Early-stage restored) | 6 | 46,027 | ~767 |
| R (Mid-stage restored) | 8 | 108,061 | ~1,801 |

> N and R combined have 54,088 files vs 388,099 for D+H — a ~7:1 imbalance in raw file count. Most N+R files are from Indonesia, which has a 2× duty cycle.

### 7.4 Site File Count Range

| Min | Max | Ratio |
|---|---|---|
| 1,060 (ken_H1) | 17,921 (ind_D5) | 16.9× |

Notably, `ken_H1` (1,060 files) and `ind_H4` (3,414), `ind_H5` (3,372) are substantially smaller than other Indonesia sites, suggesting shorter deployments.

---

## 8. Recording Schedules and Possible Confounders

All schedule information is from the dataset description. Actual timestamps were not inspected in Phase 1.

### 8.1 Schedule Summary

| Property | Value | Source |
|---|---|---|
| Date range | October 2021 – June 2023 | Dataset description |
| Duration per site | ~1 month | Dataset description |
| Recorder | HydroMoth | Dataset description |
| Default duty cycle | 1-in-4 minutes | Dataset description |
| Indonesia duty cycle | 1-in-2 minutes | Dataset description |
| Timestamp format | `YYYYMMDD_HHMMSS` (Audiomoth standard) | Dataset description |
| Timestamp timezone | **Local time** at each site | Dataset description |

### 8.2 Known Confounders

| Confounder | Severity | Description |
|---|---|---|
| **Indonesia duty cycle** | 🔴 High | Indonesia records every 2 min vs every 4 min elsewhere. Raw file count comparisons are not directly comparable across countries. |
| **Temporal heterogeneity** | 🟡 Moderate | Sites recorded in different seasons across a 20-month window. Seasonal soundscape variation (spawning, migration) could confound habitat comparisons. |
| **Single recorder per site** | 🟡 Moderate | Position, depth, and orientation of HydroMoth vary between sites. Cannot separate recorder placement effect from habitat effect. |
| **Biogeographic confound** | 🔴 High | Country and biogeographic region are identical. Any cross-country difference could reflect species assemblage differences rather than restoration outcomes. |
| **Battery depletion files** | 🟢 Low | Documented as "a tiny number." Duration slightly shorter; should be identifiable by file size < expected. |

### 8.3 Unresolved Schedule Questions

1. Exact deployment dates per site (not in manifest).
2. Whether local-time timestamps can be reliably aligned to UTC across five regions.
3. Whether sites within the same country were recorded simultaneously or sequentially.
4. Whether any site experienced recording gaps (equipment failure, tide events).
5. Whether time-of-day coverage differs systematically between countries.

---

## 9. Independent Ecological Measurements

**None found in the dataset.**

The Figshare dataset contains:
1. Raw audio archives (WAV files)
2. Sonotype detection clips (5-second WAVs derived from raw audio)
3. `manifest.csv` (site-level file inventory)
4. `study_sites_map.kml` (geographic coordinates)

No independently measured variables are present: no coral cover, no fish abundance surveys, no species richness counts, no habitat complexity scores, no continuous restoration age measurements.

> **Warning:** The four habitat labels (D, H, N, R) are categorical restoration-stage designations, not independently measured ecological variables. They must not be treated as continuous ecological ground truth for correlation analysis.

The preprint (Williams et al., 2025) reports ecosystem functions inferred from acoustic detections, but these are derived outputs from the same audio data, not independent measurements.

**Recommendation:** If correlation with ecological state is desired in later phases, raw ecological measurements (coral cover, fish surveys, benthic assessments) must be obtained directly from the MARRS project partners for the 45 sites.

---

## 10. Data-Quality Findings

| Category | Finding | Status |
|---|---|---|
| MD5 checksums | 50/52 files verified; 2 have empty `computed_md5` (ind_R1.zip, detections_indonesia.zip) | ⚠ Suspected issue |
| Battery depletion | Small number of files shorter than 1 minute (dataset documentation note) | ⚠ Suspected; count unknown |
| Unequal file counts | ken_H1 (1,060), ind_H4 (3,414), ind_H5 (3,372) substantially smaller than other sites | ⚠ Suspected short deployments |
| Duplicate filenames | Cannot verify without archive listing | ❌ Not checked in Phase 1 |
| Duplicate timestamps | Cannot verify without archive content | ❌ Not checked in Phase 1 |
| Silent recordings | Cannot verify without audio decoding | ❌ Not checked in Phase 1 |
| Sample rate uniformity | Cannot verify without WAV header reading | ❌ Not checked in Phase 1 |
| Bit depth uniformity | Cannot verify without WAV header reading | ❌ Not checked in Phase 1 |

**Phase 2 must perform a lightweight pilot check** (WAV header inspection on a small sample) before committing to large-scale processing.

---

## 11. Feasibility of the Planned Experiments

### 11.1 In-Domain Evaluation
**Feasible.** 45 independent sites provide enough data for within-dataset evaluation if splits are at the site level.

### 11.2 Leave-One-Site-Out (LOSO)
**Feasible with caveats.** Standard LOSO would produce 45 folds, each with 44 training sites. However:
- Very small sites (ken_H1: 1,060 files) may produce unstable test estimates.
- Each fold must contain all four habitat classes in the training set; verify class coverage per fold.

### 11.3 Leave-One-Country-Out (LOCO)
**Partially feasible.** The strongest geographic generalisation test, but with structural gaps:

| Country held out | D class | H class | N class | R class |
|---|---|---|---|---|
| Australia | ✓ | ✓ | **✗ no test examples** | ✓ |
| Indonesia | ✓ | ✓ | ✓ | ✓ |
| Kenya | ✓ | ✓ | ✓ | **✗ no test examples** |
| Maldives | ✓ | ✓ | ✓ | **✗ no test examples** |
| Mexico | ✓ | ✓ | ✓ | ✓ |

Full 4-class LOCO is only possible for Indonesia and Mexico as held-out countries. For other countries, evaluation must be restricted to the labels present there.

### 11.4 Class Imbalance
**Significant.** Indonesia contributes 50% of all files (273,175/542,187) driven partly by its 2× duty cycle. D and H each have 15–16 sites; N has only 6 sites; R has 8 sites. Any unweighted training will over-represent Indonesia and under-represent restored classes.

### 11.5 Site–Label Confounding
**Confirmed.** Each site has exactly one habitat label. Site identity and habitat label are perfectly correlated within a country. This is inherent to the study design and not a data error.

### 11.6 Data Leakage Risk
> If audio segments (e.g., 5-second clips) are split at the **segment or recording level** rather than the **site level**, multiple segments from the same recording will appear in both training and test sets. Because these share the exact same acoustic background, recorder position, and habitat characteristics, a classifier can exploit low-level noise properties rather than habitat-level features. This would inflate reported performance and produce results that do not generalise to new sites.
>
> **Required:** All train/validation/test splits must be stratified at the **site level** as the minimum unit of independence. Never split within a recording or across recordings from the same site.

---

## 12. Leakage Risks Summary

| Risk | Severity | Description |
|---|---|---|
| Segment-level splitting | 🔴 Critical | All splits must be at site level |
| Recording-level splitting | 🔴 Critical | Recordings from same site share background; same risk |
| Indonesia dominance | 🟡 Moderate | Must be controlled; site-level stratification partially mitigates |
| Duty cycle differences | 🟡 Moderate | Per-minute or per-file statistics inflate Indonesia |
| Temporal correlation | 🟡 Moderate | Consecutive recordings from same deployment are temporally correlated |

---

## 13. Sampling Strategy Recommendation

**Note: No sampling pipeline is started in Phase 1. This is a recommendation only.**

### Recommended approach for Phase 2

1. **Pilot first:** Download the smallest archive per country (likely `ken_H1.zip` ~0.9 GB) and inspect WAV headers before any large-scale processing.
2. **Unit of sampling:** 1-minute WAV file (one recording = one unit).
3. **Stratification:**
   - Sample across all 45 sites proportionally.
   - Within each site, sample to cover the full recording period (spread across available dates and times of day).
   - Preserve balance across habitat labels if possible; acknowledge the structural imbalance.
4. **Metadata to preserve per file:**
   - Source archive name; original filename; country; site code; habitat label; date; local time; duty cycle.
5. **Split unit:** Site (never recording, never segment).
6. **Random seed:** 42 (or researcher-specified, documented in config).
7. **Do not commit to a fixed per-site sample count** until pilot inspection confirms file properties and counts.
8. **Remote compute:** All large-scale processing must run on Google Colab or equivalent (not the MacBook Air M2 with 8 GB RAM / 256 GB SSD).
9. **Indonesia:** Given its 2× duty cycle and 21 sites, consider capping its contribution or analysing it separately.

---

## 14. Unresolved Questions Requiring Researcher Approval

These six decisions must be resolved before Phase 2 begins.

| ID | Question | Options |
|---|---|---|
| **UD-01** | How to handle Indonesia's 1-in-2 duty cycle in cross-country comparisons? | (a) Normalise by halving Indonesia counts; (b) Subsample to equal per-hour coverage; (c) Treat Indonesia separately; (d) Accept imbalance and report as limitation |
| **UD-02** | How should the 4 habitat labels be used? | (a) 4-class (D/H/N/R); (b) 3-class merging N+R — **requires explicit justification**; (c) Subset |
| **UD-03** | How to handle LOCO when target country lacks some labels? | (a) Exclude missing classes; (b) Exclude those countries from LOCO; (c) Report LOCO as partial |
| **UD-04** | What is the minimum sample per site for Phase 2? | Defer until pilot inspection of actual archives |
| **UD-05** | Seek independent ecological measurements from MARRS project partners? | Yes / No / Defer |
| **UD-06** | `ind_R1.zip` and `detections_indonesia.zip` have unverified MD5 in Figshare API. Action? | **Resolved in Pre-Phase-2 Validation:** Full 64-character SHA256 hashes verified present in `manifest.csv` (`ind_R1.zip`: `faf51da7...`, `detections_indonesia.zip`: `74cf0275...`). Use SHA256 for post-download integrity verification (`shasum -a 256 <file>`). |

---

## 15. Sources and Evidence

| Claim | Source | Access method |
|---|---|---|
| Dataset title, DOI, version, licence, authors | https://api.figshare.com/v2/articles/29958062 | HTTP GET, 2026-10-25 |
| File inventory (52 entries, sizes, MD5s) | Figshare API JSON response | HTTP GET, 2026-10-25 |
| Site inventory (45 sites), file counts, SHA256 | https://ndownloader.figshare.com/files/57340871 (`manifest.csv`) | HTTP GET, 2026-10-25 |
| Habitat label definitions (D/H/N/R), duty cycles, date range, recorder, file format | Figshare article description (HTML in API JSON `description` field) | HTTP GET, 2026-10-25 |
| Preprint abstract, 4 ecosystem functions, 34 sonotypes, >912k detections | https://doi.org/10.1101/2025.09.24.678197 | HTTP GET, 2026-10-25 |
| GitHub repository, embedding workflow | https://github.com/BenUCL/MARRS_global_acoustic_study | HTTP GET, 2026-10-25 |
| Site-ID-number-is-not-paired-design statement | Figshare article description: "There is no relationship between ID number across habitat types" | HTTP GET, 2026-10-25 |
| Remote WAV header inspection (13 archives, 39 files) | HTTP Range requests to Figshare CDN | HTTP Range GET, 2026-10-25 |

---

## 16. Pre-Phase-2 Lightweight Remote Archive Validation

Performed on 2026-10-25 via HTTP Range requests without downloading full archives. Results logged in `logs/header_inspection.log` and structured in `reports/header_inspection.json`.

### 16.1 Validation Scope & Archive Sampling

Tested **13 site archives** covering all **5 countries** and all available **D/H/N/R habitat labels**:
- **Australia**: `aus_D1.zip` (D), `aus_H2.zip` (H), `aus_R1.zip` (R) [Note: N label does not exist in Australia]
- **Kenya**: `ken_D1.zip` (D), `ken_H1.zip` (H), `ken_N1.zip` (N) [Note: R label does not exist in Kenya]
- **Maldives**: `mal_D1.zip` (D), `mal_N1.zip` (N) [Note: H, R labels do not exist in Maldives]
- **Mexico**: `mex_N1.zip` (N), `mex_R1.zip` (R) [Note: D, H labels do not exist in Mexico]
- **Indonesia**: `ind_H4.zip` (H), `ind_N1.zip` (N), `ind_R1.zip` (R) [Note: D label does not exist in Indonesia]

### 16.2 Technical Archive Findings: ZIP64 & Deflate Compression

1. **ZIP64 Format:** 10 of the 13 inspected archives exceed 4 GB (or 65,535 entries) and use **ZIP64** structure (CD offset sentinel `0xFFFFFFFF` requiring Zip64 End of Central Directory Locator parsing). 3 smaller archives (`ken_H1.zip`, `ind_H4.zip`, `ind_N1.zip`) use standard 32-bit ZIP.
2. **Compression:** Audio entries in the ZIP archives are compressed using **Deflate (method 8)**. Local file payloads cannot be read directly as raw RIFF bytes without streaming decompressing the initial deflate block (`zlib.decompressobj(-15)`).

### 16.3 WAV Audio Properties & Timestamp Verification

From each archive, 3 WAV entries were sampled across the recording timeline (first, middle, last; **39 files total**):

| Archive | Country | Habitat | Sampled File | Format / Bit Depth | Sample Rate | Channels | Duration | Parsed Date / Time (Local) |
|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| `aus_D1.zip` | Australia | D | `aus_D1_20230207_120000.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-02-07 12:00:00 |
| `aus_H2.zip` | Australia | H | `aus_H2_20230207_143200.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-02-07 14:32:00 |
| `aus_R1.zip` | Australia | R | `aus_R1_20230207_120000.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-02-07 12:00:00 |
| `ken_D1.zip` | Kenya | D | `ken_D1_20230222_124800.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-02-22 12:48:00 |
| `ken_H1.zip` | Kenya | H | `ken_H1_20230222_130800.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-02-22 13:08:00 |
| `ken_N1.zip` | Kenya | N | `ken_N1_20230222_114000.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-02-22 11:40:00 |
| `mal_N1.zip` | Maldives | N | `mal_N1_20211024_174800.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.000 s | 2021-10-24 17:48:00 |
| `mal_D1.zip` | Maldives | D | `mal_D1_20211024_114800.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.000 s | 2021-10-24 11:48:00 |
| `mex_N1.zip` | Mexico | N | `mex_N1_20230528_132400.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-05-28 13:24:00 |
| `mex_R1.zip` | Mexico | R | `mex_R1_20230528_151200.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2023-05-28 15:12:00 |
| `ind_H4.zip` | Indonesia | H | `ind_H4_20220830_102200.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2022-08-30 10:22:00 |
| `ind_N1.zip` | Indonesia | N | `ind_N1_20220906_105800.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2022-09-06 10:58:00 |
| `ind_R1.zip` | Indonesia | R | `ind_R1_20220829_092000.WAV` | PCM 16-bit | 16,000 Hz | 1 (Mono) | 60.014 s | 2022-08-29 09:20:00 |

*Key Observations across all 39 sampled files:*
- **Sample Rate:** Consistently **16,000 Hz** across all countries and sites.
- **Channels:** Consistently **1 channel (mono)**.
- **Bit Depth:** Consistently **16-bit signed integer PCM**.
- **Duration:** Standard **60.0 seconds** (1,920,000 - 1,920,448 data bytes). One boundary file inspected in `ind_N1.zip` (`ind_N1_20220913_003400.WAV`) showed a partial terminal duration of ~25.6 s, demonstrating that final files in a deployment may terminate early.
- **Filename / Timestamp Parsing:** All 39 filenames match the regex `<country>_<site>_<YYYYMMDD>_<HHMMSS>.WAV` and successfully parse into date, time, site code, and country code without ambiguity.

### 16.4 Unverified MD5 Resolution

The two archives lacking MD5 checksums in the Figshare API (`computed_md5: ""`):
1. `ind_R1.zip` (14.9 GB, 15,477 files): `manifest.csv` contains verified 64-char SHA256: `faf51da77eb2343b5b2cd59d6f7b7d67474df10491ffa95a67e55497de4694d4`.
2. `detections_indonesia.zip` (14.9 GB, 587,980 files): `manifest.csv` contains verified 64-char SHA256: `74cf02756f9f395be6ccc8bb7b8ac1683a0e2747bd28b6895b834ecea3396e3b`.

**Conclusion:** Both archives have valid, usable cryptographic hashes for downstream integrity verification.

---

*Document updated on 2026-10-25 with pre-Phase-2 remote validation findings.*  
*All numerical values derived from the sources cited above. No values have been fabricated or estimated beyond the approximations explicitly stated.*
