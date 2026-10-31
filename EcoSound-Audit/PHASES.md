# PHASES.md — EcoSound-Audit Phase Decision Log

This file records all phase-boundary decisions, option selections, and approval
dates for the EcoSound-Audit project.  
**Update this file** whenever a phase gate is crossed or a major option is selected.

---

## Phase 3.3 Decision — Benchmark Scale Option

| Field | Value |
|---|---|
| **Decision** | **Option A — Full Scale** |
| **TARGET_PER_SITE** | **100 recordings per site** |
| **Total target recordings** | 4,500 (45 sites × 100) |
| **Seed** | 42 |
| **Min duration filter** | 50 s |
| **Segments per recording** | 12 (5 s each, non-overlapping) |
| **Duty-cycle handling (Indonesia)** | `hourly_binning` |
| **Stratification** | 2-D: deployment date × hour-of-day |
| **Config file** | `config/sampling_config_full.yaml` |
| **Decision recorded** | 2026-10-31 |
| **Rationale** | Option A is recommended in PHASE3_PLAN.md §9.2 for maximal temporal coverage and scientific completeness; feature matrix stays <50 MB (safe for local storage). Option B (50/site) is the documented fallback if Colab bandwidth throttles, and can be activated by setting `target_samples_per_site: 50` in the config. |

> **Note:** Option A was already documented as the selected option in
> `PHASE3_PLAN.md §11.1` (written 2026-10-25). This entry formalises it
> as the Phase 3.3 gate decision before Phase 4 begins.

---

## Phase Approval History

| Phase | Description | Status | Date |
|---|---|---|---|
| Phase 1 | Dataset audit & manifest construction | ✅ Complete | — |
| Phase 2 | Site metadata & pilot sampling (120 recordings) | ✅ Complete | — |
| Phase 3.1 | Experimental plan revision | ✅ Complete | 2026-10-25 |
| **Phase 3.3** | **Benchmark scale decision: Option A (100/site)** | **✅ Decided** | **2026-10-31** |
| Phase 4 | Full benchmark dataset preparation (this phase) | 🔄 In Progress | 2026-10-31 |
| Phase 5 | Feature extraction (YAMNet, SurfPerch, Maad) | ⏳ Pending | — |
| Phase 6 | Linear probe evaluation & LOCO | ⏳ Pending | — |

---

## Constraints Carried Into Phase 4

1. No audio downloads. Only ZIP central-directory listings via HTTP Range requests.
2. Pilot files (`data/sampling/sampled_recordings.csv`, `data/sampling/segment_manifest.csv`,
   `data/sampling/site_metadata.csv`, `reports/sampling_summary.json`) must not be modified.
3. Site-level train/val/test split in `data/sampling/site_metadata.csv` is frozen
   (25 train / 8 val / 12 test). No site reassignment.
4. `ind_R1.zip` has no MD5 on Figshare; SHA-256 verification is flagged as pending Phase 5.
5. Do not run any model or feature extraction.
