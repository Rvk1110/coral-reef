# Reef Bioacoustics Pipeline: Evaluation & Implementation Report
## Comparing Ecoacoustic Indices, YAMNet, and SurfPerch for Reef Health Monitoring

**Project:** Site-Invariant Reef Soundscape Embeddings for Zero-Shot Restoration Monitoring  
**Dataset DOI:** `10.5522/04/29958062.v2` (Williams & Jones, 2025; UCL Research Data Repository)  
**Associated Study:** Williams et al. (2025), bioRxiv doi:`10.1101/2025.09.24.678197`  
**Date:** 2026-10-25  

---

## 1. Executive Summary

This report documents the implementation and experimental verification of the reef bioacoustics benchmarking pipeline comparing three acoustic representations across passive acoustic reef recordings:
1. **Traditional Ecoacoustic Indices:** $\mathbb{R}^5$ (ACI, NDSI, ADI, BI, AEI) computed via `scikit-maad` with marine-adapted frequency bands.
2. **YAMNet Foundation Embeddings:** $\mathbb{R}^{1024}$ general-purpose audio representation via TensorFlow Hub.
3. **SurfPerch Domain-Adapted Embeddings:** $\mathbb{R}^{1280}$ marine bioacoustic representation via Kaggle Models.

The pipeline is implemented as a modular Python package adhering strictly to scientific integrity hard rules: zero data leakage (site-grouped cross-validation), no synthetic proxies, standardized linear probe classifiers ($L_2$ Logistic Regression with train-only scaling), and support for both native 4-class (`D`, `H`, `N`, `R`) and 3-class (`Degraded`, `Healthy`, `Restoration`) regimes.

---

## 2. Dataset Audit & Ground Truth Findings

Derived from official metadata, the Figshare API, and `manifest.csv` (see [DATA_AUDIT.md](file:///c:/Users/Administrator/coral-reef/EcoSound-Audit/DATA_AUDIT.md)):

| Attribute | Verified Value in Real Dataset | Citation / Source |
|---|---|---|
| **Total Audio Files** | 542,187 nominal 1-minute WAV files (~9,036 hours) | Figshare API & `manifest.csv` |
| **Number of Sites** | Exactly 45 independent sites across 5 countries | `manifest.csv` |
| **Audio Format** | 16,000 Hz, 1-channel mono, 16-bit PCM WAV | HydroMoth / AudioMoth headers |
| **Geographic Coverage** | 5 countries: Australia (7 sites), Indonesia (21 sites), Kenya (5 sites), Maldives (5 sites), Mexico (7 sites) | Figshare file inventory |
| **Recording Cadence** | Indonesia: 1-in-2 min (duty cycle 0.50); Other 4 countries: 1-in-4 min (duty cycle 0.25) | `DATA_AUDIT.md` Section 4 |
| **Habitat Labels** | **D** (Degraded rubble), **H** (Healthy reference), **N** (Early restoration <3 mo), **R** (Mid-stage restoration 32–53 mo) | `DATA_AUDIT.md` Section 5 |

### Structural Class Gaps across Countries:
- **Australia:** Contains D (3), H (3), R (1). Class N is genuinely absent.
- **Kenya:** Contains D (2), H (2), N (1). Class R is genuinely absent.
- **Maldives:** Contains D (2), H (2), N (1). Class R is genuinely absent.
- **Indonesia & Mexico:** Contain all 4 classes (D, H, N, R).

---

## 3. Preprocessing & Segmentation Pipeline

1. **5-Second Windowing:** Audio is sliced into non-overlapping 5.0-second segments. A nominal 60.0-second recording yields exactly 12 segments.
2. **Master Segment Manifest:** Generated from sampled recordings:
   - Pilot subset: 120 recordings $\times$ 12 segments = **1,440 segments** across 12 sites.
   - Master manifest: `EcoSound-Audit/data/sampling/segment_manifest.csv`.
3. **Resampling per Model:**
   - **Maad Ecoacoustic Indices:** 16 kHz native (Nyquist cutoff 8,000 Hz).
   - **YAMNet:** 16 kHz native (processes 0.96s frames internally, mean-pooled to one 1024-d vector per 5-second segment).
   - **SurfPerch:** Polyphase resampled from 16 kHz to 32 kHz (up=2, down=1), yielding exactly 160,000 samples per 5-second segment.
4. **Deterministic Audio Anomaly Policy:**
   - Files with duration $< 50.0$s are dropped as battery-cutoff anomalies.
   - Files between $50.0$s and $60.0$s are zero-padded to exactly $60.0$s.

---

## 4. Model & Feature Extraction Architecture

### 4.1 Traditional Ecoacoustic Indices ($\mathbb{R}^5$)
- **Library:** `scikit-maad` (verified with PSD mode, FFT length 1024, 50% overlap).
- **Marine Frequency Band Adaptations:**
  - Standard terrestrial settings (up to 11–20 kHz) cannot be applied directly under an 8 kHz Nyquist limit.
  - **Anthrophony Proxy (`flim_antroPh`):** 1,000–2,000 Hz (hydrodynamic noise, boat motor signatures).
  - **Biophony Proxy (`flim_bioPh`):** 2,000–8,000 Hz (snapping shrimp clicks, biological reef choruses).
  - **Features:** ACI (Complexity), NDSI (Soundscape Index $\in [-1.0, 1.0]$), ADI (Diversity), BI (Bioacoustic Index), AEI (Evenness).

### 4.2 YAMNet Foundation Model ($\mathbb{R}^{1024}$)
- Pretrained MobileNet-v1 architecture trained on AudioSet (521 audio classes).
- Ingests 16 kHz mono float32 in $[-1.0, 1.0]$.
- Output pooled via temporal mean pooling across internal framing windows.

### 4.3 SurfPerch Domain-Adapted Model ($\mathbb{R}^{1280}$)
- EfficientNet-B0 backbone trained specifically on marine soundscapes and reef sonotypes.
- Ingests 32 kHz mono float32 ($160,000$ samples per 5.0-second segment).
- Output: 1280-d embedding vector per segment.

---

## 5. Evaluation Methodology & Hard Rules Compliance

### 5.1 Classifier Standardization
- All three representations are evaluated with the identical linear probe family: $L_2$-regularized Logistic Regression (`LogisticRegression(solver='lbfgs', max_iter=1000)`).
- **Zero Data Leakage:** Feature standardization (`StandardScaler`) is fit **strictly on the training split** and transforms test data.
- **Hyperparameter Search:** Regularization parameter $C \in [10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100]$ tuned strictly via inner site-grouped CV on training folds.
- **Class Imbalance:** Handled via `class_weight='balanced'`.

### 5.2 In-Domain Site-Grouped Cross-Validation (RQ-1)
- Grouped strictly by `site_id` using `StratifiedGroupKFold(n_splits=3)`.
- Guarantees zero site leakage: all segments from any given site remain in the same fold.
- Evaluates Macro-F1, Balanced Accuracy, AUROC (OvR macro), per-class Precision/Recall/F1, and Confusion Matrices.
- **Site-Level Diagnostic Aggregation:** Averages segment probability vectors per site with training prior tie-breaking.
- **Site-Level Cluster Bootstrap:** 95% Confidence Intervals computed over site clusters (1,000 iterations).

### 5.3 Geographic Generalization (LOCO / LORO, RQ-3)
- 5 folds (Australia, Indonesia, Kenya, Maldives, Mexico).
- Trains probe on 4 countries, evaluates on the held-out country.
- Measures Macro-F1 drop ($\Delta\text{F1} = \text{Macro-F1}_{\text{in-domain}} - \text{Macro-F1}_{\text{held-out}}$).

### 5.4 Hardware Degradation & Robustness
- Tests performance drops when test data undergoes:
  1. Downsampling to 8 kHz, then upsampled back to model rate.
  2. Bit-depth reduction to 8-bit quantization (256 discrete levels).
  3. Additive Gaussian white noise across SNR levels (+20 dB, +10 dB, 0 dB, -10 dB).

### 5.5 Restoration Trajectory Index (RTI) Ecological Validation
- **RTI Definition:**
  $$\text{RTI}(x) = \frac{(x - c_D) \cdot (c_H - c_D)}{\|c_H - c_D\|^2}$$
  where $c_D$ is the Degraded class centroid and $c_H$ is the Healthy class centroid computed strictly from training data.
- **Compliance Finding:** As confirmed in [DATA_AUDIT.md](file:///c:/Users/Administrator/coral-reef/EcoSound-Audit/DATA_AUDIT.md) Section 5, the public Figshare dataset contains categorical site codes (`D`, `H`, `N`, `R`) but **no continuous benthic surveys (live coral cover %, fish counts)**. Per Section 4.4 hard rules, empirical correlation is explicitly documented as skipped rather than substituting fabricated proxies.

---

## 6. Verification Status & Deviation Log

| Pipeline Component | Status | Verification Evidence |
|---|:---:|---|
| **Dataset Audit** | PASSED | [DATA_AUDIT.md](file:///c:/Users/Administrator/coral-reef/EcoSound-Audit/DATA_AUDIT.md) |
| **Sampling & LOCO Splits** | PASSED | [sampled_recordings.csv](file:///c:/Users/Administrator/coral-reef/EcoSound-Audit/data/sampling/sampled_recordings.csv), [site_metadata.csv](file:///c:/Users/Administrator/coral-reef/EcoSound-Audit/data/sampling/site_metadata.csv) |
| **5-Second Segment Manifest** | PASSED | [segment_manifest.csv](file:///c:/Users/Administrator/coral-reef/EcoSound-Audit/data/sampling/segment_manifest.csv) (1,440 segments) |
| **Unit Test Suite** | PASSED | `pytest tests/test_pipeline_unit.py` (7/7 tests passed in 43.9s) |
| **scikit-maad Extraction** | PASSED | Shape `(5,)`, valid marine NDSI $\in [-1.0, 1.0]$, latency $0.046$s |
| **YAMNet & SurfPerch Modules** | IMPLEMENTED | Modular extractors implemented in `src/ecosound/features.py`; Colab GPU verification notebook ready |
| **Dual Class Mapping** | PASSED | Verified both 4-class (D/H/N/R) and 3-class (Degraded/Healthy/Restoration) |

### Deviation Log:
1. **Pilot Evaluation Scope:** Supervised classifier training was not performed on the 120-recording pilot subset because class D is completely absent from the pilot training split (`train`: H: 10, N: 20, R: 30; D: 0). Training binary or 4-class probes on the pilot would require leaking test/val sites or falsifying splits. Full model training is reserved for the 45-site benchmark on GPU as specified in `PHASE3_PLAN.md`.
2. **NDSI Frequency Bands:** Terrestrial standard bands (1–2 kHz anthrophony, 2–11 kHz biophony) were adapted to 1–2 kHz and 2–8 kHz to respect the physical 8,000 Hz Nyquist cutoff of 16 kHz HydroMoth recordings.
3. **RTI Ecological Correlation:** Skipped with formal audit citation because continuous benthic coral cover measurements do not exist in the public Figshare dataset.
