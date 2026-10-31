# PHASE3_PLAN.md — Phase 3 Experimental Plan (Revision 3)
## Site-Invariant Reef Soundscape Embeddings for Zero-Shot Restoration Monitoring: Ecoacoustic Indices vs YAMNet vs SurfPerch
### EcoSound-Audit · Phase 3.1 Planning Document · Final Scientific Revision 2026-10-25

---

> **Phase Boundary & Scope Constraint:**
> This document defines Phase 3.1 ONLY (Experimental Plan Revision).
> No audio files have been downloaded, no features or embeddings have been extracted, and no models have been trained.
> Implementation of Phase 3.2+ will begin strictly upon explicit user approval.
> Existing Phase 2 sampling artifacts (`data/sampling/site_metadata.csv`, `data/sampling/sampled_recordings.csv`, `reports/sampling_summary.json`) are preserved without alteration.

---

## 1. Executive Summary & Terminology Definition

### 1.1 Project Title
**“Site-Invariant Reef Soundscape Embeddings for Zero-Shot Restoration Monitoring: Ecoacoustic Indices vs YAMNet vs SurfPerch”**

### 1.2 Defining "Zero-Shot" in this Research Framework
In mainstream machine learning, "zero-shot learning" strictly denotes classification where test categories have zero labeled examples during training (e.g. through textual semantic embeddings). 

In this bioacoustics and environmental monitoring project, we explicitly distinguish three distinct evaluation paradigms to maintain scientific precision:

1. **Frozen-Embedding Transfer Evaluation (Supervised Linear Probe):**
   - The pretrained foundation audio backbones (YAMNet, SurfPerch) and ecoacoustic index algorithms remain completely frozen (no end-to-end backpropagation or weight fine-tuning).
   - A lightweight supervised classifier (linear probe: $L_2$-regularized Logistic Regression / Linear SVM) is trained on labeled feature representations from training sites.
2. **Zero-Shot Site-Level Transfer (Cross-Site Generalization):**
   - The probe is evaluated on unseen soundscapes from entirely separate geographical recording sites within the same biogeographic regions. The classifier has had **zero exposure** to the test site's acoustic environment, bathymetry, depth, or recorder self-noise.
3. **Zero-Shot Geographic/Regional Transfer (Cross-Country Generalization via LOCO):**
   - The probe is trained on data from four countries and evaluated on a fifth held-out country. The model has had **zero training exposure** to that biogeographic region, its specific species assemblages, or its regional environmental soundscape characteristics.

> **Key Takeaway:** The term *"Zero-Shot Restoration Monitoring"* in our title refers to **Zero-Shot Site and Regional Generalization** using frozen pretrained embeddings without site-specific calibration data. It does **not** mean unsupervised classification without labeled training sites.

### 1.3 Core Testable Research Questions (RQs)
- **RQ-1 (In-Domain Representation Capacity under Site-Grouped CV):** When evaluated on unseen sites using site-grouped cross-validation across the 25 training sites (where all recordings from any site remain strictly in the same fold, guaranteeing zero site leakage), what is the upper-bound habitat discrimination performance for Ecoacoustic Indices ($\mathbb{R}^5$), YAMNet ($\mathbb{R}^{1024}$), and SurfPerch ($\mathbb{R}^{1280}$)?
- **RQ-2 (Site-Invariant Zero-Shot Generalization):** When evaluated on held-out test sites (**12 test sites in the full benchmark**), how much performance degradation occurs relative to RQ-1? Does domain-adapted pretraining on reef soundscapes (SurfPerch) yield significantly higher site-invariance than general-audio pretraining (YAMNet) or handcrafted ecoacoustic indices?
- **RQ-3 (Cross-Country Zero-Shot Transfer / LOCO):** When transferring models across entire biogeographic regions (Leave-One-Country-Out), do acoustic representations retain meaningful discrimination between degraded and healthy soundscapes, or do regional biogeographic differences overpower habitat signals?
- **RQ-4 (Feature Parsimony & Complementarity):** Does the low-dimensional handcrafted ecoacoustic feature space ($\mathbb{R}^5$) provide complementary predictive signal to the high-dimensional neural embeddings ($\mathbb{R}^{1024}$ / $\mathbb{R}^{1280}$)?

---

## 2. Dataset Structure, Ground Truth Distinctions & Known Confounders

### 2.1 Geographic and Label Inventory
The dataset (*Williams & Jones, 2025; DOI: 10.5522/04/29958062.v2*) spans **5 countries**, **45 independent sites**, and **542,187 nominal 1-minute audio recordings** (~9,036 recorded hours):

| Country | Biogeographic Region | Sites | Total Files in Manifest | Recording Duty Cycle | Habitat Classes Present |
|---|---|:---:|:---:|:---:|---|
| **Australia** | Great Barrier Reef | 7 | 96,502 | 1-in-4 min | D (3), H (3), R (1) — *N genuinely absent* |
| **Indonesia** | Coral Triangle | 21 | 273,175 | 1-in-2 min | D (6), H (6), N (3), R (6) — *All 4 present* |
| **Kenya** | Western Indian Ocean | 5 | 41,875 | 1-in-4 min | D (2), H (2), N (1) — *R genuinely absent* |
| **Maldives** | Central Indian Ocean | 5 | 57,128 | 1-in-4 min | D (2), H (2), N (1) — *R genuinely absent* |
| **Mexico** | Mesoamerican Reef | 7 | 73,507 | 1-in-4 min | D (2), H (3), N (1), R (1) — *All 4 present* |
| **Total** | **5 Regions** | **45** | **542,187** | — | **D (15), H (16), N (6), R (8)** |

### 2.2 Critical Distinction: Acoustic Habitat Discrimination vs. Ecological Restoration Success
> [!IMPORTANT]
> **Scientific Integrity Notice:**
> 1. The four labels (**D**, **H**, **N**, **R**) represent categorical site deployment categories (Degraded rubble fields, Healthy reference reefs, Early-stage restored plots <3 months post-restoration, Mid-stage restored plots 32–53 months post-restoration).
> 2. **They are NOT continuous ecological measurements.** No raw benthic surveys, coral cover percentages, fish biomass counts, or rugosity metrics exist in the public Figshare dataset.
> 3. Successfully classifying audio recordings into D/H/N/R demonstrates **acoustic habitat discrimination**, NOT proof of ecological restoration success. Acoustic features may reflect hydrophone depth, substrate acoustic reflectivity, boat traffic, or snapping shrimp density rather than ecological recovery. All project conclusions must maintain this distinction explicitly.

### 2.3 Structural Class Gaps across Countries
Because Australia lacks class N, and Kenya and Maldives lack class R, **country-held-out (LOCO) evaluation cannot evaluate all 4 classes in every test country**. 
- In LOCO folds for Australia, Kenya, and Maldives, test evaluation must be restricted to the subset of classes present in that held-out country, and reported with class support and fold-specific class coverage. Fold macro-F1 scores cannot be directly averaged without documenting the differing class denominators.

---

## 3. Two-Stage Experimental Strategy: Pilot vs. Full Benchmark

We explicitly separate the 120-recording pilot technical verification from the comprehensive 45-site benchmark to prevent invalid scientific claims:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: 120-RECORDING PILOT (Pipeline, Preprocessing & Storage Verification)│
├─────────────────────────────────────────────────────────────────────────────┤
│ • 12 representative sites (10 recordings/site = 120 files).                │
│ • Purpose: Verify remote streaming, audio decoding, resampling, & shapes.   │
│ • Split Note: Train has 0 D files; Val has 0 R files; Test has 0 N & 0 R.   │
│ • Scope: Restricted STRICTLY to technical pipeline & storage verification.  │
│ • Constraint: ZERO model training or classification evaluation on pilot.    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Verification Gates Passed
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: FULL 45-SITE BENCHMARK (Scientifically Valid 4-Class Generalization)│
├─────────────────────────────────────────────────────────────────────────────┤
│ • All 45 sites represented across 5 countries (4,500 recordings).           │
│ • Strict Site-Level Splits (from site_metadata.csv):                         │
│     - Train: 25 sites (D: 8, H: 8, N: 3, R: 6) — 3-fold Site-Grouped CV     │
│     - Val:    8 sites (D: 2, H: 3, N: 2, R: 1) — for probe tuning ONLY      │
│     - Test:  12 sites (D: 5, H: 5, N: 1, R: 1) — held-out final benchmark   │
│ • Severe Small-Sample Caveat: Test set has only 1 N site and 1 R site.      │
│ • Scope: Genuine 4-class evaluation on unseen sites across all 4 categories. │
│ • Cross-Country LOCO: 5 folds with explicit class support reporting.        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Stage 1: The 120-Recording Pilot (Pipeline, Preprocessing & Storage Verification Only)
- **Sample Allocation:** 12 sites $\times$ 10 recordings = 120 recordings (from Phase 2 `sampled_recordings.csv`).
- **Actual Split Composition & Inherent Classification Impossibility:**
  - `train`: 6 sites / 60 recordings — classes present: H (10), N (20), R (30). **Class D is completely missing (0 recordings).**
  - `val`: 4 sites / 40 recordings — classes present: D (10), H (10), N (20). Class R is missing.
  - `test`: 2 sites / 20 recordings — classes present: D (10), H (10). Classes N and R are missing.
- **Resolution of Pilot D-vs-H Contradiction:**
  - Because class D is completely absent from the pilot training split, it is mathematically and scientifically impossible to train a supervised D-vs-H binary classifier without violating site-level separation (e.g. pooling splits or leaking validation/test sites into training).
  - Therefore, we **strictly restrict the 120-recording pilot to technical pipeline, preprocessing, output-shape, and storage verification**.
  - **Exact Pilot Verification Scope:**
    1. *Remote I/O:* Verify HTTP Range byte streaming and ZIP extraction from Figshare remote archives on Google Colab.
    2. *Preprocessing & Resampling:* Verify 16 kHz native decoding, 32 kHz polyphase resampling, and duration handling.
    3. *Output Dimensionality & Numerical Integrity:* Verify exact output tensor shapes: Maad $(5,)$, YAMNet $(1024,)$, and SurfPerch $(1280,)$ with zero NaNs or Infs.
    4. *Storage Footprint & Profiling:* Verify that 120 feature representations export cleanly to `.parquet`/`.npz` ($<2\text{ MB}$) and record preliminary per-file extraction latency.
  - **Strict Prohibition:** **Zero supervised model training, hyperparameter fitting, or classification evaluation (neither 4-class nor binary D-vs-H) will be performed on the 120-recording pilot.** All classification experiments are reserved exclusively for Stage 2.

### 3.2 Stage 2: The Full 45-Site Benchmark (4-Class Evaluation & Severe Small-Sample Uncertainty)
- **Mandate:** All 45 sites must be sampled and represented ($N_{\text{target}} = 100$ recordings/site = 4,500 recordings total).
- **Split Separation (from `site_metadata.csv`):**
  - `train` (25 sites): D (8), H (8), N (3), R (6) — used strictly to fit the linear probe classifiers.
  - `val` (8 sites): D (2), H (3), N (2), R (1) — used exclusively for probe hyperparameter tuning (regularization parameter $C$).
  - `test` (12 sites): D (5), H (5), N (1), R (1) — held-out final benchmark; test sites are strictly excluded from all model selection and tuning.
- **Critical Small-Sample Uncertainty Warning on Test Set:**
  > [!CAUTION]
  > **High Statistical Uncertainty for Classes N and R in Test Split:**
  > While all 4 classes are formally represented in the 12-site test set, **class N contains exactly 1 site (`mexico_N1`) and class R contains exactly 1 site (`mexico_R1`)**.
  > 1. **Clustered Pseudoreplicates, Not Independent Replicates:** Recordings from the same site share that site's specific bathymetry, water depth, tidal currents, and recorder placement. They are clustered pseudoreplicates, NOT independent ecological replicates.
  > 2. **Extreme Metric Variance:** Because per-class performance on N and R depends entirely on a single physical site each, metrics will exhibit extreme sensitivity to single-site idiosyncrasies.
  > 3. **Bootstrap Instability:** Site-level cluster bootstrap resampling will be highly unstable for N and R: any bootstrap resample omitting `mexico_N1` or `mexico_R1` leaves that class completely absent, causing zero or undefined recall/F1.
  > 4. **No Strong Four-Class Evidence:** We **must not** claim strong four-class statistical evidence or generalizable restoration monitoring success from this test split alone. Test results for N and R must be reported with their wide uncertainty bounds, explicitly designated as preliminary, and supplemented by the within-Indonesia analysis (which contains 3 N sites and 6 R sites).

---

## 4. Sampling Strategy, Confounders & Indonesia Duty-Cycle Correction

### 4.1 What the Current Sampling Strategy Corrects
1. **Numeric Volume Bias:** By setting a uniform per-site sampling quota ($N_{\text{target}}$), Indonesia's 21 sites do not contribute 2× more audio volume per site than other regions, despite Indonesia's 1-in-2 min duty cycle.
2. **Burst Clustering Mitigation:** 2D stratification across available calendar deployment dates and the 24 diurnal hours (`hour_of_day`: 0–23) ensures samples are spread across morning, afternoon, evening, and night, rather than clumped in a single day or hour.

### 4.2 What the Sampling Strategy Does NOT Correct (Known Confounders)
> [!WARNING]
> We explicitly acknowledge the following persistent confounders that sampling cannot eliminate:
> 1. **Recording Cadence Disparity:** Indonesia files were natively recorded at 2-minute cadence; files in other countries were recorded at 4-minute cadence. Consecutive recordings in Indonesia remain closer in time than in other countries.
> 2. **Macro-Seasonal Heterogeneity:** Sites were recorded asynchronously across a 20-month project window (October 2021 to June 2023), typically for ~1 month per site. Seasonal differences (monsoon vs dry season, fish spawning peaks) are confounded with geographic region.
> 3. **Biogeographic Confounding:** Country and biogeographic region are identical in this dataset. Country differences may reflect distinct species assemblages (e.g. Great Barrier Reef vs Caribbean Mexico) rather than restoration quality.
> 4. **Hardware Placement:** Single HydroMoth recorder per site; depth and orientation variation cannot be disentangled from site effects.

### 4.3 Duty Cycle Sensitivity Analysis (`subsample_50pct`)
To evaluate whether Indonesia's 2-minute recording interval skews feature representations:
- **Baseline Protocol:** Equal per-site quota with 2D `(date, hour)` sampling.
- **Sensitivity Check:** An identical evaluation where Indonesia candidate files are strictly pre-filtered to a 4-minute grid (`minute % 4 == 0`), matching the temporal spacing of the other 4 countries.
- Results will be compared side-by-side to determine if recording cadence influences classification performance.

---

## 5. Three Standardized Representation Pipelines

All three pipelines ingest identical 1-minute audio segments and extract fixed-dimensional representations:

```
                  ┌──────────────────────────────────────────────┐
                  │ 1-minute WAV Recording (16,000 Hz, 1-ch PCM) │
                  └──────────────────────┬───────────────────────┘
                                         │
         ┌───────────────────────────────┼──────────────────────────────┐
         ▼                               ▼                              ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│ Pipeline A       │           │ Pipeline B       │           │ Pipeline C       │
│ Ecoacoustic      │           │ YAMNet           │           │ SurfPerch        │
│ Indices (Maad)   │           │ Foundation Model │           │ Bioacoustic Model│
├──────────────────┤           ├──────────────────┤           ├──────────────────┤
│ • Power Spectro. │           │ • Native 16 kHz  │           │ • Resample 32 kHz│
│ • ACI, NDSI,     │           │ • 0.96s windows  │           │ • 5.0s windows   │
│   ADI, BI, AEI   │           │ • Mean pooling   │           │ • Mean pooling   │
│ • Dim: 5         │           │ • Dim: 1,024     │           │ • Dim: 1,280     │
└────────┬─────────┘           └────────┬─────────┘           └────────┬─────────┘
         │                              │                              │
         └──────────────────────────────┼──────────────────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────────┐
                  │ Standardized Common Classifiers              │
                  │ • Dummy (Majority Class Baseline)            │
                  │ • Linear Probe: Logistic Reg (L2) / SVM      │
                  │ • Non-Linear Probe: Random Forest            │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Common Evaluation: Holdout, LOSO, LOCO       │
                  │ • Macro-F1, Balanced Acc, Per-Class Recall   │
                  │ • Country-by-Country Support & Confusion Mat.│
                  │ • Site-Level Diagnostic Voting (with tie brk)│
                  │ • Cluster Bootstrap 95% Confidence Intervals │
                  └──────────────────────────────────────────────┘
```

### 5.1 Pipeline A: Traditional Ecoacoustic Indices (Baseline)
- **Library:** `scikit-maad` (pure Python soundscape ecoacoustics).
- **Input Audio:** 60.0s audio @ 16,000 Hz ($960,000$ samples), 1-channel PCM.
  - **Nyquist Frequency:** $f_{\text{nyquist}} = 16,000 / 2 = 8,000\text{ Hz}$. All spectral upper bounds must remain strictly $\le 8,000\text{ Hz}$.
- **Spectrogram Parameters:**
  - Window: Hann, length $N = 1024$, overlap $50\%$ (512 samples), $\Delta f = 15.625\text{ Hz}$, $\Delta t = 32\text{ ms}$.
  - Mode: Power Spectral Density (`mode='power'`).
- **Standardized Acoustic Indices & Frequency Bands (5 features):**
  1. **ACI (Acoustic Complexity Index):** Intensity variability across frequency bins (`maad.features.acoustic_complexity_index`).
  2. **NDSI (Normalized Difference Soundscape Index):** Ratio of biophony to anthropophony (`maad.features.soundscape_index`):
     $$\text{NDSI} = \frac{\text{Biophony} - \text{Anthrophony}}{\text{Biophony} + \text{Anthrophony}}$$
     - **Proposed Frequency Bands:** Following the standard ecoacoustic formulation by Kasten et al. (2012) adapted for our 16 kHz sample rate (8,000 Hz Nyquist cutoff):
       - Anthrophony proxy band: 1,000–2,000 Hz (`flim_antro=(1000, 2000)`).
       - Biophony proxy band: 2,000–8,000 Hz (`flim_bio=(2000, 8000)`). (Note: Terrestrial standard extends up to 11 kHz or 20 kHz, but under an 8 kHz Nyquist limit, 8,000 Hz is the physical upper bound; low-frequency boat and hydrodynamic noise falls into the anthrophony proxy).
     - **Verification Status: Pending Implementation Verification.**
       - The exact parameter names (`flim_antro`, `flim_bio` vs alternate keywords), argument ordering, and internal spectrogram frequency-bin integration in `maad.features.soundscape_index` are **proposed settings, not yet verified against the installed `scikit-maad` environment**.
       - *Verification Requirement:* These settings remain pending until confirmed via a standalone single-file smoke test in the Phase 3.2 environment. We do NOT assume the current ranges or parameter names are verified until the smoke test executes successfully and confirms expected output bounds ($\text{NDSI} \in [-1.0, +1.0]$).
  3. **ADI (Acoustic Diversity Index):** Shannon diversity across 1 kHz frequency bins (`maad.features.acoustic_diversity_index(..., fmax=8000, bin_step=1000)` — proposed parameters, pending smoke test verification).
  4. **BI (Bioacoustic Index):** Area under the mean dB curve (`maad.features.bioacoustics_index(..., flim=(2000, 8000))` — proposed parameters, pending smoke test verification).
  5. **AEI (Acoustic Evenness Index):** Gini evenness across 1 kHz frequency bins (`maad.features.acoustic_eveness_index(..., fmax=8000, bin_step=1000)` — proposed parameters, pending smoke test verification).
- **Output:** Feature vector $x_{\text{maad}} \in \mathbb{R}^5$.

### 5.2 Pipeline B: General-Domain Foundation Embeddings (YAMNet)
- **Source:** TensorFlow Hub (`https://tfhub.dev/google/yamnet/1`).
- **Architecture:** MobileNet-v1 depthwise separable convolutions trained on AudioSet (521 classes).
- **Input Specifications:** 16,000 Hz mono float32 waveform in $[-1.0, +1.0]$ (native match to dataset; zero resampling).
- **Windowing & Pooling:**
  - Processes overlapping $0.96\text{s}$ patches every $0.48\text{s}$ (64 mel bins).
  - A 60-second file yields ~124 frame embeddings of shape $(124, 1024)$.
  - **Pooling:** Global temporal average pooling:
    $$x_{\text{yamnet}} = \frac{1}{T} \sum_{t=1}^T e_t \in \mathbb{R}^{1024}$$

### 5.3 Pipeline C: Domain-Adapted Bioacoustic Embeddings (SurfPerch)
- **Source:** Kaggle Models (`https://www.kaggle.com/models/google/surfperch`).
- **Architecture:** EfficientNet-B0 backbone domain-adapted for reef soundscapes (Google Research).
- **Input Specifications:**
  - **Sample Rate:** **32,000 Hz** (requires high-quality linear/polyphase resampling from 16 kHz to 32 kHz).
  - **Window Length:** **5.0 seconds** (160,000 samples per window).
- **Windowing & Pooling:**
  - A 60-second audio clip yields 12 non-overlapping 5-second windows.
  - Each window produces an embedding vector of dimension $1,280$.
  - **Pooling:** Global temporal average pooling:
    $$x_{\text{surfperch}} = \frac{1}{W} \sum_{w=1}^W e_w \in \mathbb{R}^{1280}$$

---

## 6. Model Verification Gates (Prerequisites Before Full Extraction)

Before executing batch feature extraction in Phase 3.2, each model component must satisfy a strict verification gate. Licenses, exact dependencies, authentication, and index parameters are treated as **pending verification** until confirmed in the Colab environment:

> [!IMPORTANT]
> **Credential Security Mandate:**
> Kaggle API tokens (`kaggle.json`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) or any authentication secrets must **NEVER** be placed in the git repository or workspace source files. They must be supplied strictly via Colab Secrets or ephemeral runtime environment variables.

| Verification Gate | Requirement | Check Method | Status | Exact Evidence Needed to Pass |
|---|---|---|:---:|---|
| **Gate 1: Source & Official License** | Confirm official repository and permitted use license | Inspect TF Hub model card, Kaggle model page, PyPI metadata | **Pending Verification** | Official license text / classifier confirmed (Expected: YAMNet: Apache 2.0; SurfPerch: Apache 2.0; Maad: BSD-3) |
| **Gate 2: Environment Compatibility** | Verify library dependency resolution in Colab runtime | Run test install of `tensorflow`, `tensorflow_hub`, `scikit-maad`, `librosa`, `soundfile` in Colab | **Pending Verification** | Clean `pip check` and import with zero version conflict warnings |
| **Gate 3: Auth & Checkpoint Download** | Verify model checkpoint weights pull successfully | Test TF Hub URL fetch and Kaggle API token authentication via runtime secrets | **Pending Verification** | YAMNet model graph and SurfPerch checkpoint load into Colab memory without HTTP 401/403/404 |
| **Gate 4: Preprocessing & Resampling** | Verify input pipeline preprocessing | Test waveform normalization in $[-1.0, +1.0]$ and polyphase resampling ($16\text{ kHz} \rightarrow 32\text{ kHz}$) | **Pending Verification** | No clipping, correct temporal duration, exactly $160,000$ float32 samples per 5-second window for SurfPerch |
| **Gate 5: Single-File Smoke Test & NDSI Check** | Run full forward pass on 1 pilot WAV file | Extract features through Maad (testing NDSI parameter syntax and band boundaries), YAMNet, and SurfPerch | **Pending Verification** | Exact output shapes verified: Maad $(5,)$ with valid NDSI $\in [-1.0, +1.0]$, YAMNet $(1024,)$, SurfPerch $(1280,)$ with zero NaNs/Infs |

---

## 7. Deterministic Handling of Audio Anomalies & Common Eligible Manifest

To prevent data alignment mismatches or silent sample drops that would invalidate comparative benchmarking:

### 7.1 Shared Common Eligible Manifest
- **Strict Rule:** All three feature pipelines (Maad, YAMNet, SurfPerch) must ingest and evaluate **the exact same list of eligible recordings**.
- If a recording fails quality or inference checks in *any* of the three pipelines, it must be excluded from *all three* pipelines to maintain an identical, fair comparison set.

### 7.2 Deterministic Anomaly Policies
1. **Recordings Shorter than 60 Seconds (Battery Cutoff / Deployment Termination):**
   - *Detection:* Audio duration $< 50.0\text{ seconds}$ (as identified in Phase 1.5, e.g. `ind_N1_...` with duration 25.6s).
   - *Policy:* Any file with duration $< 50.0\text{ seconds}$ is flagged as a truncated boundary file and deterministically excluded from the benchmark manifest. For files between $50.0\text{s}$ and $60.0\text{s}$, zero-pad with silence at the tail up to exactly $60.0\text{s}$ ($960,000$ samples @ 16 kHz) so window counts remain deterministic.
2. **Malformed or Unreadable WAV Files:**
   - *Detection:* RIFF header corruption, non-PCM encoding, or unreadable byte stream.
   - *Policy:* Deterministically logged in `logs/extraction_anomalies.json` with archive name, filename, and error string; recording is dropped from the common manifest across all three pipelines.
3. **Missing or Inaccessible Remote Archives:**
   - *Detection:* HTTP Range request returns 404, 500, or timeout after 3 retries.
   - *Policy:* Halt pipeline and alert researcher; do not proceed with partial archives without explicit documentation.
4. **Failed Model Inference / NaN Outputs:**
   - *Detection:* Forward pass produces `NaN`, `Inf`, or mismatched shape.
   - *Policy:* Logged immediately; file is dropped from the common eligible manifest across all representations.

---

## 8. Common Evaluation Protocol & Fair Comparison

To ensure scientific rigor, all three feature sets are evaluated under identical classifiers, hyperparameter tuning protocols, and statistical tests.

### 8.1 Baselines and Probes
1. **Majority-Class Baseline (Dummy Classifier):** Always predicts the most frequent training class. Establishes the performance floor under class imbalance.
2. **Primary Probe (Linear Classifier):** Logistic Regression with $L_2$ penalty (or Linear SVM).
   - Regularization parameter $C \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$ tuned strictly using inner 3-fold site-grouped cross-validation on training sites (or on the 8 held-out validation sites).
   - Every recording from a site remains grouped together in the same fold during tuning; test sites are strictly excluded from all model selection and tuning.
3. **Secondary Probe (Non-Linear Classifier):** Random Forest (100 trees, max depth 10) to assess non-linear feature interactions.
4. **Sanity Check (Label Permutation Test):** Train probe on randomly permuted training labels to verify the model cannot learn spurious patterns from noise (accuracy should collapse to chance).

### 8.2 Site-Level Diagnostic Aggregation & Tie-Breaking
In ecological monitoring, the operational decision is made at the **site level** rather than the individual recording level.
- **Recording-Level Prediction:** Each 1-minute recording receives predicted class probabilities $\hat{p}_i = [p_{i,D}, p_{i,H}, p_{i,N}, p_{i,R}]$.
- **Site-Level Aggregation:** Average predicted class probabilities across all sampled recordings from that site:
  $$\bar{p}_{\text{site}} = \frac{1}{N_{\text{site}}} \sum_{i=1}^{N_{\text{site}}} \hat{p}_i$$
  $$\hat{y}_{\text{site}} = \arg\max_{c \in \{D, H, N, R\}} \bar{p}_{\text{site}, c}$$
- **Tie-Breaking Rule:** In the event of an exact numerical tie between classes, the tie is broken in favor of the class with the higher prior frequency in the training set.

### 8.3 Primary & Secondary Evaluation Metrics & Uncertainty Bounds
- **Primary Metric:** **Macro-averaged F1-Score ($\text{Macro-F1}$)**:
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F1_k$$
  Treats all habitat stages equally, preventing classifiers from gaming metrics by ignoring underrepresented restored classes (N and R).
- **Secondary Metrics:**
  - **Balanced Accuracy:** Mean of recall across classes.
  - **Per-Class Metrics:** Precision, Recall, F1 for each class (D, H, N, R).
  - **Site-Level Accuracy:** Percentage of test sites correctly categorized by aggregated vote.
  - **Confusion Matrices:** Normalized by true class support.
- **Statistical Uncertainty:** 95% Confidence Intervals computed via **site-level cluster bootstrap** resampling (1,000 iterations).
  - *Small-Sample Instability Warning:* Because the test set contains only 1 N site (`mexico_N1`) and 1 R site (`mexico_R1`), bootstrap confidence intervals for N and R will have very high variance (intervals may span $[0.0, 1.0]$ or be undefined in draws omitting the single site), precluding strong four-class generalization claims.

### 8.4 Generalization Experiments
1. **Experiment 1: In-Domain Site-Grouped Representation Capacity (RQ-1):**
   - **Enforce Site-Grouped Cross-Validation:** Every recording from a site must remain in the same fold. Ordinary recording-level stratified CV is strictly prohibited because it leaks site-specific environmental signatures between train and validation folds.
   - **3-Fold Site-Grouped CV (`StratifiedGroupKFold(n_splits=3)` grouped by `site_id` across the 25 training sites):**
     - *Why 3 Folds instead of 5 Folds:* The 25 training sites contain: D (8), H (8), N (3), R (6). Because class N contains only 3 training sites, an ordinary 5-fold site-grouped split is mathematically infeasible for four-class evaluation, as two folds would contain zero N test sites.
     - *Partition:* A 3-fold site-grouped partition ensures that every single fold receives exactly 1 N site, 2 R sites, 2–3 D sites, and 2–3 H sites (~8–9 sites per fold).
     - *Guarantees:* Guarantees complete site isolation (zero site leakage) while ensuring all four classes are present in every validation fold.
2. **Experiment 2: Site-Held-Out Benchmark (RQ-2, 12 Test Sites):**
   - Train on 25 sites, tune on 8 val sites, evaluate on 12 test sites (using `site_metadata.csv`).
   - *Severe Limitation Notice:* With only one N site and one R site in the test set, results for restored classes are preliminary indicators of acoustic separability, not definitive proof of four-class generalization.
3. **Experiment 3: Leave-One-Country-Out (RQ-3 / LOCO Cross-Validation, 5 Folds):**
   - For each country, train on the other 4 countries, test on the held-out country.
   - **LOCO Support Reporting:** For each fold, explicitly report the classes present in the test country (e.g. Australia test evaluates D, H, R; Kenya test evaluates D, H, N). Fold metrics are computed strictly over evaluable classes and reported country-by-country.
4. **Experiment 4: Within-Region Site Generalization (Indonesia LOSO, 21 Sites):**
   - Evaluates site invariance within a single biogeographic regime and uniform duty cycle.

---

## 9. Sample Size Trade-Offs, Runtime Estimates & Independence Structure

### 9.1 Clustered Independence Structure & Small-Sample Instability
> [!IMPORTANT]
> **Recordings from the Same Site are Not Independent Ecological Replicates:**
> - Multiple recordings sampled from a single site share that site's specific physical location, water depth, seabed geomorphology, local tidal currents, recorder placement orientation, and hydrophone casing self-noise.
> - Consequently, **individual recordings from the same site are statistically clustered pseudoreplicates**, not independent biological samples of that habitat class.
> - All statistical evaluations, cross-validation folds, and bootstrap confidence intervals must operate strictly at the **site cluster level**. Resampling individual recordings independently would severely artificially deflate standard errors and produce false statistical confidence.
> - **Small-Sample Caveat on Test Split:** Because the held-out test split contains only one N site (`mexico_N1`) and one R site (`mexico_R1`), site-level cluster bootstrap resampling will be highly unstable for these classes. Drawing bootstrap iterations that omit either site leaves that class completely absent. Therefore, per-class estimates and site-level confidence intervals for N and R will have very wide uncertainty bounds, precluding strong four-class generalization claims from this split alone.

### 9.2 Benchmark Sample Size Analysis & Preliminary Runtime Estimates
*Note:* All runtime estimates in the table below are **preliminary theoretical projections** based on serial I/O and GPU batch inference. True end-to-end extraction speeds depend on Figshare CDN bandwidth and Colab GPU availability, and must be empirically calibrated during the Phase 3.2 smoke test.

| Option | Target Sample Count ($N_{\text{target}}$) | Total Recordings | Audio Volume Processed (Streamed) | Total Extracted Feature Matrix Size | Preliminary Colab GPU Runtime Estimate | Feasibility & Recommendation |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Option A (Full Scale)** | **100 / site** | **4,500** | ~8.2 GB | ~45 MB (Maad: 90KB, YAMNet: 18MB, SurfPerch: 23MB) | *Preliminary:* ~1.5 - 2.5 hours (to be measured in smoke test) | **Recommended for Final Benchmark:** High temporal coverage across site deployment window; minimal feature storage footprint ($< 50\text{ MB}$) |
| **Option B (Moderate Scale)** | **50 / site** | **2,250** | ~4.1 GB | ~23 MB | *Preliminary:* ~45 - 75 minutes | Strong fallback if Colab session timeout or bandwidth throttles |
| **Option C (Pilot Scale)** | **10 / site (12 sites)** | **120** | ~220 MB | ~1.2 MB | *Preliminary:* ~3 minutes | **Current Pilot:** Pipeline, preprocessing & storage verification ONLY (zero model training or evaluation) |

### 9.3 Strict Storage Budget (MacBook Air M2, 8 GB RAM, 256 GB SSD)
- **Local Machine Constraints:** Zero bulk WAV files downloaded. Feature matrices (`.npz` or `.parquet`) occupy **$< 50\text{ MB}$**, fully safe for local storage and GitHub.
- **Cloud Execution:** Audio extraction runs on Google Colab, streaming byte ranges from Figshare or downloading one site ZIP ephemerally, extracting features, and immediately deleting the archive.

---

## 10. Work Breakdown Structure for Student Research Team

| Work Package | Focus Area | Responsible Role | Key Deliverables |
|---|---|---|---|
| **WP 1** | Model Verification & Ingestion | Deep Learning Lead | Verify TF Hub & Kaggle models; implement smoke test on single audio file |
| **WP 2** | Feature Extraction Pipeline | Acoustic / ML Engineer | Colab extraction notebook for Maad, YAMNet, SurfPerch; export `.parquet` / `.npz` |
| **WP 3** | Benchmark & Cross-Validation | Evaluation Lead | Implement common evaluation protocol, Linear/RF probes, LOCO folds, site-level voting |
| **WP 4** | Statistical Reporting & Synthesis | Research Analyst | Cluster bootstrap CIs, confusion matrices, UMAP projections, scientific writeup |

---

## 11. Summary of Key Decisions to Resolve Before Phase 3.2

Before writing extraction code in Phase 3.2, the following methodological decisions are confirmed and documented:

1. **Benchmark Scale & Stage Separation:** Option A ($N_{\text{target}} = 100$ recordings/site = 4,500 recordings total across all 45 sites) is selected for the full benchmark; Stage 1 pilot (120 recordings) is restricted strictly to technical pipeline, preprocessing, output-shape, and storage verification with **zero supervised model training or evaluation**.
2. **RQ-1 Site-Grouped Cross-Validation:** 3-fold Site-Grouped CV (`StratifiedGroupKFold` grouped by `site_id`) across the 25 training sites is enforced to eliminate site leakage while ensuring all 4 classes are present in every validation fold (5-fold is infeasible because class N has only 3 training sites).
3. **NDSI Parameter Status:** Proposed frequency bands (1–2 kHz anthrophony, 2–8 kHz biophony under 8 kHz Nyquist limit) and `maad.features.soundscape_index` parameters remain **pending verification** until validated in the Phase 3.2 single-file smoke test.
4. **Model Verification Gates & Credential Security:** All model licenses, library compatibility, and checkpoint downloads remain **pending verification**. Kaggle API credentials must never be committed to Git and will be provided via Colab Secrets / environment variables at runtime only.
5. **Common Eligible Manifest & Deterministic Anomalies:** All three pipelines must evaluate the exact same eligible recordings. Files $<50.0\text{s}$ are dropped and logged; files between $50.0\text{s}$ and $60.0\text{s}$ are zero-padded to exactly $60.0\text{s}$.
6. **Small-Sample Uncertainty Caveat:** Test set has only one N site (`mexico_N1`) and one R site (`mexico_R1`). Multiple recordings from one site are clustered pseudoreplicates. We will not claim strong four-class generalization evidence from this split alone.
7. **Scientific Integrity:** Categorical classification of audio into D/H/N/R demonstrates acoustic habitat discrimination, not proof of ecological restoration success.

---

*End of Revised Phase 3.1 Planning Document.*

---

## 12. Phase 3.3 Gate Decision — Benchmark Scale (logged 2026-10-31)

| Parameter | Value |
|---|---|
| **Option selected** | **Option A — Full Scale** |
| `TARGET_PER_SITE` | **100** recordings per site |
| Total target recordings | 4,500 (45 sites × 100) |
| Random seed | 42 |
| Minimum duration filter | 50 s |
| Segments per recording | 12 (non-overlapping 5-second windows) |
| Indonesia duty-cycle handling | `hourly_binning` |
| Stratification | 2-D: deployment date × hour-of-day |
| Full config | `config/sampling_config_full.yaml` |

**Rationale for Option A:** Provides maximal temporal coverage per site (vs. Option B's 50/site) while keeping the entire feature matrix well under 50 MB (≈ 45 MB for all three pipelines), which is safe for local MacBook Air M2 storage. Option B remains the documented fallback in case of Colab session timeout or CDN bandwidth throttling.

> Phase 3.3 decision also recorded in `PHASES.md`.
