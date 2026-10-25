# Reef Bioacoustics Pipeline: Ecoacoustic Indices vs YAMNet vs SurfPerch

Evaluation pipeline comparing three audio representations for classifying reef health from passive acoustic recordings:
1. **Traditional Ecoacoustic Indices:** $\mathbb{R}^5$ (ACI, NDSI, ADI, BI, AEI) via `scikit-maad` with marine-adapted frequency bands.
2. **YAMNet Embeddings:** $\mathbb{R}^{1024}$ general-purpose bioacoustic foundation model via TensorFlow Hub.
3. **SurfPerch Embeddings:** $\mathbb{R}^{1280}$ domain-adapted marine bioacoustic model via Kaggle Models.

---

## 1. Project Architecture

```
coral-reef/
├── config.yaml                     # Unified pipeline configuration
├── main.py                         # Unified entry point script
├── tests/
│   └── test_pipeline_unit.py       # Unit tests for all modules
├── EcoSound-Audit/
│   ├── config.yaml
│   ├── DATA_AUDIT.md               # Deliverable 1: Dataset audit
│   ├── PHASE3_PLAN.md              # Research & experimental design
│   ├── data/
│   │   ├── manifests/manifest.csv  # 45-site inventory
│   │   └── sampling/
│   │       ├── site_metadata.csv   # 45 sites with Train/Val/Test & LOCO splits
│   │       ├── sampled_recordings.csv # 120 pilot sampled recordings
│   │       └── segment_manifest.csv   # 1,440 5-second segments
│   ├── notebooks/
│   │   ├── EcoSound_Phase3_Colab_SmokeTest.ipynb
│   │   └── EcoSound_Phase3_EndToEnd_Pipeline.ipynb
│   └── src/ecosound/
│       ├── __init__.py
│       ├── models.py               # Metadata structures & filename regex
│       ├── remote_reader.py        # HTTP Range ZIP Central Directory reader
│       ├── sampler.py              # 2D (date, hour) temporal sampling & LOCO splits
│       ├── preprocessing.py        # 5-second slicing & polyphase resampling (32 kHz)
│       ├── features.py             # Maad, YAMNet, SurfPerch extractors & caching
│       ├── training.py             # StandardScaler on train only & L2 Logistic Regression
│       ├── evaluation.py           # Site-grouped CV (3-fold), Macro-F1, AUROC, voting
│       ├── generalization.py       # Leave-One-Country-Out (LOCO) evaluation
│       ├── robustness.py           # Hardware degradations: 8 kHz, 8-bit, SNR noise
│       └── rti.py                  # Restoration Trajectory Index (D-H centroid axis)
└── results/                        # Output CSVs, confusion matrices, and figures
```

---

## 2. Key Methodological Standards & Hard Rules

- **Zero Data Leakage:** All splits and cross-validation folds group strictly by `site_id` (`StratifiedGroupKFold`). Segments from the same site never appear in both train and validation/test.
- **Fair Model Comparison:** All three representations are evaluated with the identical classifier family ($L_2$-regularized Logistic Regression), the identical hyperparameter search grid ($C \in [10^{-4}, \dots, 10^2]$), and identical class weighting (`class_weight='balanced'`).
- **Train-Only Scaling:** `StandardScaler` is fit strictly on training set folds and transforms validation/test folds without data leakage.
- **Dual Label Mode:**
  - **4-Class Native:** `D` (Degraded), `H` (Healthy), `N` (Early-stage restored <3 mo), `R` (Mid-stage restored 32–53 mo).
  - **3-Class Configurable:** `Degraded`, `Healthy`, `Restoration` (merging N + R).
- **Marine Band Adaptations:** Under an 8,000 Hz Nyquist limit (16 kHz sampling rate), NDSI anthrophony proxy is set to 1,000–2,000 Hz (hydrodynamic/boat noise) and biophony proxy is set to 2,000–8,000 Hz (snapping shrimp and biological choruses).
- **RTI Ecological Grounding:** As verified in `DATA_AUDIT.md`, the Figshare dataset contains categorical site codes but no continuous benthic ground truth (coral cover %, fish biomass). Per hard rule 4.4, empirical correlation is explicitly documented as skipped rather than substituting synthetic proxies.

---

## 3. Quickstart & Reproducibility Commands

### Step 1: Install Dependencies
```bash
pip install numpy scipy scikit-learn scikit-maad soundfile pyyaml pandas matplotlib pytest
```

### Step 2: Run Unit Tests
```bash
pytest tests/test_pipeline_unit.py -v
```

### Step 3: Generate 5-Second Segment Manifest
```bash
# 4-class native mode
python main.py --mode manifest --classes 4class

# 3-class mode
python main.py --mode manifest --classes 3class
```

### Step 4: Run Colab Smoke Test (Single File Verification)
Open `EcoSound-Audit/notebooks/EcoSound_Phase3_Colab_SmokeTest.ipynb` in Google Colab with GPU runtime and verify all three feature extractors:
- scikit-maad: `(5,)`
- YAMNet: `(1024,)`
- SurfPerch: `(1280,)`
