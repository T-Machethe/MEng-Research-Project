# Sinusitis Voice Analysis — wav2vec 2.0 Deep Learning Pipeline

## Overview

This project implements a full end-to-end deep learning pipeline to detect Chronic Rhinosinusitis (CRS) using speech biomarkers extracted from voice recordings. It is built as part of a longitudinal clinical intervention study involving Functional Endoscopic Sinus Surgery (FESS) patients recorded pre- and post-surgery, alongside Control, Septoplasty, and Tonsillitis cohorts.

The system leverages self-supervised speech representations from **wav2vec 2.0** to model subtle acoustic changes associated with upper airway inflammation and surgical recovery trajectories.

---

## Clinical Context

The dataset originates from a structured clinical study with longitudinal follow-up:

- **Groups**
  - FESS (Chronic Rhinosinusitis patients undergoing surgery)
  - Control
  - Septoplasty
  - Tonsillitis

- **FESS structure**
  - Session 1: Pre-surgery baseline
  - Session 2: Early post-surgery
  - Session 3: Late post-surgery recovery

- **Audio tasks (13 channels)**
  - Sustained vowels: a, e, i, o, u
  - Repeated vowels: a1, a2, a3
  - Speech tasks: speech, agua, brasero, dia, mesa

- **Scale**
  - 107 speakers
  - ~49,652 processed audio segments
  - 16 kHz sampling rate

The objective is to detect CRS-related acoustic signatures and quantify post-surgical vocal recovery patterns.

---

## System Architecture

Project Folder/
├── Data/
│ ├── Audios/ Raw WAV recordings
│ ├── Clinical/ Clinical metadata CSV
│ └── clean_audio/ Processed .pt segments (~49k)
│
├── scripts/ Entry points
│ ├── run_preprocessing.py
│ ├── run_experiment.py
│ ├── visualise.py
│ ├── eda.py
│ └── diagnose_pipeline.py
│
├── src/
│ ├── audio/ Signal processing pipeline
│ ├── pipeline/ Dataset + splitting logic
│ ├── experiments/ Experiment definitions (Exp1–Exp5)
│ ├── training/ Model + optimisation + reporting
│ └── config.py
│
├── results/ Experiment outputs (plots, CSV, reports)
├── Plots and visuals/ EDA + diagnostics
└── test_single_file.py


---

## Data Pipeline

### 1. Data Ingestion

- Clinical CSV drives file resolution via structured mapping (`COLUMN_TO_SUBFOLDER`)
- Handles inconsistent naming across groups and tasks
- Special handling for:
  - FESS “Sustained Vowels” naming mismatch
  - Multi-task directory alignment (Vowels / Speech / TDU)

---

### 2. Preprocessing Pipeline

Executed via `scripts/run_preprocessing.py`

Pipeline stages:

1. Load WAV → mono conversion  
2. Trim silence (0.05s)  
3. Resample to 16 kHz  
4. High-pass filter (80 Hz)  
5. Adaptive Voice Activity Detection (VAD threshold = 0.01)  
6. Peak normalisation  
7. Data augmentation (noise, pitch shift, time stretch)  
8. Sliding window segmentation (8192 samples, 50% overlap)  
9. Instance normalisation  
10. Save as `.pt` tensors  

Output format:ID{id}ses{session}{audio_type}_seg{index}.pt


### Key improvements

- Reduced over-trimming for short vowel signals
- Adaptive VAD prevents silence over-removal
- NaN-safe normalisation guard added
- Stable segmentation across all audio types

---

## Data Quality & Diagnostics

Dedicated diagnostic tools:

- `diagnose_pipeline.py` → retention tracking per stage
- `eda.py` → dataset statistics and imbalance analysis
- `visualise.py` → waveform transformation inspection

### Key findings

- Overall retention: **~76.3%**
- VAD performs reliably across groups
- Primary signal loss source: aggressive trimming in short vowel samples
- Dataset imbalance ~3:1 → handled via class weighting and sampling strategies

---

## Experiment Framework

All experiments inherit from a unified base class.

### Experiment definitions

| Experiment | Objective |
|------------|----------|
| Exp 1 | CRS vs Control classification |
| Exp 2 | Pre vs Post surgical change |
| Exp 3 | Recovery trajectory (3-class: ses1/2/3) |
| Exp 4 | Paired within-patient change detection |
| Exp 5 | Cross-domain generalisation (FESS → Sept/Tonsill) |

---

## Data Splitting Strategy

Strict leakage prevention is enforced.

- Patient-level splitting (hard assertion failure on leakage)
- Stratification by clinical group
- Dedicated split strategies:
  - `patient_level_split`
  - `paired_patient_split`
  - `generalisation_split`

No segment from the same patient appears across train/test boundaries.

---

## Model Architecture

### Backbone

- wav2vec 2.0 (`facebook/wav2vec2-base-960h`)

### Architecture

Input: [B, 8192]

→ wav2vec2 encoder
→ mean pooling over time
→ Dropout (0.1)
→ Linear layer (768 → classes)


### Training modes

- **Scratch mode**
  - Randomly initialised weights
  - Full end-to-end training

- **Finetune mode**
  - Pretrained backbone
  - CNN + bottom 6 transformer layers frozen

---

## Optimisation Setup

- Optimiser: AdamW
- Learning rate: 1e-4
- Beta2: 0.98
- Scheduler: Linear warmup
- Gradient clipping enabled
- Early stopping on validation loss
- Class imbalance handling:
  - Weighted loss OR oversampling

### Stability safeguards

- NaN loss detection (batch skip mechanism)
- Input sanitisation (`nan_to_num`)
- Epoch checkpointing (`latest_checkpoint.pt`)

---

## Training & Evaluation

### Metrics

- Accuracy
- F1-score (macro and weighted)
- ROC-AUC
- Confusion matrix

### Outputs

Each run generates:

- Loss curves
- Confusion matrix
- ROC curves
- F1 comparison plots
- CSV metrics export
- Full PDF report

All handled by `src/training/reporter.py`.

---

## Key Design Decisions

### 1. Patient-level split enforcement
Prevents inflated performance due to speaker leakage.

### 2. Segment-based learning
Transforms long recordings into high-sample acoustic learning units (~49k segments).

### 3. Self-supervised backbone choice
wav2vec 2.0 captures phonetic + suprasegmental features relevant to pathology.

### 4. Multi-task experimental design
Allows:
- classification (CRS detection)
- longitudinal modelling (recovery)
- domain generalisation testing

---

## CLI Usage

### Preprocessing
```bash
python scripts/run_preprocessing.py

Single experiment

python scripts/run_experiment.py --exp 1 --mode finetune --num_epochs 30

Scratch vs finetune comparison

python scripts/run_experiment.py --exp 1 --compare_modes --num_epochs 30

Audio type comparison

python scripts/run_experiment.py --exp 1 --compare_types --num_epochs 30

Run all experiments

python scripts/run_experiment.py --exp all --compare_modes --num_epochs 30

Engineering Highlights
Fully modular PyTorch architecture
Reproducible experiment framework
Leakage-proof dataset design
Production-grade training stability (NaN guards, checkpointing, resumption)
Unified reporting system (plots + CSV + PDF)
Scalable preprocessing pipeline (~50k segments)

Research Contribution

This system demonstrates that:

CRS-related vocal signatures are learnable from raw speech using self-supervised representations
Post-surgical recovery trajectories can be modelled as structured acoustic shifts
Domain generalisation remains a critical bottleneck in clinical speech AI
Reproducibility
Python 3.11
PyTorch + HuggingFace Transformers
torchaudio + librosa
Google Colab (T4 GPU tested)

All experiments are deterministic under fixed seeds and patient-level splits.

Author Context

Developed as part of an MSc-level research project in biomedical signal processing and machine learning, focused on clinical speech analysis for upper airway disease detection and post-surgical outcome tracking.
