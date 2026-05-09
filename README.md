# Sinusitis Voice Analysis — wav2vec 2.0 Deep Learning Pipeline

> MEng Research Project · Clinical Speech Biomarker Detection

## Overview

This project implements a full end-to-end deep learning pipeline to detect Chronic Rhinosinusitis (CRS) from voice recordings using acoustic biomarkers. It forms part of a longitudinal clinical intervention study involving Functional Endoscopic Sinus Surgery (FESS) patients recorded across multiple sessions pre- and post-surgery, alongside matched control and comparison surgical groups.

The system leverages **wav2vec 2.0** (`facebook/wav2vec2-base-960h`) — a self-supervised transformer pretrained on 960 hours of English speech — to extract high-level acoustic representations from raw waveforms, and evaluates two learning paradigms:

- **Scratch mode** — randomly initialised backbone, fully trained end-to-end
- **Finetune mode** — pretrained backbone with layer-wise LR decay and two-phase head warmup; compared against an SVM on frozen backbone embeddings to validate the transfer learning hypothesis

---

## Clinical Context

### Study design

| Property | Detail |
|---|---|
| Target condition | Chronic Rhinosinusitis (CRS) |
| Intervention | Functional Endoscopic Sinus Surgery (FESS) |
| Comparison groups | Control, Septoplasty, Tonsillitis |
| Longitudinal sessions | Session 1 (pre-op), Session 2 (early post-op), Session 3 (late recovery) |
| Total speakers | 107 |
| Sample rate | 16 kHz |

### Audio task channels (13 types)

| Group | Columns | Clinical purpose |
|---|---|---|
| Vowels | a, e, i, o, u | Steady-state phonation quality |
| Sustained | a1, a2, a3 | Repeated phonation stability |
| Speech | speech | Connected discourse, prosodic patterns |
| TDU | agua, brasero, dia, mesa | Word-level articulation (Spanish) |

### Research hypothesis

CRS alters the vocal tract through mucosal inflammation and blocked nasal passages, producing measurable acoustic signatures in vowel quality, resonance, and airflow. If FESS resolves these changes, a model trained to detect CRS should also track post-surgical acoustic recovery — providing a non-invasive, objective complement to subjective symptom scores such as SNOT-22.

---

## Repository Structure

```
MEng-Research-Project/
├── scripts/
│   ├── run_experiment.py           # Main entry point — all experiments
│   ├── run_preprocessing.py        # Segment extraction from raw WAV files
│   ├── run_diagnostic_pipeline.py
│   ├── run_exploratory_DA.py
│   └── run_visualisations.py
│
├── src/
│   ├── audio/
│   │   ├── io.py                   # WAV loading, mono conversion
│   │   ├── cleaning.py             # Silence trim, high-pass filter, VAD
│   │   ├── segmentation.py         # Sliding window, instance normalisation
│   │   └── augmentation.py         # Noise, pitch shift, time stretch
│   │
│   ├── pipeline/
│   │   ├── dataset.py              # SinusitisDataset, PairedDataset
│   │   ├── dataloader.py           # Collate, per-sample normalisation, samplers
│   │   └── splits.py               # Patient-level / paired / generalisation splits
│   │
│   ├── experiments/
│   │   ├── base.py                 # ExperimentConfig, base experiment class
│   │   ├── all_experiments.py      # Exp1-Exp5 class definitions
│   │   └── exp4_paired_change.py   # Paired within-patient experiment
│   │
│   ├── training/
│   │   ├── trainer.py              # Training loop, Focal Loss, head warmup
│   │   ├── metrics.py              # Accuracy, F1, AUC, per-audio-type
│   │   ├── reporter.py             # Academic PDF report generator
│   │   ├── eval_utils.py           # Threshold calibration, patient-level voting
│   │   ├── svm_classifier.py       # SVM on frozen backbone embeddings
│   │   ├── imbalance.py            # Oversampling, weighted sampling
│   │   └── checkpoint.py           # Checkpoint inspection utilities
│   │
│   ├── utils/paths.py              # CSV path resolution to audio files
│   └── config.py                   # Global constants
```

---

## Preprocessing Pipeline

Executed via `scripts/run_preprocessing.py`. Produces `.pt` tensor files consumed directly by the training pipeline.

### Pipeline stages

1. Load WAV → mono float32 tensor
2. Trim 50ms leading silence (reduced from 150ms to protect short vowel recordings)
3. Resample to 16 kHz
4. High-pass filter at 80 Hz (removes low-frequency handling noise)
5. Adaptive VAD (threshold = 5% of file peak RMS, floored at 0.01)
6. Peak amplitude normalisation → range [-1, +1]
7. Optional augmentation: Gaussian noise (SNR 10-30 dB), pitch shift (±2 semitones)
8. Sliding window segmentation — **49,152 samples = 3.072s** at 50% overlap
9. Instance normalisation (zero-mean, unit-variance per segment)
10. Save as `.pt` tensors

### Segment naming convention

```
ID{subject_id}_ses{session}_{audio_col}_seg{index:04d}.pt
```

### Why 3.072 seconds

wav2vec 2.0's CNN encoder downsamples by factor 320 (one latent frame per 20ms). The positional convolution kernel is 128 frames wide.

| Window | Duration | Frames | Valid |
|---|---|---|---|
| 8,192 samples | 0.512s | 25.6 | No — kernel (128) larger than sequence |
| **49,152 samples** | **3.072s** | **153.6** | **Yes — comfortably within range** |

Segments shorter than 128 frames produce invalid positional encodings and place the model outside its pretraining distribution. The 0.512s window used in earlier versions was a primary cause of degraded performance.

### Running preprocessing in Colab

```python
from src.pipeline.preprocess import process_from_csv

process_from_csv(
    csv_path     = "/content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv",
    project_root = "/content/drive/MyDrive",   # resolve_path appends Data/... from here
    output_dir   = "/content/clean_audio_3s",
    mode         = "scratch",
    augment      = False,
)
```

On subsequent Colab sessions, restore from the saved zip on Drive instead of re-processing.

---

## Five Experiments

All experiments enforce patient-level splitting with a hard assertion — no patient appears in more than one split.

| Exp | Clinical question | Task | Groups |
|---|---|---|---|
| **Exp 1** | Can we distinguish CRS from healthy controls acoustically? | Binary | FESS vs Control |
| **Exp 2** | Does the voice change measurably after sinus surgery? | Binary | FESS pre-op vs post-op |
| **Exp 3** | Can we track three stages of recovery from voice alone? | 3-class | Session 1 / 2 / 3 |
| **Exp 4** | Can we detect within-patient change without speaker identity? | Paired binary | Pre/post segment pairs, same patient |
| **Exp 5** | Do CRS-trained features generalise across surgical groups? | Binary, cross-domain | Train: FESS — Test: Septoplasty + Tonsillitis |

### Exp 4 — Paired change detection

Eliminates the speaker identity confound by training on pre/post pairs from the same patient. The model must detect acoustic change without using voice identity as a shortcut.

**Positive pairs (label 1):** pre-op segment + post-op segment, same patient  
**Negative pairs (label 0):** two segments within the same recording session

**Known limitation:** Pre/post pairs also differ in recording session conditions. A matched non-surgical control group recorded at equivalent time intervals would be needed to isolate the surgery-specific acoustic effect from ambient confounds.

### Exp 5 — Generalisation

The most scientifically honest experiment. No patient overlap between train and test by design. Results from Exp 5 provide the most reliable indication of whether the model has learned genuine CRS-related acoustic features rather than speaker identity or recording artefacts.

---

## Model Architecture

### Wav2Vec2Classifier

```
Input waveform [B, 49152]       (3.072s at 16kHz)
        |
        v
wav2vec 2.0 backbone
  CNN feature extractor (7 layers, total stride = 320)
        |   [B, 153, 512]
        v
  Transformer encoder (12 layers, hidden = 768)
        |   [B, 153, 768]
        v
  Mean pooling over time
        |   [B, 768]
        v
  MLP Classification Head
    Linear(768 -> 256)
    LayerNorm(256)
    ReLU
    Dropout(0.3)
    Linear(256 -> num_classes)
        |   [B, num_classes]
```

The MLP head replaces the earlier single `Linear(768 -> num_classes)` layer. The bottleneck projection forces compression rather than memorisation, reducing scratch overfitting. LayerNorm stabilises finetune training during the head warmup phase.

### SVM baseline (finetune mode, optional)

After neural training, the frozen backbone produces mean-pooled embeddings `[N, 768]` fed to an RBF-SVM with `class_weight='balanced'`. This directly tests the transfer learning hypothesis: if frozen pretrained features + SVM outperform end-to-end scratch training, the pretrained representations carry clinically relevant structure that full fine-tuning alone fails to exploit.

Activate with `--use_svm`.

---

## Training Configuration

### Key ExperimentConfig fields

| Field | Default | Description |
|---|---|---|
| `mode` | `finetune` | `scratch` or `finetune` |
| `learning_rate` | `1e-4` | Head LR; backbone gets lower LR via layerwise decay |
| `layerwise_lr_decay` | `0.8` | Each lower transformer layer gets LR x 0.8^n |
| `head_warmup_epochs` | `3` | Epochs of backbone-frozen head warmup (finetune only) |
| `early_stop_metric` | `val_f1` | Stop on val F1 macro, not val loss |
| `early_stop_patience` | `5` | Epochs without improvement before stopping |
| `use_focal_loss` | `True` | Focal Loss (gamma=2) instead of CrossEntropy |
| `focal_gamma` | `2.0` | Focal down-weighting strength |
| `label_smoothing` | `0.1` | Applied inside Focal Loss |
| `imbalance_strategy` | `oversample` | `oversample`, `weighted`, or `none` |
| `save_every` | `5` | Checkpoint every N epochs |
| `keep_last_n` | `2` | Max epoch checkpoints on disk (Drive space management) |

### Focal Loss

The dataset has a ~3:1 class imbalance. Focal Loss down-weights gradient contributions from easy majority-class examples:

```
FL(p_t) = -(1 - p_t)^gamma * log(p_t)
```

At gamma=2, a sample predicted with 90% confidence contributes only 1% of its standard CrossEntropy gradient, forcing the model to focus on hard misclassified samples.

### Head warmup (finetune only)

Without warmup, a randomly-initialised head produces large biased gradients from step 0 that corrupt pretrained backbone representations before any stable features form. Empirically: finetune collapsed to predicting only one class from epoch 1 without this mechanism.

Phase 1 (epochs 1-N): only the MLP head trains; backbone fully frozen.  
Phase 2 (epoch N+1 onwards): backbone unfrozen with layerwise LR decay; optimizer and scheduler rebuilt.

### Early stopping on val F1

Val loss and val F1 are not reliably correlated in this dataset. The model that minimises loss is often a collapsed model (predicting one class confidently gives low entropy). Stopping on F1 macro ensures the saved checkpoint actually classifies both classes.

### Layer-wise LR decay (finetune)

```
Classifier head       lr = 1e-5      (full)
Transformer layer 11  lr = 8.0e-6   (x 0.8^1)
Transformer layer 10  lr = 6.4e-6   (x 0.8^2)
        ...
Transformer layer 0   lr = 1.1e-6   (x 0.8^12)
Feature projection    lr = 8.6e-7   (x 0.8^13)
```

Lower layers encode general speech phonetics from LibriSpeech pretraining. They should adapt slowly on a 40-patient clinical dataset.

---

## Evaluation Framework

### Metrics per split

- Accuracy, F1-macro, F1 per class, ROC-AUC, confusion matrix

### Per-audio-type breakdown (test set)

Predictions are grouped by audio column to report separate metrics for vowels, sustained, speech, and TDU types. Identifies which acoustic task carries the most diagnostic signal.

### Threshold calibration (binary experiments)

After training, the optimal decision threshold is found on the val set by sweeping 0.1-0.9, selecting t* that maximises val F1. Applied to test predictions. The default 0.5 threshold is sub-optimal when oversampled (50/50) training data is evaluated against the true (75/25) test distribution.

### Patient-level voting

Segment-level predictions are aggregated per patient by majority vote and mean probability. The patient-level metric is the clinically meaningful unit — the question is "does this patient have CRS", not "does this 3-second segment have CRS".

---

## Google Colab Workflow

**Cell 1 — GPU and Drive**
```python
import torch
from google.colab import drive
print(torch.cuda.get_device_name(0))
drive.mount('/content/drive')
```

**Cell 2 — GitHub credentials**
```python
with open("/content/drive/MyDrive/secrets/github_token.txt") as f:
    TOKEN = f.read().strip()
import subprocess
subprocess.run(["git", "config", "--global", "credential.helper", "store"])
with open("/root/.git-credentials", "w") as f:
    f.write(f"https://{TOKEN}:x-oauth-basic@github.com\n")
```

**Cell 3 — Clone or sync repo**
```python
import os
PROJECT_ROOT = "/content/project"
if not os.path.exists(PROJECT_ROOT):
    os.system(f"git clone https://github.com/T-Machethe/MEng-Research-Project.git {PROJECT_ROOT}")
else:
    os.chdir(PROJECT_ROOT)
    os.system("git fetch --all && git reset --hard origin/main && git clean -fd")
os.chdir(PROJECT_ROOT)
```

**Cell 4 — Segment extraction or zip restore**

First run: extracts 3.072s segments from original WAV files, saves zip to Drive.  
All subsequent sessions: restores from zip (~2 min).

**Cell 5 — Training**
```python
cmd = [
    sys.executable, "scripts/run_experiment.py",
    "--exp",                "all",
    "--compare_modes",
    "--num_epochs",         "30",
    "--batch_size",         "8",         # 3s segments — reduce to 4 if OOM
    "--imbalance",          "oversample",
    "--warmup_steps",       "200",
    "--learning_rate",      "1e-5",
    "--layerwise_lr_decay", "0.8",
    "--label_smoothing",    "0.1",
    "--head_warmup_epochs", "3",
    "--early_stop_metric",  "val_f1",
    "--focal_gamma",        "2.0",
    "--save_every",         "5",
    "--keep_last_n",        "2",
    "--use_svm",
    "--segment_dir",        "/content/clean_audio_3s",
    "--csv_path",           CSV_PATH,
    "--output_dir",         OUTPUT_DIR,
]
```

---

## CLI Reference

```bash
# All experiments, scratch vs finetune, with SVM
python scripts/run_experiment.py \
    --exp all --compare_modes \
    --num_epochs 30 --batch_size 8 \
    --learning_rate 1e-5 \
    --head_warmup_epochs 3 \
    --early_stop_metric val_f1 \
    --focal_gamma 2.0 \
    --use_svm \
    --segment_dir /path/to/segments \
    --csv_path /path/to/clinical_all_sessions.csv \
    --output_dir /path/to/results

# Single experiment, finetune only
python scripts/run_experiment.py --exp 1 --mode finetune --num_epochs 30

# Re-run preprocessing with corrected window size
python scripts/run_preprocessing.py
```

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--exp` | required | `1`-`5` or `all` |
| `--mode` | `finetune` | `scratch`, `finetune`, or `both` |
| `--compare_modes` | False | Run scratch then finetune, generate comparison report |
| `--num_epochs` | 30 | Max training epochs per mode |
| `--batch_size` | 16 | Use 8 for 3s segments on T4 |
| `--learning_rate` | 1e-4 | Head LR; use 1e-5 for finetune |
| `--layerwise_lr_decay` | 0.8 | Finetune backbone LR decay per layer |
| `--head_warmup_epochs` | 3 | Backbone-frozen warmup epochs |
| `--early_stop_metric` | `val_f1` | `val_f1` or `val_loss` |
| `--early_stop_patience` | 5 | Epochs without improvement |
| `--focal_gamma` | 2.0 | Focal Loss focusing parameter |
| `--label_smoothing` | 0.1 | Label smoothing coefficient |
| `--imbalance` | `oversample` | `oversample`, `weighted`, or `none` |
| `--warmup_steps` | 200 | LR warmup steps |
| `--save_every` | 5 | Epoch checkpoint frequency |
| `--keep_last_n` | 2 | Max epoch checkpoints retained |
| `--use_svm` | False | Run SVM on frozen finetune embeddings |
| `--svm_C` | 1.0 | SVM regularisation |
| `--svm_kernel` | `rbf` | `rbf` or `linear` |
| `--verbose` | False | Full INFO log to console |
| `--seed` | 42 | Random seed |

---

## Output Structure

```
results/
└── exp1_crs_vs_control_finetune/
    ├── best_model.pt               # Best checkpoint by val F1
    ├── latest_checkpoint.pt        # Most recent epoch (for resume)
    ├── epoch_005.pt                # Periodic checkpoint (pruned automatically)
    ├── results_summary.json        # All metrics, training history, SVM results
    ├── training.log                # Full DEBUG log
    ├── report.pdf                  # Academic PDF (cover + all plots)
    ├── tables/
    │   ├── summary_table.csv       # Val/test metrics
    │   └── per_audio_type.csv      # Per-audio-type breakdown
    ├── plots/
    │   ├── 01_loss_curves.png
    │   ├── 02_confusion_matrix.png
    │   ├── 03_roc_curve.png
    │   └── 04_f1_bars.png
    ├── svm/
    │   └── svm_results.csv         # SVM metrics across splits
    └── tensorboard/                # TensorBoard event files
```

Epoch checkpoints are pruned to `keep_last_n` most recent. `best_model.pt` and `latest_checkpoint.pt` are never deleted.

---

## Logging

| Stream | Level | Content |
|---|---|---|
| Console | WARNING + epoch_summary | One line per epoch: loss, acc, F1, AUC, time |
| `training.log` on Drive | DEBUG | Per-step loss, classification reports, confusion matrices, all events |

Pass `--verbose` to promote console to full INFO level for debugging.

---

## Known Limitations

1. **Small patient cohort** — ~107 speakers, ~40 in training after splits. Segment-level count (8k-50k) inflates apparent data size but all segments from the same patient stay on the same side of every split boundary.

2. **Domain mismatch** — wav2vec 2.0 was pretrained on English LibriSpeech. Clinical audio is Spanish, hospital-recorded. Pretrained phonetic representations do not directly encode Spanish phonology.

3. **Session confound (Exp 2, Exp 4)** — pre-op and post-op recordings may differ in room acoustics, microphone position, and patient state independently of the surgical effect. Matched non-surgical controls recorded at equivalent intervals would be needed to isolate surgery-specific acoustic change.

4. **Segment independence assumption** — overlapping sliding window segments share acoustic context but are treated as independent training examples.

---

## Research Contribution

This system provides structured empirical evidence on three questions:

1. **Are CRS-related vocal signatures learnable from raw speech?** Exp 1 evaluates above-chance classification of CRS vs healthy controls using only voice recordings and no hand-crafted features.

2. **Does transfer learning from speech SSL models benefit clinical voice analysis?** The three-way comparison (scratch neural / finetune neural / SVM on frozen features) provides a controlled test. SVM on pretrained features outperforming scratch training indicates the pretrained representations carry clinically relevant structure.

3. **Do learned features generalise across surgical groups?** Exp 5 tests this directly with no patient overlap between train and test, providing the most reliable estimate of true acoustic signal.

---

## Dependencies

```
Python         >= 3.10
PyTorch        >= 2.6
torchaudio     >= 2.0
transformers   >= 4.35
scikit-learn   >= 1.3
pandas         >= 2.0
matplotlib     >= 3.7
scipy          >= 1.11
```

Tested on Google Colab T4 GPU (16GB VRAM).  
Use `--batch_size 8` for 3.072s segments. Reduce to 4 if CUDA OOM.

---

## Reproducibility

All experiments are deterministic under a fixed seed. Patient-level split assignment is fully deterministic given the same CSV and seed — re-running produces identical train/val/test patient assignments.

To resume a crashed run:

```bash
python scripts/run_experiment.py --exp 1 --mode finetune \
    --output_dir /path/to/existing/results
# Automatically resumes from latest_checkpoint.pt
```

---

*MEng Research Project — Biomedical Signal Processing and Machine Learning*  
*Clinical Speech Analysis for Upper Airway Disease Detection and Post-Surgical Outcome Tracking*