# Sinusitis voice analysis — Transfer learning pipeline

> MEng Research assignment - Improving sinusitis diagnosis from voice recordings using transfer learning.

 
## Overview

This project implements a full end-to-end deep learning pipeline to detect Chronic Rhinosinusitis (CRS) from voice recordings using acoustic biomarkers. It forms part of a longitudinal clinical intervention study involving Functional Endoscopic Sinus Surgery (FESS) patients recorded across multiple sessions pre- and post-surgery, alongside matched control and comparison surgical groups.

The central assignment question is:

> **Can transfer learning from English self-supervised speech models improve sinusitis detection from Spanish clinical voice recordings?**

Three backbone architectures are compared across two learning paradigms (scratch and finetune) and a linear SVM probe on frozen embeddings:

| Backbone | Pretraining | Languages | Parameters |
|---|---|---|---|
| `facebook/wav2vec2-base-960h` | Contrastive (wav2vec 2.0) | English only | ~95M |
| `microsoft/wavlm-base` | Denoising masking (WavLM) | English only | ~95M |
| `facebook/wav2vec2-xls-r-300m` | Contrastive (XLS-R) | 128 languages incl. Spanish | ~300M |

---

## Clinical Context

### Study Design

| Property | Detail |
|---|---|
| Target condition | Chronic Rhinosinusitis (CRS) |
| Intervention | Functional Endoscopic Sinus Surgery (FESS) |
| Comparison groups | Control, Septoplasty, Tonsillectomy |
| Longitudinal sessions | Session 1 (pre-op), Session 2 (2-week post-op), Session 3 (3-month post-op) |
| Patient-level splits | 20 train / 2 val / 5 test (no patient leakage) |
| Sample rate | 16 kHz |
| Language | Spanish |

### Audio Task Channels (13 types)

| Group | Columns | Clinical purpose |
|---|---|---|
| Vowels | a, e, i, o, u | Steady-state phonation quality |
| Sustained | a1, a2, a3 | Repeated phonation stability |
| Speech | speech | Connected discourse, prosodic patterns |
| TDU words | agua, brasero, dia, mesa | Word-level articulation (Spanish) |

### Research Hypothesis

CRS alters the vocal tract through mucosal inflammation and blocked nasal passages, producing measurable acoustic signatures in vowel quality, resonance, and airflow. If FESS resolves these changes, a model trained to detect CRS should also track post-surgical acoustic recovery — providing a non-invasive, objective complement to subjective symptom scores such as SNOT-22.

The domain mismatch between English-pretrained SSL models and Spanish clinical audio is a deliberate design choice and central finding: scratch training on in-domain data consistently outperforms monolingual finetune models, while the multilingual XLS-R backbone substantially closes this gap.

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
│   │   └── all_experiments.py      # Exp1–Exp5 class definitions
│   │
│   ├── training/
│   │   ├── trainer.py              # Training loop, AMP, Focal Loss, head warmup
│   │   ├── metrics.py              # Accuracy, F1, AUC, per-audio-type breakdown
│   │   ├── reporter.py             # Academic PDF report generator (individual + combined)
│   │   ├── svm_classifier.py       # SVM linear probe on frozen backbone embeddings
│   │   ├── imbalance.py            # Oversampling, weighted sampling, class weights
│   │   └── checkpoint.py           # Checkpoint inspection utilities
│   │
│   └── utils/paths.py              # CSV path resolution to audio files
```

---

## Preprocessing Pipeline

Executed via `scripts/run_preprocessing.py`. Produces `.pt` tensor files consumed directly by the training pipeline.

### Pipeline Stages

1. Load WAV → mono float32 tensor
2. Trim 50ms leading silence
3. Resample to 16 kHz
4. High-pass filter at 80 Hz (removes low-frequency handling noise)
5. Adaptive VAD (threshold = 5% of file peak RMS, floored at 0.01)
6. Peak amplitude normalisation → range [−1, +1]
7. Sliding window segmentation — **16,000 samples = 1.0s**
8. Instance normalisation (zero-mean, unit-variance per segment)
9. Save as `.pt` tensors

### Segment Naming Convention

```
ID{subject_id}_ses{session}_{audio_col}_seg{index:04d}.pt
```

### Why 1 Second — Not 3.072 Seconds

The theoretically ideal segment length for wav2vec 2.0 is 3.072 seconds (49,152 samples). This is because the model's CNN feature extractor downsamples by a factor of 320 (one latent frame per 20ms), and the positional convolution kernel is 128 frames wide. A segment must therefore produce at least 128 frames to keep the model within its pretraining distribution:

| Window | Duration | Samples | Transformer frames | Valid for pretraining range |
|---|---|---|---|---|
| Target | 3.072s | 49,152 | 153 | Yes |
| **Actual (this work)** | **1.0s** | **16,000** | **50** | **Below kernel width** |

The clinical audio recordings in this dataset are short phonation tasks — sustained vowels (a, e, i, o, u), repeated productions (a1, a2, a3), and single Spanish words (agua, brasero, dia, mesa). The majority of these recordings are naturally shorter than 3 seconds in duration. Applying a 3.072-second sliding window with 50% overlap would produce either zero segments for most recordings, or segments dominated by silence and zero-padding that would corrupt the acoustic signal used for training.

A 1-second window was selected as the longest window that produces a reliable, non-trivial number of segments across all audio types and all patients, validated empirically:

```
Extracted 34,368 segments from the full dataset
Segment stats: min=1.0s  max=1.0s  mean=1.0s  (50 transformer frames avg)
```

#### How 1-Second Segments Were Made to Work

Operating below the 128-frame kernel width means the model receives inputs outside the length range seen during pretraining. This is mitigated in three ways:

**1. Positional encoding extrapolation.** The wav2vec 2.0 and WavLM positional convolution (a grouped causal CNN) operates locally and degrades gracefully on shorter sequences — it does not fail catastrophically but produces slightly less well-conditioned position representations for short inputs. In practice the models converge stably on 50-frame inputs.

**2. Mean pooling.** The classifier aggregates the full sequence of 50 frame representations by mean pooling before the MLP head. Shorter sequences produce fewer frames to average, but the pooled representation still captures the per-frame spectral and phonetic features encoded by the backbone. The classification task depends on these averaged acoustic features, not on long-range temporal dependencies that require 128+ frames.

**3. Instance normalisation per segment.** Each segment is independently normalised to zero mean and unit variance before being passed to the backbone. This prevents the amplitude differences that arise from extracting a 1-second window from different positions within a recording from influencing the backbone's representations.

#### Impact on Each Backbone

| Backbone | Pretraining length | Impact of 1s input | Severity |
|---|---|---|---|
| wav2vec2-base | Trained on long utterances (LibriSpeech) | Below positional kernel width; local features intact but long-range temporal encoding less reliable | Moderate |
| WavLM-base | Same architecture and pretraining length as wav2vec2 | Same impact; denoising objective may make local features slightly more robust | Moderate |
| XLS-R-300M | Same CNN encoder, 24 transformer layers | Same kernel width constraint; larger model has more redundant capacity so shorter inputs may be partially compensated by deeper feature extraction | Moderate |

All three backbones are affected equally in terms of the positional encoding constraint. The relative rankings observed in the results — scratch outperforming finetune on all models — are therefore unlikely to be artefacts of the short segment length, since all models operate under the same constraint. The short segments do add noise to the finetune models' representations since the pretrained positional encodings are calibrated for longer inputs, which may partially explain why finetune underperforms scratch more than would be expected from domain mismatch alone.

This is acknowledged as a methodological limitation. Future work with access to longer uninterrupted phonation recordings could validate whether 3.072-second segments improve results and narrow the scratch vs finetune gap.

### Running Preprocessing in Colab

```python
from src.pipeline.preprocess import process_from_csv

process_from_csv(
    csv_path     = "/content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv",
    project_root = "/content/drive/MyDrive",
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
| **Exp 2** | Does session confound dominate the acoustic signal? | Binary | Pre-op vs post-op (session confound) |
| **Exp 3** | Can we track three stages of recovery from voice alone? | 3-class | Session 1 / 2 / 3 |
| **Exp 4** | Can we detect within-patient change without speaker identity? | Paired binary | Pre/post segment pairs, same patient |
| **Exp 5** | Do CRS-trained features generalise across surgical groups? | Binary cross-domain | Train: FESS — Test: Septoplasty + Tonsillectomy |

### Exp 4 — Paired Change Detection

Eliminates the speaker identity confound by training on pre/post pairs from the same patient. The model detects acoustic change without using voice identity as a shortcut.

**Positive pairs (label 1):** pre-op segment + post-op segment, same patient  
**Negative pairs (label 0):** two segments within the same recording session  
**Class balance:** `neg_ratio=2.0` produces approximately 33% positive / 67% negative pairs

### Exp 5 — Generalisation

The most scientifically honest experiment. No patient overlap between train and test populations by design. Results from Exp 5 provide the most reliable indication of whether the model has learned genuine CRS-related acoustic features rather than speaker identity or recording artefacts.

---

## Model Architecture

### Backbone Comparison

Three backbones are evaluated. XLS-R has a larger hidden dimension and deeper architecture:

| Backbone | Hidden size | Transformer layers | Scratch supported |
|---|---|---|---|
| wav2vec2-base | 768 | 12 | Yes |
| wavlm-base | 768 | 12 | Yes |
| XLS-R-300M | 1024 | 24 | No (multilingual pretraining is the point) |

### Wav2Vec2Classifier

```
Input waveform [B, 16000]       (1.0s at 16kHz)
        |
        v
Backbone (wav2vec2 / WavLM / XLS-R)
  CNN feature extractor
        |   [B, T, 512]
        v
  Transformer encoder
        |   [B, T, H]   H=768 or 1024
        v
  Mean pooling over time
        |   [B, H]
        v
  MLP Classification Head
    Linear(H -> 256)
    LayerNorm(256)
    ReLU
    Dropout(0.3)
    Linear(256 -> num_classes)
        |   [B, num_classes]
```

### SVM Linear Probe (finetune mode only)

After neural training, the frozen backbone produces mean-pooled embeddings fed to an RBF-SVM with `class_weight='balanced'`. This directly tests the transfer learning hypothesis: if frozen pretrained features + SVM outperform end-to-end finetune training, the pretrained representations carry clinically relevant structure that the MLP head alone fails to exploit on this small dataset.

Key finding: **SVM consistently outperforms the MLP head for finetune mode**, sometimes dramatically (Exp 5 wav2vec2: MLP F1=0.256 vs SVM F1=0.494).

Activate with `--use_svm`.

---

## Training Configuration

### Key Findings on Hyperparameters

The following configuration was used for final reported results:

```python
--num_epochs         20
--batch_size         16        # 8 for XLS-R (larger model)
--learning_rate      1e-5
--layerwise_lr_decay 0.8       # auto-adjusted to 0.9 for XLS-R (24 layers)
--label_smoothing    0.1
--head_warmup_epochs 1
--early_stop_metric  val_f1
--early_stop_patience 3        # 5 for XLS-R
--freeze_layers      4
--focal_gamma        2.0
--imbalance          weights
--use_svm
```

### Key ExperimentConfig Fields

| Field | Default | Description |
|---|---|---|
| `mode` | `finetune` | `scratch` or `finetune` |
| `backbone` | `wav2vec2` | `wav2vec2`, `wavlm`, or `xlsr` |
| `learning_rate` | `1e-5` | Head LR; backbone gets lower LR via layerwise decay |
| `layerwise_lr_decay` | `0.8` | Each lower transformer layer gets LR × decay^n |
| `head_warmup_epochs` | `1` | Epochs of backbone-frozen head warmup (finetune only) |
| `freeze_layers` | `4` | Number of bottom transformer layers to keep frozen |
| `early_stop_metric` | `val_f1` | `val_f1` or `val_loss` |
| `early_stop_patience` | `3` | Epochs without improvement before stopping |
| `focal_gamma` | `2.0` | Focal Loss focusing parameter |
| `label_smoothing` | `0.1` | Applied inside Focal Loss |
| `imbalance_strategy` | `weights` | `oversample`, `weighted`, or `none` |
| `save_every` | `5` | Checkpoint every N epochs |
| `keep_last_n` | `2` | Max epoch checkpoints on disk |

### Focal Loss

```
FL(p_t) = -(1 - p_t)^gamma * log(p_t)
```

At gamma=2, a sample predicted with 90% confidence contributes only 1% of its standard CrossEntropy gradient, forcing the model to focus on hard misclassified minority-class samples.

### AMP (Automatic Mixed Precision)

All training uses float16 forward passes with float32 loss computation via GradScaler. For XLS-R the initial GradScaler scale is reduced from 65536 to 16384 to prevent activation overflow in the deeper 24-layer network.

### Layer-wise LR Decay (finetune)

```
Classifier head       lr = 1e-5
Transformer layer 11  lr = 8.0e-6   (× 0.8^1)
Transformer layer 10  lr = 6.4e-6   (× 0.8^2)
        ...
Transformer layer 0   lr = 1.1e-6   (× 0.8^12)
```

XLS-R auto-adjusts from 0.8 to 0.9 decay (0.8^24 ≈ 0.005 is too aggressive; 0.9^24 ≈ 0.08 is appropriate).

---

## Output Structure

```
MSc_Sinusitis_results/
├── exp1_backbone_comparison/
│   ├── wav2vec2_scratch/
│   │   ├── checkpoints/
│   │   ├── plots/
│   │   ├── tables/
│   │   ├── results_summary.json
│   │   └── report.pdf              # Individual run report
│   ├── wav2vec2_finetune/
│   ├── wavlm_scratch/
│   ├── wavlm_finetune/
│   ├── xlsr_finetune/
│   ├── backbone_comparison.json    # All models combined metrics
│   └── report.pdf                  # Combined 8-page comparison report
├── exp2_backbone_comparison/
├── exp3_backbone_comparison/
├── exp4_backbone_comparison/
├── exp5_backbone_comparison/
└── training.log
```

### Combined Comparison Report (report.pdf)

The combined PDF report is auto-generated after each `--compare_backbones` run. It contains 8 pages:

| Page | Content |
|---|---|
| 1 | Cover + full metrics table (val/test acc/F1/AUC + SVM rows) |
| 2 | Test F1 + AUC bar charts — all models |
| 3 | Val F1 + AUC bar charts — all models |
| 4 | MLP head vs SVM probe — finetune models only |
| 5 | Per-class F1 comparison — test/val/train |
| 6 | Per-split summary table (loss, acc, F1, AUC per model) |
| 7 | Per-audio-type heatmap + per-model tables |
| 8 | Test confusion matrices |

---

## Google Colab Workflow

The full research pipeline is run from a single Colab notebook (`MSc_Experiments.ipynb`), organised into 13 stages. Each stage is a markdown header followed by one code cell. Stages are designed to be re-run safely — completed work (segments, checkpoints, results) is detected on Drive and skipped rather than redone.

| # | Stage | Purpose | Runtime |
|---|---|---|---|
| 1 | GPU and Drive mount | Confirm GPU, mount Google Drive | seconds |
| 2 | GitHub Token Authentication | Load PAT from Drive, configure git credential store | seconds |
| 3 | Clone and/or pull Repo | Clone on first run, hard-reset to `origin/main` on subsequent runs | seconds |
| 4 | EDA and Diagnostic Plots | Cohort/session/class-balance plots, window-size analysis | ~2 min (`--counts_only`) / ~30 min (full audio scan) |
| 5 | Preprocessing | WAV → clean `.pt` segments; restores from Drive zip if already processed | ~15–25 min (first run only) |
| 6 | Patient demographics | Cohort table, per-experiment split table, test-patient profiles → LaTeX | ~15 sec |
| 7 | Segment Validation | Confirms every experiment (1–5) has usable segments before training starts | seconds |
| 8 | Main training cell | Runs all 5 experiments × all backbones × both modes, with SVM probe | hours (GPU-dependent) |
| 9 | Training Results | Reads `results_summary.json` / `backbone_comparison.json`, prints run status and metrics table | seconds |
| 10 | Ablation Training | Runs one ablation factor group (`freeze` / `loss` / `decay`) per session | ~2.5–4 hrs per factor |
| 11 | Ablation Results | Aggregates all three factors from `ablation_results.json` → console tables + LaTeX | ~5 sec |
| 12 | Statistical Tests and results | Builds per-experiment predictions JSON, runs permutation / bootstrap / McNemar tests → `.tex` | seconds–minutes |
| 13 | Thesis Figures | Generates all programmatic figures (main results, appendix, ablation, synthesis) | ~3–4 min |

### 1 — GPU and Drive mount
```python
import torch
from google.colab import drive

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
drive.mount('/content/drive', force_remount=True)
```

### 2 — GitHub Token Authentication
```python
with open("/content/drive/MyDrive/secrets/github_token.txt") as f:
    TOKEN = f.read().strip()

!git config --global credential.helper store
with open("/root/.git-credentials", "w") as f:
    f.write(f"https://{TOKEN}:x-oauth-basic@github.com\n")
!chmod 600 /root/.git-credentials
```

### 3 — Clone and/or pull Repo
```python
import os

PROJECT_ROOT = "/content/project"
REPO_URL = "https://github.com/T-Machethe/MEng-Research-Project.git"

if not os.path.exists(PROJECT_ROOT):
    !git clone {REPO_URL} {PROJECT_ROOT}
else:
    %cd {PROJECT_ROOT}
    !git fetch --all
    !git reset --hard origin/main
    !git clean -fd

%cd {PROJECT_ROOT}
```

### 4 — EDA and Diagnostic Plots
Runs `run_exploratory_DA.py` (structure plots, optionally a full audio scan) and `run_diagnostic_pipeline.py --section b` (window-size / segment-yield analysis). Output goes to `MSc_Sinusitis_results/Plots and visuals/{eda,diagnostics}`.
```python
run("run_exploratory_DA.py", "--counts_only --csv_path '...' --data_root '...' --output_dir '...'")
run("run_diagnostic_pipeline.py", "--section b --csv_path '...' --data_root '...' --output_dir '...'")
```

### 5 — Preprocessing
Checks the runtime, then a Drive zip, before falling back to `process_from_csv(...)` on raw WAV files. Successful runs are re-zipped to `/content/drive/MyDrive/clean_audio_3s.zip` so future sessions skip extraction entirely. Ends with a segment-length sanity check (must exceed 3s worth of frames for the transformer kernel width, see [Why 1 Second — Not 3.072 Seconds](#why-1-second--not-3072-seconds)).

### 6 — Patient demographics
Produces three tables — overall cohort (n, age, gender per surgical group), split breakdown (train/val/test per experiment), and individual test-patient profiles with experiment coverage. Tables 1 and 2 are written as LaTeX to `thesis_outputs/patient_demographics.tex`.

### 7 — Segment Validation
Indexes all `.pt` files by `(patient_id, session, audio_column)` and checks each of the five experiments has both classes/groups present with usable segment counts. **A ✓ on every experiment is required before running Stage 8.**

### 8 — Main training cell
```python
cmd = [
    sys.executable, "scripts/run_experiment.py",
    "--exp", "all", "--compare_backbones",
    "--num_epochs", "30", "--batch_size", "8",
    "--imbalance", "weights", "--warmup_steps", "200",
    "--learning_rate", "1e-5", "--layerwise_lr_decay", "0.8",
    "--label_smoothing", "0.1", "--head_warmup_epochs", "1",
    "--early_stop_metric", "val_f1", "--early_stop_patience", "5",
    "--freeze_layers", "4", "--focal_gamma", "2.0",
    "--save_every", "5", "--keep_last_n", "2", "--use_svm",
    "--segment_dir", "/content/clean_audio_3s",
    "--csv_path", CSV_PATH, "--output_dir", OUTPUT_DIR,
]
```
Streamed via `subprocess.Popen` with HuggingFace load-noise suppressed and a `tqdm` bar for backbone weight loading. Already-completed runs are automatically skipped — re-running is safe and will only execute missing or incomplete experiments.

### 9 — Training Results
Walks every `expN_backbone_comparison/{run}/results_summary.json`, reporting status (`NOT STARTED` / `PARTIAL` / `COMPLETE`), checkpoint epoch, and test/val accuracy, F1, AUC (plus SVM metrics for finetune runs). Also prints a condensed summary straight from each experiment's `backbone_comparison.json`.

### 10 — Ablation Training
```python
ABLATION_FACTOR = "all"   # or "freeze" / "loss" / "decay"
# python scripts/run_ablation.py --run --factor {ABLATION_FACTOR} \
#     --segment_dir ... --csv_path ... --output_dir ... --num_epochs 8
```
Run one factor group per Colab session (`freeze` ≈ 4 hrs / 6 variants, `loss` and `decay` ≈ 2.5 hrs / 4 variants each). Progress accumulates in `ablation_results.json`; completed variants are skipped automatically on re-run.

### 11 — Ablation Results
Run only after `freeze`, `loss`, and `decay` have all completed. Reads `ablation_results.json`, prints console tables, and writes `thesis_outputs/ablation_tables.tex`.

### 12 — Statistical Tests and results
For each of the five experiments: builds a `expN_predictions.json` from `backbone_comparison.json` (with a fallback to individual `results_summary.json` files and full-collapse detection for degenerate runs), then runs `run_statistical_tests.py` — permutation tests, bootstrap CIs, and McNemar's tests — writing four `.tex` files per experiment (`stat_permutation_test`, `stat_bootstrap_ci`, `stat_mcnemar`, `stat_mcnemar_prose`) plus a `stat_manifest.json` summary.

### 13 — Thesis Figures
Verifies which `backbone_comparison.json` files and the ablation results are available, then runs `run_visualisations.py` to generate all programmatic figures (main results, appendix, ablation, cross-experiment synthesis) to `MSc_Sinusitis_results/Thesis_Figures/`. A single figure can be regenerated with `--figure N` after updating results.

---

## CLI Reference

```bash
# All experiments, all backbones (wav2vec2 + WavLM + XLS-R), with SVM
python scripts/run_experiment.py \
    --exp all --compare_backbones \
    --num_epochs 20 --batch_size 8 \
    --learning_rate 1e-5 \
    --head_warmup_epochs 1 \
    --early_stop_metric val_f1 \
    --focal_gamma 2.0 \
    --use_svm \
    --segment_dir /path/to/segments \
    --csv_path /path/to/clinical_all_sessions.csv \
    --output_dir /path/to/results

# Single experiment, single backbone
python scripts/run_experiment.py \
    --exp 1 --backbone wavlm --mode finetune \
    --num_epochs 20 --use_svm

# XLS-R only across all experiments
python scripts/run_experiment.py \
    --exp all --backbone xlsr --mode finetune \
    --num_epochs 20 --batch_size 8 \
    --early_stop_patience 5 --use_svm
```

### Full Argument Reference

| Argument | Default | Description |
|---|---|---|
| `--exp` | required | `1`–`5` or `all` |
| `--backbone` | `wav2vec2` | `wav2vec2`, `wavlm`, or `xlsr` |
| `--mode` | `finetune` | `scratch` or `finetune` |
| `--compare_backbones` | False | Run all backbones × modes, generate combined report |
| `--num_epochs` | 20 | Max training epochs |
| `--batch_size` | 16 | Use 8 for XLS-R on T4 |
| `--learning_rate` | 1e-5 | Head LR |
| `--layerwise_lr_decay` | 0.8 | Finetune backbone LR decay per layer |
| `--head_warmup_epochs` | 1 | Backbone-frozen warmup epochs |
| `--freeze_layers` | 4 | Bottom transformer layers to freeze |
| `--early_stop_metric` | `val_f1` | `val_f1` or `val_loss` |
| `--early_stop_patience` | 3 | Epochs without improvement |
| `--focal_gamma` | 2.0 | Focal Loss focusing parameter |
| `--label_smoothing` | 0.1 | Label smoothing coefficient |
| `--imbalance` | `weights` | `oversample`, `weights`, or `none` |
| `--warmup_steps` | 200 | LR warmup steps |
| `--save_every` | 5 | Epoch checkpoint frequency |
| `--keep_last_n` | 2 | Max epoch checkpoints retained |
| `--use_svm` | False | Run SVM probe on frozen finetune embeddings |
| `--svm_C` | 1.0 | SVM regularisation |
| `--svm_kernel` | `rbf` | `rbf` or `linear` |
| `--verbose` | False | Full INFO log to console |
| `--seed` | 42 | Random seed |

---

## Evaluation Framework

### Metrics Per Split

- Accuracy, F1-macro, F1 per class, ROC-AUC (macro for multiclass), confusion matrix

### Per-Audio-Type Breakdown

Predictions are grouped by audio column (vowel, sustained, speech, TDU word) to report separate metrics for each type. Identifies which acoustic task carries the most diagnostic signal per experiment.

### SVM vs MLP Comparison

For finetune runs, both the neural MLP head result and the SVM linear probe result are reported side by side. The SVM result is the headline finetune number in the assignment (reported as "finetune backbone + linear probe"). The MLP result is reported as the end-to-end finetune result.

---

## Key Results (Current)

| Run | Test F1 | Test AUC | SVM F1 | SVM AUC |
|---|---|---|---|---|
| **Exp1** (binary sinusitis detection) | | | | |
| wav2vec2 scratch | **0.7605** | **0.8313** | — | — |
| wav2vec2 finetune | 0.5671 | 0.6654 | 0.5622 | 0.6365 |
| WavLM scratch | 0.6465 | 0.7673 | — | — |
| WavLM finetune | 0.5792 | 0.6605 | 0.5844 | 0.6533 |
| XLS-R finetune | 0.6962 | **0.7909** | 0.6848 | 0.7770 |
| **Exp2** (session confound) | | | | |
| wav2vec2 scratch | 0.5086 | 0.5171 | — | — |
| WavLM scratch | 0.5163 | 0.5234 | — | — |
| XLS-R finetune (SVM) | — | — | **0.5701** | **0.6602** |
| **Exp5** (cross-population generalisation) | | | | |
| wav2vec2 scratch | 0.5157 | 0.5247 | — | — |
| WavLM scratch | 0.5200 | 0.5413 | — | — |

**Central finding:** Scratch training outperforms monolingual finetune across all experiments and both base backbones. XLS-R's multilingual pretraining substantially closes the domain gap — achieving AUC 0.7909 on Exp1 vs 0.6654 for wav2vec2 finetune — supporting the argument that language coverage in pretraining matters for cross-lingual clinical audio transfer.

---

## Known Limitations

1. **Small patient cohort** — 27 patients per group, 20 in training after splits. Segment-level count (8k–50k) inflates apparent data size but all segments from the same patient stay on the same side of every split boundary.

2. **Domain mismatch** — all backbones were pretrained on English speech. Clinical audio is Spanish, hospital-recorded. XLS-R partially addresses this through multilingual pretraining on 128 languages including Spanish.

3. **Val set size** — only 2 validation patients, making early stopping signal very noisy. Val F1 = 0.4111 across all Exp4 models reflects the majority-class baseline on 2 patients with a 30/70 class distribution, not model failure.

4. **Session confound (Exp 2, Exp 4)** — pre-op and post-op recordings may differ in room acoustics and patient state independently of surgical effect. Exp 2 near-random results confirm session dominates over clinical signal.

5. **Segment independence assumption** — overlapping sliding window segments share acoustic context but are treated as independent training examples.

6. **XLS-R AMP instability** — the 24-layer XLS-R architecture is prone to float16 overflow with the default GradScaler scale. Mitigated by reducing `init_scale` from 65536 to 16384.

---

## Research Contribution

This system provides structured empirical evidence on three questions:

1. **Are CRS-related vocal signatures learnable from raw speech?** Exp 1 demonstrates above-chance binary classification (AUC 0.83) using only voice recordings and no hand-crafted features.

2. **Does transfer learning from SSL speech models help or hurt on domain-mismatched clinical audio?** The three-way comparison (scratch / finetune MLP / SVM linear probe) shows monolingual finetune consistently underperforms scratch, while SVM on frozen features recovers much of the gap — indicating the pretrained representations contain relevant structure that end-to-end finetune disrupts on small datasets.

3. **Does multilingual pretraining reduce the domain mismatch penalty?** Monolingual finetune models (wav2vec2-FT AUC 0.67, WavLM-FT AUC 0.66) both fall well below their scratch counterparts (wav2vec2-scratch AUC 0.83, WavLM-scratch AUC 0.77), confirming that English-only pretraining actively hurts performance on Spanish clinical audio. XLS-R — pretrained on 128 languages including Spanish — achieves AUC 0.79 in finetune mode, substantially closing the gap toward scratch performance and outperforming all monolingual finetune models by a wide margin. This demonstrates that multilingual pretraining meaningfully reduces but does not eliminate the domain mismatch penalty, and that language coverage in SSL pretraining is a significant factor in cross-lingual clinical audio transfer.

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
Use `--batch_size 8` for XLS-R. Base models run at batch size 16.

---

## Reproducibility

All experiments are deterministic under a fixed seed. Patient-level split assignment is fully deterministic given the same CSV and seed.

To resume a crashed or interrupted run:

```bash
python scripts/run_experiment.py --exp 1 --compare_backbones \
    --output_dir /path/to/existing/results
# Automatically skips completed runs, resumes from latest_checkpoint.pt
```

---

## Citation

If you use this repository in academic work, please cite:

> Machethe, T. D. (2026).  
> *Improving sinusitis diagnosis from voice recordings using transfer learning.*  
> Master’s research assignment, Stellenbosch University.

---

## Author

**Tumelo Machethe**  
Master’s Student, Industrial Engineering  
Stellenbosch University

---

## License

This repository is provided for academic and research purposes only.

For commercial use, redistribution, or derivative applications, please contact the author.

---

<div align="center">

### Research Assignment

Presented in partial fulfilment of the requirements for the degree of  
Master of Engineering (Structured) (Industrial Engineering)  
in the Faculty of Engineering at Stellenbosch University.

</div>