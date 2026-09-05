"""
scripts/run_retroactive_svm.py
─────────────────────────────────────────────────────────────────────────────
Retroactively fits the SVM head onto ALREADY-TRAINED Exp1 checkpoints.

NO retraining of the neural network happens here. Two things changed after
your Exp1 training run already completed:
  1. SVM now also runs in scratch mode (previously finetune-only).
  2. The fitted SVM + StandardScaler are now persisted to disk
     (svm/svm_model.joblib) so Exp5 can load and run inference through
     them — previously only metrics were saved, never the model itself.

Neither change requires the underlying neural network to be retrained.
The SVM has always been a POST-HOC step (see src/training/svm_classifier
.py): it extracts frozen embeddings from an already-trained backbone
(one forward pass per segment, no gradients) and fits a separate sklearn
classifier on top. This script:

  1. Rebuilds the exact deterministic train/val/test split Exp1 used —
     via Exp1CRSvsControl.prepare(), seed=42 — the same mechanism
     scripts/run_patient_level_aggregation.py already relies on.
  2. Loads each already-saved best_model.pt (untouched — no gradient
     steps taken against it here).
  3. Runs ONLY train_svm() against it.
  4. Merges the result into the EXISTING results_summary.json under a
     new "svm" key. Every other key (test/accuracy, test/patient_level/*,
     training_history, ...) is left exactly as it was. A timestamped
     backup of the original file is written first, so this is always
     reversible.

Your existing neural (MLP) results are not touched or recomputed.

Note: this does NOT update backbone_comparison.json (the consolidated,
cross-backbone summary written once at the end of the original
--compare_backbones run) — that file is already a known-unsafe source
for arrays (see project notes: it truncates via json.dump(default=str)).
Downstream scripts (statistical tests, Exp5) already prefer each
backbone's own results_summary.json for exactly this reason, so this is
consistent with existing convention, not a shortcut.

Usage
──────
    python scripts/run_retroactive_svm.py \\
        --csv_path     /content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv \\
        --segment_dir  /content/clean_audio_3s \\
        --results_dir  /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback \\
        --device       cuda

Optional: --backbones wav2vec2_scratch,wavlm_scratch restricts to a
subset (e.g. just the scratch-mode backbones, since finetune-mode SVMs
may already exist from your original run — this script skips a backbone
entirely if svm/svm_model.joblib already exists for it, unless
--overwrite is passed, so it's safe to just run against all six).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path as _Path
from typing import Dict, List, Optional

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.experiments.base import ExperimentConfig
from src.experiments.all_experiments import Exp1CRSvsControl
from src.training.checkpoint import load_checkpoint
from src.training.patient_metrics import (
    patient_ids_from_samples, audio_types_from_samples, aggregate_to_patient_level,
    compute_patient_level_metrics,
)

# Must match src/experiments/base.py::EXPERIMENT_MAP registration and
# scripts/run_cross_cohort_specificity.py's BACKBONE_JOBS — six backbones,
# NOT a curated subset. Duplicated here (rather than imported from the
# sibling script) to avoid relying on scripts/ being importable as a
# package from an arbitrary Colab working directory.
BACKBONE_JOBS = [
    # (output_dir_name, backbone_type, training_mode, pretrained_id)
    ("wav2vec2_scratch",  "wav2vec2", "scratch",  None),
    ("wav2vec2_finetune", "wav2vec2", "finetune", "facebook/wav2vec2-base-960h"),
    ("wavlm_scratch",     "wavlm",    "scratch",  None),
    ("wavlm_finetune",    "wavlm",    "finetune", "microsoft/wavlm-base"),
    ("xlsr_scratch",      "xlsr",     "scratch",  None),
    ("xlsr_finetune",     "xlsr",     "finetune", "facebook/wav2vec2-xls-r-300m"),
]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def build_inference_model(backbone_type: str, mode: str, pretrained: Optional[str],
                           num_classes: int = 2):
    """
    Mirrors src/training/trainer.py::Trainer._build_model() exactly —
    same function used in scripts/run_cross_cohort_specificity.py, kept
    as a local duplicate for the same import-robustness reason noted
    there.
    """
    from transformers import (
        Wav2Vec2Model, Wav2Vec2Config, WavLMModel, WavLMConfig,
    )
    from src.training.trainer import Wav2Vec2Classifier

    if mode == "scratch":
        if backbone_type == "xlsr":
            config = Wav2Vec2Config(hidden_size=1024, num_hidden_layers=24,
                                     num_attention_heads=16, intermediate_size=4096)
            backbone = Wav2Vec2Model(config)
        elif backbone_type == "wavlm":
            config = WavLMConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
            backbone = WavLMModel(config)
        else:
            config = Wav2Vec2Config(hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
            backbone = Wav2Vec2Model(config)
    else:
        if backbone_type == "wavlm":
            backbone = WavLMModel.from_pretrained(pretrained, mask_time_prob=0.0, mask_feature_prob=0.0)
        else:
            backbone = Wav2Vec2Model.from_pretrained(pretrained, mask_time_prob=0.0, mask_feature_prob=0.0)

    hidden_size = backbone.config.hidden_size
    model = Wav2Vec2Classifier(backbone, hidden_size, num_classes)
    return model


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def run_one_backbone(job_name: str, backbone_type: str, mode: str, pretrained: Optional[str],
                      csv_path: str, segment_dir: str, results_dir: _Path,
                      project_root: _Path, batch_size: int, svm_C: float, svm_kernel: str,
                      test_size: float, val_size: float, seed: int,
                      device: torch.device, overwrite: bool) -> bool:
    job_dir = results_dir / "exp1_backbone_comparison" / job_name
    ckpt_path = job_dir / "best_model.pt"
    summary_path = job_dir / "results_summary.json"
    svm_path = job_dir / "svm" / "svm_model.joblib"

    if not ckpt_path.exists():
        log.warning(f"  {job_name}: no best_model.pt at {ckpt_path} — skipping.")
        return False
    if not summary_path.exists():
        log.warning(f"  {job_name}: no results_summary.json at {summary_path} — "
                    f"skipping (need existing MLP results to merge into).")
        return False
    if svm_path.exists() and not overwrite:
        log.info(f"  {job_name}: svm_model.joblib already exists — skipping "
                 f"(pass --overwrite to refit anyway).")
        return False

    log.info(f"\n{'─'*70}\n  {job_name}  (mode={mode})\n{'─'*70}")

    # ── Load the already-trained neural model (no gradient steps here) ────
    model = build_inference_model(backbone_type, mode, pretrained)
    load_checkpoint(str(ckpt_path), model)
    model.to(device)

    # ── Rebuild the EXACT deterministic split Exp1 used ────────────────────
    cfg = ExperimentConfig(
        project_root=str(project_root), segment_dir=segment_dir, csv_path=csv_path,
        mode=mode, backbone=backbone_type, pretrained=pretrained or "facebook/wav2vec2-base-960h",
        batch_size=batch_size, test_size=test_size, val_size=val_size, seed=seed,
        imbalance_strategy="weights",
    )
    exp = Exp1CRSvsControl(cfg)
    exp.prepare()   # builds train/val/test loaders only — no training

    # ── Fit + evaluate the SVM (the only thing that actually runs here) ────
    from src.training.svm_classifier import train_svm
    svm_results = train_svm(
        model=model, train_loader=exp.train_loader, val_loader=exp.val_loader,
        test_loader=exp.test_loader, num_classes=2, output_dir=str(job_dir),
        device=device, C=svm_C, kernel=svm_kernel,
    )

    # ── Patient-level SVM metrics, same convention as base.py's hook ───────
    test_ds = exp.test_loader.dataset
    if hasattr(test_ds, "samples") and "test/all_probs" in svm_results:
        patient_ids = patient_ids_from_samples(test_ds.samples)
        audio_types = audio_types_from_samples(test_ds.samples)
        probs  = np.asarray(svm_results["test/all_probs"])
        labels = np.asarray(svm_results["test/all_labels"])
        if len(patient_ids) == len(probs):
            patient_df = aggregate_to_patient_level(probs, patient_ids, labels, recording_ids=audio_types)
            patient_metrics = compute_patient_level_metrics(patient_df, num_classes=2, split_name="test")
            svm_results.update(patient_metrics)
            svm_results["test/patient_level/per_patient"] = patient_df.to_dict(orient="records")
            log.info(f"  [patient-level][SVM] {len(patient_df)} patients — "
                     f"accuracy={patient_metrics.get('test/patient_level/accuracy', float('nan')):.4f}")

    # ── Merge into existing results_summary.json — backup first ────────────
    with open(summary_path) as f:
        existing = json.load(f)

    backup_path = job_dir / f"results_summary.backup-{datetime.now():%Y%m%d-%H%M%S}.json"
    shutil.copy(summary_path, backup_path)
    log.info(f"  Backed up original → {backup_path.name}")

    existing["svm"] = svm_results
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2, cls=_NumpyEncoder)
    log.info(f"  Updated {summary_path} (added 'svm' key, everything else unchanged)")

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv_path", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" / "Clinical" / "clinical_all_sessions.csv"))
    parser.add_argument("--segment_dir", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" / "clean_audio"))
    parser.add_argument("--results_dir", type=str,
                         default=str(PROJECT_ROOT / "MSc_Sinusitis_results"))
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--backbones", type=str, default=None,
                         help="Comma-separated subset, e.g. wav2vec2_scratch,wavlm_scratch. Default: all six.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--svm_C", type=float, default=1.0)
    parser.add_argument("--svm_kernel", type=str, default="rbf")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true",
                         help="Refit even if svm_model.joblib already exists.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    results_dir = _Path(args.results_dir)
    project_root = _Path(args.project_root)

    jobs = BACKBONE_JOBS
    if args.backbones:
        wanted = set(args.backbones.split(","))
        jobs = [j for j in BACKBONE_JOBS if j[0] in wanted]
        missing = wanted - {j[0] for j in jobs}
        if missing:
            log.warning(f"  Unknown backbone name(s) ignored: {missing}")

    log.info(f"Device: {device}")
    log.info(f"Running retroactive SVM fit for: {[j[0] for j in jobs]}")

    done = 0
    for job_name, backbone_type, mode, pretrained in jobs:
        ok = run_one_backbone(
            job_name, backbone_type, mode, pretrained,
            args.csv_path, args.segment_dir, results_dir, project_root,
            args.batch_size, args.svm_C, args.svm_kernel,
            args.test_size, args.val_size, args.seed, device, args.overwrite,
        )
        done += int(ok)

    log.info(f"\n{'='*70}\nDone. SVM fitted/updated for {done}/{len(jobs)} backbones.\n{'='*70}")


if __name__ == "__main__":
    main()
