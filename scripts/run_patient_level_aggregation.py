"""
scripts/run_patient_level_aggregation.py
─────────────────────────────────────────────────────────────────────────────
Aggregates Exp1's TEST predictions back to patient-level RETROACTIVELY (no
re-inference — reuses each backbone's already-saved results_summary.json),
and VAL predictions via a fresh, cheap INFERENCE PASS (not retraining — see
below for why val can't reuse the saved JSON the same way test can). Also
computes the patient-level, per-audio-type breakdown for both splits.

Neither the val inference nor anything else here trains anything. Loading
best_model.pt and running it in eval() mode over a DataLoader is a forward
pass, not a training step — same order of cost as the retroactive SVM
script (minutes, not hours), not comparable to re-running Exp1.

Why test is free but val needs a fresh pass
──────────────────────────────────────────────
test/all_probs in results_summary.json was already computed against the
reloaded best_model.pt (see Trainer.fit()), so reprocessing it is exact
and free. val/all_probs, before a recent trainer.py fix, reflected
whatever epoch the training loop happened to stop at — with early
stopping, that can be several epochs AFTER the actual best checkpoint —
so for any results_summary.json generated before that fix, the saved
val/all_probs may not correspond to best_model.pt. Rather than trust
possibly-misaligned data, this script re-evaluates val fresh against the
checkpoint that's actually on disk. Pass --skip_val to skip this and
save the inference time, if you know your results were generated after
the trainer.py fix (check for a "Loading best model for test
evaluation..." log line followed by val being re-evaluated in the same
place, in your original training log).

How the retroactive TEST recovery works
──────────────────────────────────────────
1. patient_level_split(seed=42) is purely deterministic — re-running it on
   the original CSV via Exp1CRSvsControl(cfg).prepare_data() reproduces the
   EXACT SAME val_df/test_df used in the original training run, with no
   need for a saved split manifest.
2. build_experiment_loaders() builds loaders with shuffle=False (confirmed
   in src/pipeline/dataloader.py). So SinusitisDataset(df, segment_dir,
   label_fn).samples — built by iterating the same df rows over the same
   segment_dir — is in the SAME ORDER the original loader iterated it in.
3. Therefore: rebuild that dataset now, parse each sample's patient ID
   (filename) and audio_col (already a tuple element — see
   src/training/patient_metrics.py::audio_types_from_samples), and zip
   those against the saved test/all_probs / test/all_labels arrays.
   Order-matched, no re-inference required for test.

This ONLY works if segment_dir and audio_cols are unchanged since the
original run. The script asserts length-matches and refuses to proceed on
mismatch rather than silently misaligning IDs to predictions.

Aggregation method
────────────────────
Mean predicted P(CRS) across a patient's segments (soft aggregation), not
majority vote — same convention used throughout this repo. For the
audio-type breakdown, mean is taken within each (patient, audio_type)
group separately — see src/training/patient_metrics.py.

CLI
───
    --csv_path         path to clinical_all_sessions.csv
    --segment_dir      segment dir used for the ORIGINAL Exp1 training run
    --results_dir      root containing exp1_backbone_comparison/
    --output_dir       where to write patient-level results (default:
                       <results_dir>/exp1_patient_level)
    --threshold        decision threshold on mean P(CRS) (default: 0.5)
    --skip_val         skip the fresh val inference pass (use only if you
                       know your results predate the misalignment issue,
                       or don't need val patient-level numbers at all)
    --skip_audio_type  skip the per-audio-type breakdown (cheaper, test-only
                       output closer to what this script used to produce)
    --device           cuda | cpu | auto (default: auto)
    --batch_size       inference batch size for the val pass (default: 16)
    --test_size / --val_size / --seed : must match the original training
                       run's split config (defaults: 0.20 / 0.10 / 42)

Usage
─────
    python scripts/run_patient_level_aggregation.py \\
        --csv_path /content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv \\
        --segment_dir /content/clean_audio_3s \\
        --results_dir /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.experiments.base import ExperimentConfig
from src.experiments.all_experiments import Exp1CRSvsControl
from src.pipeline.dataset import SinusitisDataset
from src.training.checkpoint import load_checkpoint
from src.training.patient_metrics import (
    patient_ids_from_samples, audio_types_from_samples,
    aggregate_to_patient_level as _shared_aggregate,
    compute_patient_level_metrics,
    aggregate_to_patient_audiotype_level, compute_patient_audiotype_metrics,
)

# (job_name, backbone_type, mode, pretrained_id) — matches the convention
# used in scripts/run_cross_cohort_specificity.py and run_retroactive_svm.py.
# Not a curated subset — Exp1 trains and saves results for all six.
BACKBONE_JOBS = [
    ("wav2vec2_scratch",  "wav2vec2", "scratch",  None),
    ("wav2vec2_finetune", "wav2vec2", "finetune", "facebook/wav2vec2-base-960h"),
    ("wavlm_scratch",     "wavlm",    "scratch",  None),
    ("wavlm_finetune",    "wavlm",    "finetune", "microsoft/wavlm-base"),
    ("xlsr_scratch",      "xlsr",     "scratch",  None),
    ("xlsr_finetune",     "xlsr",     "finetune", "facebook/wav2vec2-xls-r-300m"),
]


class _NumpyEncoder(json.JSONEncoder):
    """Same convention as src/experiments/base.py::BaseExperiment.report()."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_inference_model(backbone_type: str, mode: str, pretrained: Optional[str],
                           num_classes: int = 2):
    """
    Mirrors src/training/trainer.py::Trainer._build_model() exactly —
    duplicated here rather than imported, same import-robustness reasoning
    as the other scripts that need this (avoids relying on scripts/ being
    importable as a package from an arbitrary Colab working directory).
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


def rebuild_exp1_val_test_dfs(csv_path: str, project_root: _Path,
                               segment_dir: _Path, test_size: float,
                               val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reuses Exp1CRSvsControl.prepare_data() directly rather than
    re-implementing the filter/split logic — same pattern established in
    run_mfcc_svm_baseline.py. Guarantees the Session-1-only filtering
    (both arms) matches whatever Exp1 actually used, even if that logic
    changes again later. Returns (val_df, test_df) — train_df is unused
    here.
    """
    cfg = ExperimentConfig(
        project_root=str(project_root), segment_dir=str(segment_dir),
        csv_path=str(csv_path), test_size=test_size, val_size=val_size, seed=seed,
    )
    exp = Exp1CRSvsControl(cfg)
    _train_df, val_df, test_df = exp.prepare_data()[:3]
    log.info(
        f"\n  Rebuilt Exp1 splits: val={len(val_df)} rows/"
        f"{val_df['ID'].nunique()} patients, "
        f"test={len(test_df)} rows/{test_df['ID'].nunique()} patients "
        f"(seed={seed}, test_size={test_size}, val_size={val_size})"
    )
    return val_df, test_df


def build_dataset_and_ids(df: pd.DataFrame, segment_dir: str):
    """
    Rebuilds the SAME SinusitisDataset a shuffle=False loader over `df`
    would iterate, and parses patient ID + audio_col in that same order —
    via the shared helpers in src/training/patient_metrics.py.
    """
    dataset = SinusitisDataset(
        df, segment_dir=segment_dir, label_fn=lambda row: int(row["label"]),
    )
    patient_ids = patient_ids_from_samples(dataset.samples)
    audio_types = audio_types_from_samples(dataset.samples)
    return dataset, patient_ids, audio_types


def load_saved_test_predictions(results_summary_path: _Path):
    """TEST only — see module docstring for why test can be reused as-is
    but val cannot."""
    with open(results_summary_path) as f:
        results = json.load(f)

    if "test/all_probs" not in results or "test/all_labels" not in results:
        raise KeyError(
            f"{results_summary_path} has no 'test/all_probs' / "
            f"'test/all_labels' keys — was this run with an older trainer "
            f"version, or is this the wrong JSON (e.g. the unsafe, "
            f"truncating backbone_comparison.json rather than a per-run "
            f"results_summary.json)?"
        )

    probs  = np.array(results["test/all_probs"], dtype=np.float32)
    labels = np.array(results["test/all_labels"], dtype=int)
    return probs, labels, results


def run_val_inference(backbone_type: str, mode: str, pretrained: Optional[str],
                       ckpt_path: _Path, val_dataset: SinusitisDataset,
                       device: torch.device, batch_size: int) -> np.ndarray:
    """
    Fresh forward pass (no gradients, no optimizer — not training) over
    val, using the checkpoint that's actually on disk right now. See
    module docstring for why this can't just reuse results_summary.json's
    saved val/all_probs the way test can.
    """
    from torch.utils.data import DataLoader
    from src.pipeline.dataloader import collate_standard

    model = build_inference_model(backbone_type, mode, pretrained)
    load_checkpoint(str(ckpt_path), model)
    model.to(device)
    model.eval()

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_standard)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_values = torch.nan_to_num(
                batch["input_values"].to(device),
                nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_values=input_values, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 2), dtype=np.float32)


def aggregate_to_patient_level(probs: np.ndarray, labels: np.ndarray,
                                patient_ids: List[str]) -> pd.DataFrame:
    """
    Thin wrapper around the shared aggregate_to_patient_level(), renaming
    its generic p_class0/p_class1 columns to this script's p_crs_mean
    convention and adding std (useful for spotting low-confidence
    patients — not produced by the shared helper, which is mean-only).
    """
    assert len(probs) == len(labels) == len(patient_ids), (
        f"Length mismatch: probs={len(probs)}, labels={len(labels)}, "
        f"patient_ids={len(patient_ids)} — segment_dir/audio_cols likely "
        f"differ from the original training run, so ordering can't be "
        f"trusted. Re-check --segment_dir against what Exp1 actually used."
    )
    try:
        agg = _shared_aggregate(probs, patient_ids, labels)
    except ValueError as e:
        log.warning(f"  {e}")
        rec = pd.DataFrame({"ID": patient_ids, "label": labels})
        conflicted = rec.groupby("ID")["label"].nunique()
        conflicted = conflicted[conflicted > 1].index.tolist()
        log.warning(f"  Proceeding using first-seen label per patient for: {conflicted}")
        agg = _shared_aggregate(probs, patient_ids, labels=None)
        first_labels = rec.groupby("ID")["label"].first()
        agg["label"] = agg["ID"].map(first_labels)

    std_df = pd.DataFrame({"ID": patient_ids, "p_crs": probs[:, 1]})
    std_by_id = std_df.groupby("ID")["p_crs"].std()
    agg["p_crs_std"] = agg["ID"].map(std_by_id)

    agg = agg.rename(columns={"p_class1": "p_crs_mean"}).drop(columns=["p_class0"])
    return agg[["ID", "p_crs_mean", "p_crs_std", "n_segments", "label"]]


def patient_level_metrics(patient_df: pd.DataFrame, threshold: float,
                           split_name: str = "test") -> Dict:
    """
    Thin wrapper around the shared compute_patient_level_metrics(), which
    expects p_class{0,1} columns — remapped from this script's
    p_crs_mean/label convention, then to "patient_level_<split>/*" key
    prefix for this script's output.
    """
    shared_input = pd.DataFrame({
        "ID":      patient_df["ID"],
        "p_class0": 1 - patient_df["p_crs_mean"],
        "p_class1": patient_df["p_crs_mean"],
        "label":   patient_df["label"],
    })
    raw = compute_patient_level_metrics(shared_input, num_classes=2, split_name=split_name)
    metrics = {
        k.replace(f"{split_name}/patient_level", f"patient_level_{split_name}"): v
        for k, v in raw.items()
    }
    metrics[f"patient_level_{split_name}/n_patients"] = int(len(patient_df))
    metrics[f"patient_level_{split_name}/threshold"]  = float(threshold)
    return metrics


def process_split(split_name: str, probs: np.ndarray, labels: np.ndarray,
                   patient_ids: List[str], audio_types: List[str],
                   threshold: float, skip_audio_type: bool) -> Dict:
    """One split's worth of patient-level + audio-type-level results."""
    patient_df = aggregate_to_patient_level(probs, labels, patient_ids)
    metrics = patient_level_metrics(patient_df, threshold, split_name)

    result = {
        "patient_level_metrics": metrics,
        "per_patient": patient_df.to_dict(orient="records"),
    }

    if not skip_audio_type:
        at_df = aggregate_to_patient_audiotype_level(probs, patient_ids, audio_types, labels)
        at_metrics = compute_patient_audiotype_metrics(at_df, num_classes=2, split_name=split_name)
        result["per_audio_type"] = at_df.to_dict(orient="records")
        result["per_audio_type_metrics"] = at_metrics.to_dict(orient="records")

    return result


def merge_into_results_summary(job_dir: _Path, split_name: str, split_result: Dict):
    """
    Merges patient-level (+ audio-type) results into the backbone's
    EXISTING results_summary.json, backing up the original first — same
    pattern as run_retroactive_svm.py. Only adds keys under
    'test/patient_level/*' or 'val/patient_level/*'; every other key is
    left untouched.
    """
    summary_path = job_dir / "results_summary.json"
    if not summary_path.exists():
        log.warning(f"  No results_summary.json at {summary_path} — skipping merge.")
        return

    with open(summary_path) as f:
        existing = json.load(f)

    backup_path = job_dir / f"results_summary.backup-{datetime.now():%Y%m%d-%H%M%S}.json"
    shutil.copy(summary_path, backup_path)

    metrics = split_result["patient_level_metrics"]
    for k, v in metrics.items():
        # patient_level_test/accuracy -> test/patient_level/accuracy
        prefix = f"patient_level_{split_name}"
        if k.startswith(prefix):
            new_key = k.replace(prefix, f"{split_name}/patient_level", 1)
            existing[new_key] = v
    existing[f"{split_name}/patient_level/per_patient"] = split_result["per_patient"]
    if "per_audio_type" in split_result:
        existing[f"{split_name}/patient_level/by_audio_type"] = split_result["per_audio_type"]
        existing[f"{split_name}/patient_level/by_audio_type_metrics"] = split_result["per_audio_type_metrics"]

    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2, cls=_NumpyEncoder)
    log.info(f"  Merged {split_name} patient-level results -> {summary_path} "
             f"(backup: {backup_path.name})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv_path", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" /
                                     "Clinical" / "clinical_all_sessions.csv"))
    parser.add_argument("--segment_dir", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" /
                                     "clean_audio"))
    parser.add_argument("--results_dir", type=str,
                         default=str(PROJECT_ROOT / "MSc_Sinusitis_results"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_val", action="store_true",
                         help="Skip the fresh val inference pass (cheaper, but see "
                              "module docstring for the misalignment risk this avoids fixing).")
    parser.add_argument("--skip_audio_type", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    project_root = _Path(args.project_root)
    results_dir  = _Path(args.results_dir)
    output_dir   = _Path(args.output_dir) if args.output_dir else \
        results_dir / "exp1_patient_level"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    val_df, test_df = rebuild_exp1_val_test_dfs(
        args.csv_path, project_root, _Path(args.segment_dir),
        args.test_size, args.val_size, args.seed,
    )
    test_dataset, test_patient_ids, test_audio_types = build_dataset_and_ids(
        test_df, args.segment_dir)
    if not args.skip_val:
        val_dataset, val_patient_ids, val_audio_types = build_dataset_and_ids(
            val_df, args.segment_dir)

    combined = {}
    rows_for_comparison_table = []

    for job_name, backbone_type, mode, pretrained in BACKBONE_JOBS:
        job_dir = results_dir / "exp1_backbone_comparison" / job_name
        summary_path = job_dir / "results_summary.json"
        if not summary_path.exists():
            log.warning(f"  {job_name}: no results_summary.json found at {summary_path} — skipping.")
            continue

        log.info(f"\n{'─'*70}\n  {job_name}\n{'─'*70}")

        # ── TEST: free, reprocess existing saved predictions ────────────────
        try:
            test_probs, test_labels, _raw = load_saved_test_predictions(summary_path)
        except KeyError as e:
            log.warning(f"  {e}")
            continue
        try:
            test_result = process_split(
                "test", test_probs, test_labels, test_patient_ids, test_audio_types,
                args.threshold, args.skip_audio_type,
            )
        except AssertionError as e:
            log.error(f"  test: {e}")
            continue

        tm = test_result["patient_level_metrics"]
        log.info(f"  TEST  patient-level acc={tm.get('patient_level_test/accuracy', 0):.4f}  "
                 f"f1={tm.get('patient_level_test/f1_macro', 0):.4f}  "
                 f"(n={tm.get('patient_level_test/n_patients')} patients)")

        merge_into_results_summary(job_dir, "test", test_result)

        row = {
            "backbone": job_name,
            "test_patient_level_accuracy": tm.get("patient_level_test/accuracy"),
            "test_patient_level_f1_macro": tm.get("patient_level_test/f1_macro"),
            "test_n_patients": tm.get("patient_level_test/n_patients"),
        }

        # ── VAL: fresh inference pass against best_model.pt ─────────────────
        val_result = None
        if not args.skip_val:
            ckpt_path = job_dir / "best_model.pt"
            if not ckpt_path.exists():
                log.warning(f"  No best_model.pt at {ckpt_path} — skipping val.")
            else:
                val_probs = run_val_inference(
                    backbone_type, mode, pretrained, ckpt_path, val_dataset,
                    device, args.batch_size,
                )
                val_labels_arr = np.array([lbl for _, lbl, _ in val_dataset.samples])
                val_result = process_split(
                    "val", val_probs, val_labels_arr, val_patient_ids, val_audio_types,
                    args.threshold, args.skip_audio_type,
                )
                vm = val_result["patient_level_metrics"]
                log.info(f"  VAL   patient-level acc={vm.get('patient_level_val/accuracy', 0):.4f}  "
                         f"f1={vm.get('patient_level_val/f1_macro', 0):.4f}  "
                         f"(n={vm.get('patient_level_val/n_patients')} patients)")
                merge_into_results_summary(job_dir, "val", val_result)
                row["val_patient_level_accuracy"] = vm.get("patient_level_val/accuracy")
                row["val_patient_level_f1_macro"] = vm.get("patient_level_val/f1_macro")
                row["val_n_patients"] = vm.get("patient_level_val/n_patients")

        combined[job_name] = {"test": test_result, "val": val_result}
        rows_for_comparison_table.append(row)

        with open(output_dir / f"{job_name}_patient_level.json", "w") as f:
            json.dump(combined[job_name], f, indent=2, cls=_NumpyEncoder)

    with open(output_dir / "exp1_patient_level_summary.json", "w") as f:
        json.dump(combined, f, indent=2, cls=_NumpyEncoder)

    if rows_for_comparison_table:
        comparison_df = pd.DataFrame(rows_for_comparison_table)
        comparison_df.to_csv(output_dir / "exp1_patient_level_comparison.csv", index=False)
        log.info(f"\n{comparison_df.to_string(index=False)}")

    log.info(f"\n  Results saved -> {output_dir}")


if __name__ == "__main__":
    main()
