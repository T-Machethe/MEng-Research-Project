"""
scripts/run_patient_level_aggregation.py
─────────────────────────────────────────────────────────────────────────────
Aggregates Exp1's segment-level test predictions back to patient-level
predictions and metrics — RETROACTIVELY, from each backbone's already-saved
results_summary.json. No re-inference, no checkpoint loading.

Why this exists
────────────────
SinusitisDataset.__getitem__ returns (waveform, label, audio_col) — no
patient ID. So test/all_probs and test/all_labels in results_summary.json
are segment-level: a patient with 40 segments counts 40 times toward
"test accuracy". That's a different (and usually noisier / more optimistic)
number than "fraction of patients correctly classified", which is the
clinically meaningful one and the one worth reporting alongside the
segment-level number in the thesis.

How the retroactive recovery works
────────────────────────────────────
1. patient_level_split(seed=42) is purely deterministic — re-running it on
   the original CSV via Exp1CRSvsControl(cfg).prepare_data() reproduces the
   EXACT SAME test_df used in the original training run, with no need for a
   saved split manifest.
2. build_experiment_loaders() builds the test DataLoader with shuffle=False
   (confirmed in src/pipeline/dataloader.py). So SinusitisDataset(test_df,
   segment_dir, label_fn, audio_cols).samples — built by iterating the same
   test_df rows over the same segment_dir/audio_cols — is in the SAME ORDER
   the original test loader iterated it in.
3. Therefore: rebuild that dataset now, parse each sample's patient ID from
   its filename (ID{subject}_ses{session}_{col}_seg{idx}.pt), and zip that
   ID list against the saved test/all_probs / test/all_labels arrays from
   results_summary.json. Order-matched, no re-inference required.

This ONLY works if segment_dir and audio_cols are unchanged since the
original run (same segment files must still be on disk, nothing added or
removed). The script asserts len(dataset) == len(saved_probs) and refuses
to proceed on mismatch rather than silently misaligning IDs to predictions.

Aggregation method
────────────────────
Mean predicted P(CRS) across a patient's segments (soft aggregation), not
majority vote on hard per-segment predictions — same convention already
used in lund_mackay_correlation.py and run_cross_cohort_specificity.py.
Avoids ties and preserves confidence information a hard vote would discard.
(eval_utils.py::patient_level_vote() exists but is dead code / hard-vote —
not used here for consistency with the rest of the repo.)

CLI
───
    --csv_path         path to clinical_all_sessions.csv
    --segment_dir      segment dir used for the ORIGINAL Exp1 training run
    --results_dir      root containing exp1_backbone_comparison/ (default:
                       <project_root>/MSc_Sinusitis_results)
    --output_dir       where to write patient-level results (default:
                       <results_dir>/exp1_patient_level)
    --threshold        decision threshold on mean P(CRS) (default: 0.5)
    --test_size / --val_size / --seed : must match the original training
                       run's split config (defaults: 0.20 / 0.10 / 42,
                       matching ExperimentConfig defaults)

Usage
─────
    python scripts/run_patient_level_aggregation.py \\
        --csv_path /content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv \\
        --segment_dir /content/clean_audio \\
        --results_dir /content/drive/MyDrive/MSc_Sinusitis_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path as _Path
from typing import Dict, List, Optional

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.experiments.base import ExperimentConfig
from src.experiments.all_experiments import Exp1CRSvsControl
from src.pipeline.dataset import SinusitisDataset
from src.training.patient_metrics import (
    patient_ids_from_samples, aggregate_to_patient_level as _shared_aggregate,
    compute_patient_level_metrics,
)

# 3 backbones × 2 training modes = 6 combinations, matching the fixed
# `for backbone in ["wav2vec2","wavlm","xlsr"]: for mode in ["scratch","finetune"]`
# loop in run_experiment.py (run_key = f"{backbone}_{mode}"). Not a curated
# subset — Exp1 trains and saves results for all six.
BACKBONE_JOBS = [
    "wav2vec2_scratch",
    "wav2vec2_finetune",
    "wavlm_scratch",
    "wavlm_finetune",
    "xlsr_scratch",
    "xlsr_finetune",
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


def rebuild_exp1_test_df(csv_path: str, project_root: _Path,
                          segment_dir: _Path, test_size: float,
                          val_size: float, seed: int) -> pd.DataFrame:
    """
    Reuses Exp1CRSvsControl.prepare_data() directly rather than
    re-implementing the filter/split logic — same pattern already
    established in run_mfcc_svm_baseline.py. This guarantees the
    Session-1-only filtering (both arms) matches whatever Exp1 actually
    used, even if that logic changes again later.
    """
    cfg = ExperimentConfig(
        project_root=str(project_root),
        segment_dir=str(segment_dir),
        csv_path=str(csv_path),
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )
    exp = Exp1CRSvsControl(cfg)
    _train_df, _val_df, test_df = exp.prepare_data()[:3]
    log.info(
        f"\n  Rebuilt Exp1 test_df: {len(test_df)} rows, "
        f"{test_df['ID'].nunique()} patients "
        f"(seed={seed}, test_size={test_size}, val_size={val_size})"
    )
    return test_df


def build_test_dataset_and_patient_ids(test_df: pd.DataFrame, segment_dir: str):
    """
    Rebuilds the SAME SinusitisDataset the original shuffle=False test
    loader iterated (see module docstring for why the ordering matches),
    and parses patient IDs from each sample's filename, in that same
    order — via the shared helper in src/training/patient_metrics.py, so
    Exp1's retroactive aggregation and Exp5's cross-cohort specificity
    script recover patient identity the same way.
    """
    dataset = SinusitisDataset(
        test_df, segment_dir=segment_dir,
        label_fn=lambda row: int(row["label"]),
    )
    patient_ids = patient_ids_from_samples(dataset.samples)
    return dataset, patient_ids


def load_saved_segment_predictions(results_summary_path: _Path):
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


def aggregate_to_patient_level(probs: np.ndarray, labels: np.ndarray,
                                patient_ids: List[str]) -> pd.DataFrame:
    """
    Thin wrapper around the shared
    src/training/patient_metrics.py::aggregate_to_patient_level(), kept as
    a local name for backward compatibility with the rest of this file.
    Renames its generic p_class0/p_class1 columns to this script's
    p_crs_mean convention (p_class1 == P(CRS), since label 1 = CRS in
    Exp1CRSvsControl) and reports std alongside the shared function's mean,
    since std is useful here for spotting low-confidence patients.
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
        # Shared helper raises on label inconsistency; this script instead
        # warns and proceeds with the first label seen, since it's meant
        # as a diagnostic/reporting tool, not a hard gate — the underlying
        # bug (if any) is still visible in the warning either way.
        log.warning(f"  {e}")
        rec = pd.DataFrame({"ID": patient_ids, "label": labels})
        conflicted = rec.groupby("ID")["label"].nunique()
        conflicted = conflicted[conflicted > 1].index.tolist()
        log.warning(f"  Proceeding using first-seen label per patient for: {conflicted}")
        agg = _shared_aggregate(probs, patient_ids, labels=None)
        first_labels = rec.groupby("ID")["label"].first()
        agg["label"] = agg["ID"].map(first_labels)

    # Segment-level std, for spotting low-confidence / high-variance
    # patients — not produced by the shared helper (mean-only), computed
    # here directly.
    std_df = pd.DataFrame({"ID": patient_ids, "p_crs": probs[:, 1]})
    std_by_id = std_df.groupby("ID")["p_crs"].std()
    agg["p_crs_std"] = agg["ID"].map(std_by_id)

    agg = agg.rename(columns={"p_class1": "p_crs_mean"}).drop(columns=["p_class0"])
    return agg[["ID", "p_crs_mean", "p_crs_std", "n_segments", "label"]]


def patient_level_metrics(patient_df: pd.DataFrame, threshold: float) -> Dict:
    """
    Thin wrapper around the shared compute_patient_level_metrics(), which
    expects p_class{0,1} columns — remapped from this script's
    p_crs_mean/label convention, then back to the "patient_level_test/*"
    key prefix this script's callers/output files already expect.
    """
    shared_input = pd.DataFrame({
        "ID":      patient_df["ID"],
        "p_class0": 1 - patient_df["p_crs_mean"],
        "p_class1": patient_df["p_crs_mean"],
        "label":   patient_df["label"],
    })
    raw = compute_patient_level_metrics(shared_input, num_classes=2, split_name="test")
    # Remap "test/patient_level/*" -> "patient_level_test/*" for backward
    # compatibility with this script's existing output/log key names.
    metrics = {
        k.replace("test/patient_level", "patient_level_test"): v
        for k, v in raw.items()
    }
    metrics["patient_level_test/n_patients"] = int(len(patient_df))
    metrics["patient_level_test/threshold"]  = float(threshold)
    return metrics


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
    args = parser.parse_args()

    project_root = _Path(args.project_root)
    results_dir  = _Path(args.results_dir)
    output_dir   = _Path(args.output_dir) if args.output_dir else \
        results_dir / "exp1_patient_level"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = rebuild_exp1_test_df(
        args.csv_path, project_root, _Path(args.segment_dir),
        args.test_size, args.val_size, args.seed,
    )
    dataset, patient_ids = build_test_dataset_and_patient_ids(
        test_df, args.segment_dir)

    log.info(
        f"\n  Test dataset rebuilt: {len(dataset)} segments, "
        f"{len(set(patient_ids))} unique patient IDs parsed from filenames."
    )

    combined = {}
    rows_for_comparison_table = []

    for job_name in BACKBONE_JOBS:
        summary_path = results_dir / "exp1_backbone_comparison" / job_name / "results_summary.json"
        if not summary_path.exists():
            log.warning(f"  {job_name}: no results_summary.json found at {summary_path} — skipping.")
            continue

        try:
            probs, labels, _raw = load_saved_segment_predictions(summary_path)
        except KeyError as e:
            log.warning(f"  {job_name}: {e}")
            continue

        try:
            patient_df = aggregate_to_patient_level(probs, labels, patient_ids)
        except AssertionError as e:
            log.error(f"  {job_name}: {e}")
            continue

        metrics = patient_level_metrics(patient_df, args.threshold)

        seg_acc = None
        # Segment-level accuracy for side-by-side comparison, straight from
        # the saved segment-level predictions (not recomputed).
        seg_preds = probs.argmax(axis=1)
        seg_acc = float((seg_preds == labels).mean())

        log.info(
            f"\n  {job_name:<18} "
            f"segment-level acc={seg_acc:.4f}  |  "
            f"patient-level acc={metrics.get('patient_level_test/accuracy', 0):.4f}  "
            f"f1={metrics.get('patient_level_test/f1_macro', 0):.4f}  "
            f"(n={metrics.get('patient_level_test/n_patients')} patients)"
        )

        combined[job_name] = {
            "segment_level_accuracy": seg_acc,
            "patient_level_metrics":  metrics,
            "per_patient": patient_df.to_dict(orient="records"),
        }
        rows_for_comparison_table.append({
            "backbone": job_name,
            "segment_level_accuracy": seg_acc,
            "patient_level_accuracy": metrics.get("patient_level_test/accuracy"),
            "patient_level_f1_macro": metrics.get("patient_level_test/f1_macro"),
            "n_patients": metrics.get("patient_level_test/n_patients"),
        })

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
