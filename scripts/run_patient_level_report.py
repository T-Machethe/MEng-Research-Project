"""
scripts/run_patient_level_report.py
─────────────────────────────────────────────────────────────────────────────
Generates the patient-level equivalent of Exp1's segment-level backbone-
comparison PDF report — same charts, same tables, same structure — by
reusing reporter.py's EXISTING, already-tested plotting code unchanged.

How this works: no new plotting code
──────────────────────────────────────
src/training/reporter.py::ExperimentReporter._plot_backbone_comparison()
reads segment-level keys (test/accuracy, test/confusion_matrix,
test/per_audio_type, svm.test/f1_macro, etc.) straight from each
backbone's results dict. Rather than duplicate that ~500-line function
with a patient-level copy, this script builds an ADAPTER dict per
backbone that has the SAME key names and SHAPES as the segment-level
dict, but populated from that backbone's ALREADY-COMPUTED
test/patient_level/* and val/patient_level/* values (see
run_patient_level_aggregation.py and base.py's _add_patient_level_metrics
— both must have already run against your results for this script to
have anything to read). Then calls _plot_backbone_comparison() completely
unmodified against the adapted dicts.

This means: if reporter.py's segment-level report is ever fixed or
extended, the patient-level report benefits automatically — there's one
plotting implementation, not two to keep in sync — and any bug already
fixed there (e.g. the stale-fallback / undefined-color-constant bugs
fixed earlier in this project) is fixed here too, for free.

Requires patient-level data to already exist
──────────────────────────────────────────────
Run scripts/run_patient_level_aggregation.py first (retroactive, no
retraining — see that script's docstring). This script only reads
results_summary.json files; it computes no aggregation itself.

What does NOT carry over from segment-level
───────────────────────────────────────────────
Loss curves (page/figure showing train/val loss per epoch) — loss is a
segment-level training artifact with no patient-level equivalent, so
that page is skipped entirely rather than showing something meaningless.
Everything else in the comparison PDF (headline table, bar charts,
MLP-vs-SVM, per-class F1, per-split summary, audio-type heatmap,
confusion matrices) has a direct patient-level counterpart and is
included.

Usage
──────
    python scripts/run_patient_level_report.py \\
        --results_dir /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path
from typing import Dict, List

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BACKBONE_JOBS = [
    "wav2vec2_scratch", "wav2vec2_finetune",
    "wavlm_scratch",     "wavlm_finetune",
    "xlsr_scratch",      "xlsr_finetune",
]


def flatten_audio_type(by_audio_type_metrics: List[Dict]) -> Dict:
    """
    Converts the list-of-dicts shape produced by
    src/training/patient_metrics.py::compute_patient_audiotype_metrics()
    ([{"audio_type": "a", "level": "column", "accuracy":..., "f1_macro":
    ..., "roc_auc":..., "n_patients":...}, ...]) into the dict-keyed-by-
    audio_type shape src/training/metrics.py::evaluate_by_audio_type()
    produces at segment level ({"a": {"f1_macro":..., "accuracy":...,
    ...}}), since that's the shape reporter.py's per-audio-type heatmap
    code expects. Includes every tier (individual columns, groups,
    overall) in one flat dict — same structure the segment-level
    version's table already has (compare against the 17-row heatmap in
    the segment-level PDF: 12 individual columns + 4 groups + overall).
    """
    out = {}
    for row in by_audio_type_metrics:
        out[row["audio_type"]] = {
            "accuracy":   row.get("accuracy"),
            "f1_macro":   row.get("f1_macro"),
            "roc_auc":    row.get("roc_auc"),
            "n_segments": row.get("n_patients"),  # relabeled — same slot the heatmap reads
        }
    return out


def adapt_to_segment_shape(results: Dict) -> Dict:
    """
    Builds a dict with segment-level key NAMES but PATIENT-LEVEL values,
    so reporter.py's existing _plot_backbone_comparison() can be called
    completely unmodified. See module docstring.
    """
    adapted = {
        # Only used for the "Epochs" column (len(training_history)) —
        # not itself a patient-level quantity, just training metadata
        # passed through unchanged.
        "training_history": results.get("training_history", []),
    }
    for split in ("val", "test"):
        prefix = f"{split}/patient_level"
        adapted[f"{split}/accuracy"]         = results.get(f"{prefix}/accuracy")
        adapted[f"{split}/f1_macro"]         = results.get(f"{prefix}/f1_macro")
        adapted[f"{split}/roc_auc"]          = results.get(f"{prefix}/roc_auc")
        adapted[f"{split}/f1_per_class"]     = results.get(f"{prefix}/f1_per_class")
        adapted[f"{split}/confusion_matrix"] = results.get(f"{prefix}/confusion_matrix")
        by_at = results.get(f"{prefix}/by_audio_type_metrics")
        if by_at:
            adapted[f"{split}/per_audio_type"] = flatten_audio_type(by_at)

    svm_results = results.get("svm")
    if svm_results:
        svm_adapted = {}
        for split in ("val", "test"):
            prefix = f"{split}/patient_level"
            svm_adapted[f"{split}/accuracy"]         = svm_results.get(f"{prefix}/accuracy")
            svm_adapted[f"{split}/f1_macro"]         = svm_results.get(f"{prefix}/f1_macro")
            svm_adapted[f"{split}/roc_auc"]          = svm_results.get(f"{prefix}/roc_auc")
            svm_adapted[f"{split}/f1_per_class"]     = svm_results.get(f"{prefix}/f1_per_class")
            svm_adapted[f"{split}/confusion_matrix"] = svm_results.get(f"{prefix}/confusion_matrix")
            by_at = svm_results.get(f"{prefix}/by_audio_type_metrics")
            if by_at:
                svm_adapted[f"{split}/per_audio_type"] = flatten_audio_type(by_at)
        adapted["svm"] = svm_adapted

    return adapted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = _Path(args.results_dir)
    output_dir = _Path(args.output_dir) if args.output_dir else \
        results_dir / "exp1_patient_level_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    exp1_dir = results_dir / "exp1_backbone_comparison"
    all_results_adapted = {}
    missing_patient_level = []

    for job_name in BACKBONE_JOBS:
        summary_path = exp1_dir / job_name / "results_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            results = json.load(f)

        if "test/patient_level/accuracy" not in results:
            missing_patient_level.append(job_name)
            continue

        all_results_adapted[job_name] = adapt_to_segment_shape(results)

    if missing_patient_level:
        print(f"⚠ No patient-level data for: {missing_patient_level} — "
              f"run scripts/run_patient_level_aggregation.py first. "
              f"Continuing with whatever backbones DO have it.")

    if not all_results_adapted:
        print("✗ No backbones have patient-level data yet. Nothing to report. "
              "Run scripts/run_patient_level_aggregation.py first.")
        return

    print(f"Generating patient-level report for: {list(all_results_adapted.keys())}")

    from src.training.reporter import ExperimentReporter
    reporter = ExperimentReporter(
        results={}, output_dir=str(output_dir),
        experiment_name="Exp1 CRS vs Control (Patient-Level)",
        num_classes=2,
    )
    reporter._plot_backbone_comparison(all_results_adapted, exp_key="1 (Patient-Level)")

    print(f"\nPatient-level report saved -> {output_dir / 'report.pdf'}")


if __name__ == "__main__":
    main()
