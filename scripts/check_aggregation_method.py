"""
scripts/check_aggregation_method.py
─────────────────────────────────────────────────────────────────────────────
Scans every patient-level result already on disk — Exp1, Exp5, and nested
CV alike — and reports whether each one was computed with the corrected
two-stage (recording-then-patient) aggregation or the old, flat one-stage
mean. Read-only; changes nothing.

How it decides, per result
────────────────────────────
1. Preferred: check for an explicit "<split>/patient_level/aggregation_method"
   key (added going forward by src/training/patient_metrics.py::
   compute_patient_level_metrics()) — unambiguous, no inference needed.
2. Fallback, for files that predate that marker: check whether ANY
   per-patient record in that split has an "n_recordings" key — only the
   two-stage aggregation path ever adds this field (confirmed directly
   from aggregate_to_patient_level()'s implementation — the old flat path
   never wrote it), so its presence is a reliable retroactive signal even
   without the explicit marker.
3. If neither is present at all (e.g. patient-level metrics were never
   computed for this split), reported as "not_computed", not "old_flat" —
   an important distinction: "not computed" means run the relevant script;
   "old_flat" means rerun it to get the corrected numbers.

Nested CV inner-run caches (<job>_fold<i>_inner<j>_combo<k>.json) are
skipped entirely — they only ever store a single scalar score (see
run_nested_cv.py's inner-run checkpointing), nothing to classify. The
fold-level fingerprint's "_aggregation_method" field is reported alongside
each nested-CV fold anyway, since that's what actually determines whether
resuming will auto-recompute it (see run_nested_cv.py::config_fingerprint).

Usage
──────
    python scripts/check_aggregation_method.py \\
        --results_dir /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path as _Path
from typing import Dict


def classify(obj: Dict, split: str) -> str:
    """
    Returns 'two_stage' | 'old_flat' | 'not_computed' for one split's
    patient-level result within a results dict (Exp1's per-backbone
    results_summary.json shape, or its 'svm' sub-dict).
    """
    method_key = f"{split}/patient_level/aggregation_method"
    if method_key in obj:
        return "two_stage" if obj[method_key] == "two_stage_recording_then_patient" else "old_flat"

    per_patient = obj.get(f"{split}/patient_level/per_patient")
    if not per_patient:
        return "not_computed"
    return "two_stage" if any("n_recordings" in row for row in per_patient) else "old_flat"


def check_exp1(results_dir: _Path) -> None:
    print("\n" + "=" * 70)
    print("  EXP1 — exp1_backbone_comparison/")
    print("=" * 70)
    exp1_dir = results_dir / "exp1_backbone_comparison"
    if not exp1_dir.exists():
        print("  (not found)")
        return

    found_any = False
    for job_dir in sorted(exp1_dir.iterdir()):
        summary_path = job_dir / "results_summary.json"
        if not summary_path.exists():
            continue
        found_any = True
        with open(summary_path) as f:
            res = json.load(f)
        for split in ("test", "val"):
            status = classify(res, split)
            svm_res = res.get("svm")
            svm_str = f"   SVM={classify(svm_res, split)}" if svm_res else ""
            print(f"  {job_dir.name:<20} {split:<5} {status:<14}{svm_str}")

    if not found_any:
        print("  (no backbones found)")


def check_exp5(results_dir: _Path) -> None:
    print("\n" + "=" * 70)
    print("  EXP5 — exp5_cross_cohort_specificity/")
    print("=" * 70)
    exp5_dir = results_dir / "exp5_cross_cohort_specificity"
    if not exp5_dir.exists():
        print("  (not found)")
        return

    spec_files = sorted(exp5_dir.glob("*_specificity.json"))
    if not spec_files:
        print("  (no per-backbone specificity files found)")
        return

    for spec_path in spec_files:
        with open(spec_path) as f:
            res = json.load(f)
        for grp in ("Sept", "Tonsill"):
            grp_res = res.get(grp)
            if not grp_res:
                continue
            per_patient = grp_res.get("per_patient")
            if not per_patient:
                status = "not_computed"
            else:
                status = "two_stage" if any("n_recordings" in r for r in per_patient) else "old_flat"
            svm_per_patient = grp_res.get("svm_per_patient")
            svm_str = ""
            if svm_per_patient is not None:
                svm_status = ("two_stage" if any("n_recordings" in r for r in svm_per_patient)
                               else "old_flat") if svm_per_patient else "not_computed"
                svm_str = f"   SVM={svm_status}"
            print(f"  {spec_path.stem:<25} {grp:<8} {status:<14}{svm_str}")


def check_nested_cv(results_dir: _Path) -> None:
    print("\n" + "=" * 70)
    print("  NESTED CV — nested_cv/")
    print("=" * 70)
    nested_dir = results_dir / "nested_cv"
    if not nested_dir.exists():
        print("  (not found)")
        return

    fold_files = sorted(p for p in nested_dir.glob("*_fold*.json") if "_inner" not in p.stem)
    if not fold_files:
        print("  (no completed outer-fold results found)")
        return

    for fold_path in fold_files:
        with open(fold_path) as f:
            res = json.load(f)
        status = classify(res, "outer_test")
        fp = res.get("_config_fingerprint", {})
        fp_marker = fp.get("_aggregation_method", "MISSING (pre-fix cache — will "
                                                   "auto-recompute on next resume)")
        print(f"  {fold_path.stem:<35} {status:<14} fingerprint={fp_marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    results_dir = _Path(args.results_dir)

    if not results_dir.exists():
        print(f"✗ {results_dir} does not exist.")
        return

    check_exp1(results_dir)
    check_exp5(results_dir)
    check_nested_cv(results_dir)

    print("\n" + "=" * 70)
    print("  'old_flat' entries need re-running to get corrected two-stage")
    print("  patient-level metrics: Exp1/Exp5 need --overwrite passed explicitly")
    print("  (neither script versions its cache); nested CV recomputes")
    print("  automatically on the next resume, via the fingerprint mismatch —")
    print("  no --overwrite needed there.")
    print("=" * 70)


if __name__ == "__main__":
    main()
