"""
scripts/compute_nested_cv_control_fpr.py
─────────────────────────────────────────────────────────────────────────────
Computes a genuinely held-out Control false-positive rate per shortlisted
backbone, from the nested-CV outer-fold per-patient predictions, for use as
the Exp5 comparison anchor when evaluating the nested-CV FINAL REFIT model
(scripts/run_experiment.py --output_dir .../nested_cv_final, per the
notebook's "Nested CV Results — winning hyperparameters and the final
refit step" cell).

Why this is needed
──────────────────
run_cross_cohort_specificity.py's load_exp1_baseline() normally reads a
Control-FPR straight from results_summary.json's held-out Exp1 test split.
The nested-CV final refit model is trained on the FULL filtered Exp1
population (see the notebook) — it has no held-out Exp1 split of its own,
so that approach doesn't apply to it.

What this script does instead
──────────────────────────────
For each shortlisted backbone, its {job}_nested_cv.json already contains,
per outer fold, "outer_test_per_patient" (MLP head) and
"svm_outer_test_per_patient" (SVM head) — each a two-stage-aggregated,
per-patient record with "ID", "p_class0", "p_class1", "label", plus
"n_segments"/"n_recordings" (see src/training/patient_metrics.py). Every
Exp1 patient appears in EXACTLY ONE outer fold's test set across the 5
folds (the outer folds partition the full population), so pooling
outer_test_per_patient across all 5 folds gives one row per Exp1 patient —
a genuinely held-out prediction for every patient, made by a model trained
under the SAME architecture/hyperparameters as the final refit (just not
literally the same weights, since the final refit is trained on all of
them at once). This is the best available substitute for a held-out
Control-FPR when no such split exists for the final model itself.

label == 0 is Control (per Exp1CRSvsControl's labelling convention — see
src/experiments/all_experiments.py). Control-FPR = fraction of pooled
Control patients with p_class1 >= --threshold.

Output
──────
A single JSON keyed by job_name, e.g.:

    {
      "xlsr_finetune": {
        "mlp": {"control_false_positive_rate": 0.1897, "control_n_patients": 29},
        "svm": {"control_false_positive_rate": 0.2069, "control_n_patients": 29}
      },
      ...
    }

Pass this file to run_cross_cohort_specificity.py via --control_fpr_json.

CLI
───
    --nested_cv_dir   directory containing {job}_nested_cv.json (default:
                       <project_root>/MSc_Sinusitis_results.../nested_cv)
    --jobs             comma-separated job names (default: the 3 shortlisted
                       configs — xlsr_finetune,xlsr_scratch,wav2vec2_finetune)
    --threshold        decision threshold on P(class1) (default: 0.5)
    --output_path      where to write the JSON (default:
                       <nested_cv_dir>/nested_cv_control_fpr.json)

Usage
─────
    python scripts/compute_nested_cv_control_fpr.py \\
        --nested_cv_dir /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback/nested_cv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_JOBS = ["xlsr_finetune", "xlsr_scratch", "wav2vec2_finetune"]


def _pool_control_fpr(fold_records_key: str, nested_cv_json: dict, threshold: float) -> dict | None:
    """
    Pools per-patient records for `fold_records_key` ("outer_test_per_patient"
    or "svm_outer_test_per_patient") across every fold in nested_cv_json,
    filters to Control (label == 0), and returns
    {"control_false_positive_rate": ..., "control_n_patients": ...}.

    Returns None if this head's per-patient records aren't present for any
    fold (e.g. no SVM head for that run).
    """
    all_rows = []
    for fold in nested_cv_json.get("folds", []):
        rows = fold.get(fold_records_key)
        if not rows:
            return None
        all_rows.extend(rows)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    if "label" not in df.columns or "p_class1" not in df.columns:
        log.warning(f"  Pooled records missing 'label'/'p_class1' columns "
                    f"for {fold_records_key} — skipping.")
        return None

    n_before = df["ID"].nunique()
    if len(df) != n_before:
        log.warning(f"  {len(df)} pooled rows but only {n_before} unique "
                    f"patient IDs for {fold_records_key} — a patient appears "
                    f"in more than one outer fold's test set. This should "
                    f"not happen (outer folds partition the population); "
                    f"check the nested-CV run before trusting this number.")

    control = df[df["label"] == 0]
    if len(control) == 0:
        return None

    flagged = (control["p_class1"] >= threshold).mean()
    return {
        "control_false_positive_rate": float(flagged),
        "control_n_patients": int(len(control)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nested_cv_dir", type=str, required=True)
    parser.add_argument("--jobs", type=str, default=",".join(DEFAULT_JOBS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    nested_cv_dir = Path(args.nested_cv_dir)
    jobs = [j.strip() for j in args.jobs.split(",")]
    output_path = Path(args.output_path) if args.output_path else \
        nested_cv_dir / "nested_cv_control_fpr.json"

    result = {}
    for job_name in jobs:
        job_path = nested_cv_dir / f"{job_name}_nested_cv.json"
        if not job_path.exists():
            log.warning(f"✗ {job_path} not found — skipping {job_name}.")
            continue

        with open(job_path) as f:
            nested_cv_json = json.load(f)

        mlp = _pool_control_fpr("outer_test_per_patient", nested_cv_json, args.threshold)
        svm = _pool_control_fpr("svm_outer_test_per_patient", nested_cv_json, args.threshold)

        if mlp is None:
            log.warning(f"  No usable MLP per-patient records for {job_name} — "
                        f"check {job_path} contains 'outer_test_per_patient' "
                        f"per fold (older/stale fold files may not).")
            continue

        entry = {"mlp": mlp}
        if svm is not None:
            entry["svm"] = svm
        result[job_name] = entry

        log.info(f"  {job_name:<20} MLP Control-FPR = {mlp['control_false_positive_rate']:.4f} "
                 f"(n={mlp['control_n_patients']})"
                 + (f"   SVM Control-FPR = {svm['control_false_positive_rate']:.4f} "
                    f"(n={svm['control_n_patients']})" if svm else "   (no SVM head)"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\n  Wrote {output_path}")


if __name__ == "__main__":
    main()
