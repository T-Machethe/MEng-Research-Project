"""
scripts/check_pipeline_status.py
─────────────────────────────────────────────────────────────────────────────
Read-only status check across the whole examiner-feedback pipeline: Exp1,
Exp5, ablation (per backbone/mode), and nested CV. Touches nothing —
doesn't train, doesn't delete, doesn't move files. Just tells you what
already exists on Drive and what's still outstanding, so you don't have
to eyeball five different directory trees by hand every time you pick
this back up.

Usage
──────
    python scripts/check_pipeline_status.py \\
        --results_dir /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback

Optional: --backbones to only report ablation/nested-CV status for a
specific shortlist (defaults to checking all six).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BACKBONE_JOBS = [
    "wav2vec2_scratch", "wav2vec2_finetune",
    "wavlm_scratch",     "wavlm_finetune",
    "xlsr_scratch",      "xlsr_finetune",
]


def check_exp1(results_dir: _Path, backbones):
    print("═" * 78)
    print("  EXP1 — CRS vs Control (Session 1 both arms)")
    print("═" * 78)
    exp_dir = results_dir / "exp1_backbone_comparison"
    if not exp_dir.exists():
        print("  ✗ Not started — no exp1_backbone_comparison/ directory found.\n")
        return

    HDR = f"  {'BACKBONE':<20} {'TRAINED':<9} {'PATIENT-LVL':<12} {'SVM':<9} {'SVM PT-LVL':<11}"
    print(HDR)
    print("  " + "─" * (len(HDR) - 2))
    for job in backbones:
        job_dir = exp_dir / job
        summary_path = job_dir / "results_summary.json"
        trained = "✓" if summary_path.exists() else "✗"
        patient_lvl, svm, svm_patient_lvl = "✗", "✗", "✗"
        if summary_path.exists():
            with open(summary_path) as f:
                res = json.load(f)
            patient_lvl = "✓" if "test/patient_level/accuracy" in res else "✗"
            svm = "✓" if "svm" in res else "✗"
            svm_patient_lvl = "✓" if res.get("svm", {}).get("test/patient_level/accuracy") is not None else "✗"
        print(f"  {job:<20} {trained:<9} {patient_lvl:<12} {svm:<9} {svm_patient_lvl:<11}")
    print()


def check_exp5(results_dir: _Path):
    print("═" * 78)
    print("  EXP5 — Cross-cohort specificity")
    print("═" * 78)
    summary_path = results_dir / "exp5_cross_cohort_specificity" / "cross_cohort_specificity_summary.json"
    if not summary_path.exists():
        print("  ✗ Not run — no cross_cohort_specificity_summary.json found.\n")
        return

    with open(summary_path) as f:
        summary = json.load(f)
    print(f"  ✓ Evaluated {len(summary)} backbone(s): {list(summary.keys())}")
    for job, res in summary.items():
        has_svm = any("svm_metrics" in res.get(g, {}) for g in ["Sept", "Tonsill"])
        print(f"    {job:<20} SVM head evaluated: {'✓' if has_svm else '✗'}")
    print()


def check_ablation(results_dir: _Path, backbones):
    print("═" * 78)
    print("  ABLATION")
    print("═" * 78)
    ablation_dir = results_dir / "ablation"
    if not ablation_dir.exists():
        print("  ✗ Not started — no ablation/ directory found.\n")
        return

    # Old (pre-backbone-parameterization) flat layout: ablation/<factor>/...
    old_layout_factors = [d.name for d in ablation_dir.iterdir()
                           if d.is_dir() and d.name in ("freeze", "loss", "decay")]
    if old_layout_factors:
        print(f"  ⚠ OLD FLAT LAYOUT detected: {ablation_dir}/{{{','.join(old_layout_factors)}}}/")
        print("    This predates backbone-namespaced ablation output and doesn't indicate")
        print("    which backbone it was run against by its path alone. If this was run")
        print("    against the old (pre-Session-1-fix, XLS-R-only) Exp1 code, it no longer")
        print("    matches your current pipeline — see the cleanup command below.")
        print()

    # New namespaced layout: ablation/<backbone>_<mode>/<factor>/...
    for job in backbones:
        job_dir = ablation_dir / job
        agg_path = job_dir / "ablation_results.json"
        if not agg_path.exists():
            continue
        with open(agg_path) as f:
            all_results = json.load(f)
        factors_done = list(all_results.keys())
        print(f"  {job:<20} factors complete: {factors_done}")
    if not any((ablation_dir / job / "ablation_results.json").exists() for job in backbones):
        print("  (no backbone-namespaced ablation results yet)")
    print()


def check_nested_cv(results_dir: _Path, backbones):
    print("═" * 78)
    print("  NESTED CV")
    print("═" * 78)
    nested_dir = results_dir / "nested_cv"
    if not nested_dir.exists():
        print("  ✗ Not started — no nested_cv/ directory found.\n")
        return

    summary_path = nested_dir / "nested_cv_summary.json"
    if not summary_path.exists():
        print("  ✗ No nested_cv_summary.json yet (individual backbone runs may be in progress).\n")
        return

    with open(summary_path) as f:
        summary = json.load(f)
    for job, s in summary.items():
        print(f"  {job:<20} patient-F1 = {s['neural_patient_f1_mean']:.4f} ± "
              f"{s['neural_patient_f1_std']:.4f}  ({s['outer_folds']} outer folds)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--backbones", type=str, default=None,
                         help="Comma-separated subset. Default: all six.")
    args = parser.parse_args()

    results_dir = _Path(args.results_dir)
    backbones = args.backbones.split(",") if args.backbones else BACKBONE_JOBS

    if not results_dir.exists():
        print(f"✗ {results_dir} does not exist — nothing has been run yet.")
        return

    print(f"\nPipeline status: {results_dir}\n")
    check_exp1(results_dir, backbones)
    check_exp5(results_dir)
    check_ablation(results_dir, backbones)
    check_nested_cv(results_dir, backbones)

    print("═" * 78)
    print("  SUGGESTED NEXT STEP")
    print("═" * 78)
    exp1_done = (results_dir / "exp1_backbone_comparison" / "wav2vec2_finetune" / "results_summary.json").exists()
    exp5_done = (results_dir / "exp5_cross_cohort_specificity" / "cross_cohort_specificity_summary.json").exists()
    old_ablation = (results_dir / "ablation" / "freeze").exists()

    if not exp1_done:
        print("  → Run Exp1 training first.")
    elif not exp5_done:
        print("  → Exp1 looks done — run Exp5 evaluate next.")
    elif old_ablation:
        print("  → Exp1 + Exp5 done, but an OLD-LAYOUT ablation run exists.")
        print("    Delete it before starting the backbone-parameterized version:")
        print(f"      rm -rf {results_dir / 'ablation'}")
        print("    Then re-run with the updated script:")
        print("      python scripts/run_ablation.py --run --backbone <shortlisted> "
              "--mode finetune --factor all ...")
    else:
        print("  → Exp1 + Exp5 done. Pick your shortlist from the results above, "
              "then run ablation + nested CV scoped to those backbones.")


if __name__ == "__main__":
    main()
