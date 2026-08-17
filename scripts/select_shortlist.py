"""
scripts/select_shortlist.py
─────────────────────────────────────────────────────────────────────────────
Selects which backbones proceed to ablation + nested CV, purely from
whatever Exp1 and Exp5 results already exist on disk. Nothing here is
hardcoded to a specific backbone name — run it again after any new
training and it re-decides from the latest numbers.

Criteria, in order
────────────────────
1. DISCARD anything degenerate, regardless of headline accuracy. A model
   is flagged degenerate if EITHER:
     - its predicted P(CRS) is near-constant across patients on its own
       Exp1 test set (std < --std_threshold, default 0.05) — the
       xlsr_scratch pattern from this project's first run: ~0.55 for
       every patient regardless of identity, not real discrimination.
     - its Exp1 patient-level accuracy or macro-F1 sits at/below
       near-chance (--acc_threshold / --f1_threshold, default 0.55/0.45).
   Degenerate candidates are EXCLUDED from ranking entirely, never
   selected regardless of any other score.

2. Among the rest, score = Exp1 patient-level macro-F1 MINUS a penalty
   for Exp5 specificity failure. The penalty only counts EXCESS
   flagging over that same model's own held-out Control false-positive
   rate (max(0, flagged_rate - own_control_fpr), averaged over Sept and
   Tonsill) — being BELOW its own baseline isn't penalized, since that's
   not evidence of anything wrong. --gap_weight (default 1.0) controls
   how much a specificity failure costs relative to raw F1; the default
   means a 10-point excess flagging rate costs as much as a 0.10 drop in
   patient-level F1 — a deliberately punishing default, since a model
   that can't tell CRS from "recently had airway surgery" is exactly
   the failure mode this whole examiner-feedback branch exists to catch.

3. Family-pair preference: the single best-scoring FAMILY (wav2vec2 /
   wavlm / xlsr, by whichever of its variants scores highest) has BOTH
   its scratch and finetune variants included, if both are non-
   degenerate — this preserves a nested-CV-confirmable scratch-vs-
   finetune (pretraining) comparison for at least one architecture,
   which is this project's primary research question. Remaining
   --top_n slots are filled by whatever's left with the next-highest
   scores, from any family.

Nothing above references a specific backbone name — every step operates
on whatever job_names are actually present in the results directories.

Usage
──────
    python scripts/select_shortlist.py \\
        --results_dir /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback \\
        --top_n 3

Writes shortlist.json to --results_dir, consumed by the ablation and
nested-CV notebook cells so they never need a hardcoded backbone list
either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Maps a job_name to its architecture family. Derived mechanically from
# the job_name string (everything before the last "_"), not a fixed list
# of expected names — works for any "<family>_<mode>" naming, including
# ones not yet seen (e.g. a 7th backbone added later).
def family_of(job_name: str) -> str:
    return job_name.rsplit("_", 1)[0]


def load_exp1_results(results_dir: _Path) -> Dict[str, Dict]:
    """Loads every results_summary.json actually present under
    exp1_backbone_comparison/ — however many backbones that turns out to
    be, not a fixed set of six."""
    exp1_dir = results_dir / "exp1_backbone_comparison"
    out = {}
    if not exp1_dir.exists():
        return out
    for job_dir in exp1_dir.iterdir():
        summary_path = job_dir / "results_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                out[job_dir.name] = json.load(f)
    return out


def load_exp5_results(results_dir: _Path) -> Dict[str, Dict]:
    summary_path = results_dir / "exp5_cross_cohort_specificity" / "cross_cohort_specificity_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        return json.load(f)


def check_degeneracy(exp1_result: Dict, std_threshold: float,
                      acc_threshold: float, f1_threshold: float) -> Tuple[bool, List[str]]:
    reasons = []
    per_patient = exp1_result.get("test/patient_level/per_patient", [])
    if not per_patient:
        return True, ["no patient-level data in results_summary.json — "
                       "re-run with the patient-level metrics patches applied"]

    probs = [r.get("p_class1") for r in per_patient if r.get("p_class1") is not None]
    if probs:
        std = float(np.std(probs))
        if std < std_threshold:
            reasons.append(f"near-constant P(CRS) across patients (std={std:.4f} < {std_threshold})")

    acc = exp1_result.get("test/patient_level/accuracy")
    if acc is not None and acc <= acc_threshold:
        reasons.append(f"near/below-chance patient-level accuracy ({acc:.3f} <= {acc_threshold})")

    f1 = exp1_result.get("test/patient_level/f1_macro")
    if f1 is not None and f1 <= f1_threshold:
        reasons.append(f"very low patient-level macro-F1 ({f1:.3f} <= {f1_threshold})")

    return (len(reasons) > 0), reasons


def compute_specificity_gap(exp5_job_result: Dict, head: str = "mlp") -> Tuple[Optional[float], str]:
    """
    Mean, over Sept and Tonsill, of max(0, flagged_rate - own_control_fpr)
    — how much this model over-flags these cohorts relative to its OWN
    baseline false-positive rate on Control. Returns (None, reason) if
    Exp5 hasn't been run for this backbone/head yet.
    """
    baseline = exp5_job_result.get("exp1_baseline", {})
    if head == "svm":
        baseline = baseline.get("svm", {})
    ctrl_fpr = baseline.get("test/patient_level/control_false_positive_rate")
    if ctrl_fpr is None:
        return None, f"no Exp1 Control baseline available for head={head}"

    gaps = []
    for cohort in ["Sept", "Tonsill"]:
        cohort_result = exp5_job_result.get(cohort, {})
        metrics_key = "svm_metrics" if head == "svm" else "metrics"
        metrics = cohort_result.get(metrics_key, {})
        flagged = metrics.get("specificity/flagged_crs_rate")
        if flagged is not None:
            gaps.append(max(0.0, flagged - ctrl_fpr))
    if not gaps:
        return None, f"no Exp5 cohort data available for head={head}"
    return float(np.mean(gaps)), ""


def score_candidates(exp1_results: Dict, exp5_results: Dict, std_threshold: float,
                      acc_threshold: float, f1_threshold: float, gap_weight: float) -> Dict:
    candidates = {}
    for job_name, exp1_res in exp1_results.items():
        degenerate, reasons = check_degeneracy(exp1_res, std_threshold, acc_threshold, f1_threshold)
        f1 = exp1_res.get("test/patient_level/f1_macro")
        acc = exp1_res.get("test/patient_level/accuracy")

        exp5_res = exp5_results.get(job_name, {})
        gap, gap_reason = compute_specificity_gap(exp5_res, head="mlp") if exp5_res else (None, "Exp5 not run yet")

        svm_f1 = exp1_res.get("svm", {}).get("test/patient_level/f1_macro")
        svm_gap, _ = compute_specificity_gap(exp5_res, head="svm") if exp5_res else (None, None)

        if degenerate:
            score = float("-inf")
        elif f1 is None:
            degenerate, reasons = True, ["missing test/patient_level/f1_macro"]
            score = float("-inf")
        elif gap is None:
            # Exp1 exists but Exp5 hasn't run for this backbone yet — can't
            # score it against the specificity criterion. Not "degenerate",
            # just not ready to rank; excluded from selection with a clear
            # reason rather than silently scored on Exp1 alone (which would
            # be exactly the mistake this whole criterion exists to avoid).
            score = None
        else:
            score = f1 - gap_weight * gap

        candidates[job_name] = {
            "family": family_of(job_name),
            "degenerate": degenerate,
            "degenerate_reasons": reasons,
            "exp1_patient_f1": f1,
            "exp1_patient_acc": acc,
            "exp5_specificity_gap": gap,
            "exp5_gap_reason": gap_reason,
            "svm_patient_f1": svm_f1,
            "svm_specificity_gap": svm_gap,
            "score": score,
        }
    return candidates


def select_shortlist(candidates: Dict, top_n: int, prefer_family_pair: bool) -> Tuple[List[str], List[str]]:
    reasoning = []
    rankable = {j: c for j, c in candidates.items()
                if not c["degenerate"] and c["score"] is not None}

    if not rankable:
        reasoning.append("No rankable candidates — every backbone is either degenerate, "
                          "missing Exp1 results, or missing Exp5 results. Nothing to shortlist yet.")
        return [], reasoning

    selected: List[str] = []

    if prefer_family_pair:
        family_best_score: Dict[str, float] = {}
        for job, c in rankable.items():
            fam = c["family"]
            if fam not in family_best_score or c["score"] > family_best_score[fam]:
                family_best_score[fam] = c["score"]
        top_family = max(family_best_score, key=family_best_score.get)

        family_jobs = sorted(
            [j for j, c in rankable.items() if c["family"] == top_family],
            key=lambda j: -rankable[j]["score"],
        )
        if len(family_jobs) >= 2:
            selected.extend(family_jobs)
            reasoning.append(
                f"Top-scoring family: '{top_family}' — included BOTH "
                f"{family_jobs} (both non-degenerate) to preserve a "
                f"nested-CV-confirmable pretraining (scratch-vs-finetune) comparison."
            )
        else:
            selected.append(family_jobs[0])
            reasoning.append(
                f"Top-scoring family: '{top_family}' — only one non-degenerate "
                f"variant available ({family_jobs[0]}); no pretraining-contrast "
                f"pair possible for this family."
            )

    remaining_slots = max(0, top_n - len(selected))
    ranked_rest = sorted(
        [j for j in rankable if j not in selected],
        key=lambda j: -rankable[j]["score"],
    )
    fill = ranked_rest[:remaining_slots]
    if fill:
        reasoning.append(f"Filled {len(fill)} remaining slot(s) with next-highest-scoring "
                          f"candidate(s): {fill}")
    selected.extend(fill)

    return selected, reasoning


def print_report(candidates: Dict, selected: List[str], reasoning: List[str], gap_weight: float):
    print("═" * 100)
    print("  SHORTLIST SELECTION — scored from current Exp1 + Exp5 results")
    print("═" * 100)
    HDR = (f"  {'BACKBONE':<20} {'STATUS':<12} {'Exp1 F1':>8} {'Exp5 gap':>9} "
           f"{'score':>8} {'SVM F1':>8} {'SVM gap':>8}")
    print(HDR)
    print("  " + "─" * (len(HDR) - 2))

    def fmt(v, d=3):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) and v not in (float("-inf"),) else "—"

    for job, c in sorted(candidates.items(), key=lambda kv: -(kv[1]["score"] or float("-inf"))):
        status = "SELECTED" if job in selected else ("DEGENERATE" if c["degenerate"] else
                  ("NOT SCORED" if c["score"] is None else "candidate"))
        print(f"  {job:<20} {status:<12} {fmt(c['exp1_patient_f1']):>8} "
              f"{fmt(c['exp5_specificity_gap']):>9} {fmt(c['score']):>8} "
              f"{fmt(c['svm_patient_f1']):>8} {fmt(c['svm_specificity_gap']):>8}")
        if c["degenerate"] and c["degenerate_reasons"]:
            for r in c["degenerate_reasons"]:
                print(f"      ⚠ {r}")
        if c["score"] is None and not c["degenerate"]:
            print(f"      ⚠ {c['exp5_gap_reason']}")

    print(f"\n  Scoring: score = Exp1_patient_F1 - {gap_weight} × Exp5_specificity_gap")
    print("\n  Selection reasoning:")
    for r in reasoning:
        print(f"    • {r}")

    print(f"\n  SHORTLIST: {selected}")
    if selected:
        print(f"  --backbones {','.join(selected)}")
    print("═" * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--gap_weight", type=float, default=1.0)
    parser.add_argument("--std_threshold", type=float, default=0.05)
    parser.add_argument("--acc_threshold", type=float, default=0.55)
    parser.add_argument("--f1_threshold", type=float, default=0.45)
    parser.add_argument("--no_family_pair", action="store_true",
                         help="Disable the family-pair preference; pure top-N by score.")
    args = parser.parse_args()

    results_dir = _Path(args.results_dir)
    exp1_results = load_exp1_results(results_dir)
    exp5_results = load_exp5_results(results_dir)

    if not exp1_results:
        print(f"✗ No Exp1 results found under {results_dir / 'exp1_backbone_comparison'}")
        sys.exit(1)

    candidates = score_candidates(exp1_results, exp5_results, args.std_threshold,
                                   args.acc_threshold, args.f1_threshold, args.gap_weight)
    selected, reasoning = select_shortlist(candidates, args.top_n, not args.no_family_pair)
    print_report(candidates, selected, reasoning, args.gap_weight)

    out_path = results_dir / "shortlist.json"
    with open(out_path, "w") as f:
        json.dump({
            "selected": selected,
            "reasoning": reasoning,
            "candidates": candidates,
            "params": vars(args),
        }, f, indent=2, default=str)
    print(f"\nWritten → {out_path}")


if __name__ == "__main__":
    main()
