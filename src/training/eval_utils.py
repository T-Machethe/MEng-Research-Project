"""
src/training/eval_utils.py
─────────────────────────────────────────────────────────────────────────────
Post-training evaluation utilities.

calibrate_threshold()   — find the probability threshold on the val set that
                          maximises val F1-macro, then apply it to test.
patient_level_vote()    — aggregate segment-level predictions to patient-level
                          by majority vote, which is the clinically meaningful
                          unit of analysis.

Both are non-destructive: they produce additional metrics stored alongside
the segment-level results, not replacing them.
"""

from __future__ import annotations
import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(
    val_probs:   np.ndarray,   # [N_val, num_classes]
    val_labels:  np.ndarray,   # [N_val]
    test_probs:  np.ndarray,   # [N_test, num_classes]
    test_labels: np.ndarray,   # [N_test]
    num_classes: int,
    search_steps: int = 100,
) -> dict:
    """
    For binary classification: sweep the decision threshold on val set,
    pick the threshold t* that maximises val F1-macro, then evaluate test
    using t*.

    For multiclass: argmax is always optimal; threshold calibration is not
    meaningful, so this returns a note and skips.

    Why this matters:
        We train with oversampled (balanced) data but test on the true
        class distribution. A model trained on 50/50 data with a 0.5
        threshold performs poorly on 75/25 test data because the 0.5
        cutoff doesn't account for the prior shift. The optimal threshold
        is typically 0.3–0.4 in this setting.

    Returns
    -------
    dict with keys:
        optimal_threshold, val_f1_at_threshold, test_f1_calibrated,
        test_acc_calibrated, test_auc_calibrated,
        test_confusion_calibrated
    """
    if num_classes != 2:
        return {"note": "Threshold calibration skipped for multiclass."}

    # Probability of class 1
    val_p1  = val_probs[:, 1]
    test_p1 = test_probs[:, 1]

    # Grid search over thresholds
    thresholds = np.linspace(0.1, 0.9, search_steps)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (val_p1 >= t).astype(int)
        f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # Apply optimal threshold to test set
    test_preds_cal = (test_p1 >= best_t).astype(int)
    test_f1_cal    = f1_score(test_labels, test_preds_cal,
                              average="macro", zero_division=0)
    test_acc_cal   = accuracy_score(test_labels, test_preds_cal)

    try:
        test_auc_cal = roc_auc_score(test_labels, test_p1)
    except Exception:
        test_auc_cal = float("nan")

    from sklearn.metrics import confusion_matrix
    test_cm_cal = confusion_matrix(test_labels, test_preds_cal).tolist()

    log.info(
        f"  [Threshold] optimal t={best_t:.3f} | "
        f"val_f1={best_f1:.4f} | "
        f"test_f1={test_f1_cal:.4f} (was {f1_score(test_labels, (test_p1>=0.5).astype(int), average='macro', zero_division=0):.4f} at t=0.5)"
    )

    return {
        "optimal_threshold":       float(best_t),
        "val_f1_at_threshold":     float(best_f1),
        "test_f1_calibrated":      float(test_f1_cal),
        "test_acc_calibrated":     float(test_acc_cal),
        "test_auc_calibrated":     float(test_auc_cal),
        "test_confusion_calibrated": test_cm_cal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level majority vote
# ─────────────────────────────────────────────────────────────────────────────

def patient_level_vote(
    all_labels:   list,          # segment-level true labels
    all_preds:    list,          # segment-level predicted labels
    all_probs:    np.ndarray,    # segment-level probabilities [N, C]
    patient_ids:  list,          # patient ID for each segment
    num_classes:  int,
) -> dict:
    """
    Aggregate segment-level predictions to patient level.

    Two aggregation strategies:
      vote  — majority vote on predicted class per patient
      prob  — average probability per patient, then argmax

    Why this matters:
        The clinical question is 'does this patient have CRS', not 'does
        this segment have CRS'. A patient with 20 segments, 14 predicted
        as class 1 and 6 as class 0, should count as class 1. Segment-level
        accuracy inflates results when a patient has many easy segments.
        Patient-level evaluation is the honest metric.

    Returns
    -------
    dict with vote and prob results for each strategy.
    """
    from collections import defaultdict

    patients = sorted(set(patient_ids))
    if not patients:
        return {"note": "No patient IDs provided."}

    # Bucket segments by patient
    buckets_labels = defaultdict(list)
    buckets_preds  = defaultdict(list)
    buckets_probs  = defaultdict(list)

    for lbl, pred, prob, pid in zip(all_labels, all_preds,
                                     all_probs, patient_ids):
        buckets_labels[pid].append(lbl)
        buckets_preds[pid].append(pred)
        buckets_probs[pid].append(prob)

    pt_true, pt_vote, pt_prob_pred, pt_prob = [], [], [], []

    for pid in patients:
        true_lbls = buckets_labels[pid]
        # Ground truth: majority of segment labels (should be uniform per patient)
        pt_true.append(max(set(true_lbls), key=true_lbls.count))

        # Strategy 1: majority vote on predicted class
        preds = buckets_preds[pid]
        pt_vote.append(max(set(preds), key=preds.count))

        # Strategy 2: mean probability → argmax
        mean_prob = np.mean(buckets_probs[pid], axis=0)
        pt_prob.append(mean_prob)
        pt_prob_pred.append(int(np.argmax(mean_prob)))

    pt_true      = np.array(pt_true)
    pt_vote      = np.array(pt_vote)
    pt_prob_pred = np.array(pt_prob_pred)
    pt_prob_arr  = np.array(pt_prob)   # [N_patients, C]

    def _metrics(true, pred, proba=None):
        acc = accuracy_score(true, pred)
        f1  = f1_score(true, pred, average="macro", zero_division=0)
        auc = float("nan")
        if proba is not None and num_classes == 2:
            try:
                auc = roc_auc_score(true, proba[:, 1])
            except Exception:
                pass
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true, pred).tolist()
        return {"n_patients": len(true), "accuracy": float(acc),
                "f1_macro": float(f1), "roc_auc": float(auc),
                "confusion_matrix": cm}

    vote_metrics     = _metrics(pt_true, pt_vote)
    prob_avg_metrics = _metrics(pt_true, pt_prob_pred, pt_prob_arr)

    log.info(
        f"  [Patient vote]    n={len(patients)} | "
        f"acc={vote_metrics['accuracy']:.4f} | "
        f"f1={vote_metrics['f1_macro']:.4f}"
    )
    log.info(
        f"  [Patient prob_avg] n={len(patients)} | "
        f"acc={prob_avg_metrics['accuracy']:.4f} | "
        f"f1={prob_avg_metrics['f1_macro']:.4f}"
    )

    return {
        "majority_vote": vote_metrics,
        "prob_average":  prob_avg_metrics,
    }