"""
Evaluation metrics for all five experiments.
 
Reports:
  Binary tasks  (Exp 1, 2, 4, 5) : Accuracy, F1, ROC-AUC, Confusion Matrix
  Multi-class   (Exp 3)           : Accuracy, macro-F1, per-class F1
"""
 
from __future__ import annotations
 
import numpy as np
from typing import Dict, List, Optional
from src.pipeline.dataset import AUDIO_TYPE_GROUPS, COL_TO_GROUP
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

log = logging.getLogger(__name__)
log_metrics = logging.getLogger(__name__)
 
 
 
def compute_metrics(all_labels: List[int],
                    all_preds:  List[int],
                    all_probs:  Optional[np.ndarray],
                    num_classes: int,
                    split_name: str = "test") -> Dict:
    """
    Compute and log all evaluation metrics.
 
    Parameters
    ----------
    all_labels  : ground-truth integer labels.
    all_preds   : predicted integer labels (argmax).
    all_probs   : predicted probabilities [N, num_classes] for AUC.
    num_classes : 2 for binary, 3 for trajectory experiment.
    split_name  : label for logging ("val" or "test").
 
    Returns
    -------
    metrics dict with keys: accuracy, f1_macro, f1_per_class,
                             roc_auc (binary only), confusion_matrix.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report
    )
 
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
 
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per   = f1_score(labels, preds, average=None, zero_division=0).tolist()
    cm       = confusion_matrix(labels, preds).tolist()
 
    metrics = {
        f"{split_name}/accuracy":     float(accuracy),
        f"{split_name}/f1_macro":     float(f1_macro),
        f"{split_name}/f1_per_class": f1_per,
        f"{split_name}/confusion_matrix": cm,
    }
 
    # ROC-AUC (binary only — multi-class needs OvR)
    if all_probs is not None:
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, all_probs[:, 1])
                metrics[f"{split_name}/roc_auc"] = float(auc)
            else:
                auc = roc_auc_score(labels, all_probs,
                                    multi_class="ovr", average="macro")
                metrics[f"{split_name}/roc_auc_macro"] = float(auc)
        except Exception as e:
            log.warning(f"ROC-AUC computation failed: {e}")
 
    # Log human-readable report
    log.info(f"\n{'─'*50}")
    log.info(f"  Evaluation [{split_name.upper()}]")
    log.info(f"{'─'*50}")
    log.info(f"  Accuracy  : {accuracy:.4f}")
    log.info(f"  F1 (macro): {f1_macro:.4f}")
    if f"{split_name}/roc_auc" in metrics:
        log.info(f"  ROC-AUC   : {metrics[f'{split_name}/roc_auc']:.4f}")
    log.info(f"\n  Classification Report:")
    report = classification_report(labels, preds, zero_division=0)
    for line in report.split("\n"):
        log.info(f"    {line}")
    log.info(f"\n  Confusion Matrix:")
    for row in cm:
        log.info(f"    {row}")
 
    return metrics


def evaluate_by_audio_type(
    all_labels:    List[int],
    all_preds:     List[int],
    all_probs:     np.ndarray,
    all_audio_cols: List[str],
    num_classes:   int,
    split_name:    str = "test",
) -> Dict[str, Dict]:
    """
    Slice evaluation results by audio type and compute metrics per type.
 
    Parameters
    ----------
    all_labels     : ground-truth labels for every test segment.
    all_preds      : predicted labels for every test segment.
    all_probs      : predicted probabilities [N, num_classes].
    all_audio_cols : audio column tag for every test segment.
    num_classes    : 2 for binary, 3 for trajectory.
    split_name     : "test" or "val".
 
    Returns
    -------
    per_type_metrics : {
        "vowels":    {accuracy, f1_macro, roc_auc, n_segments, ...},
        "sustained": {...},
        "speech":    {...},
        "tdu":       {...},
        "a":         {...},   ← also per individual column
        ...
    }
    """
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)
    cols   = np.array(all_audio_cols)
 
    per_type_metrics: Dict[str, Dict] = {}
 
    # ── Per individual column (a, e, speech, agua, ...) ──────────────────
    for col in np.unique(cols):
        mask = cols == col
        if mask.sum() < 5:
            # Too few samples for reliable metrics — skip
            continue
 
        y_true = labels[mask]
        y_pred = preds[mask]
        y_prob = probs[mask]
 
        metrics = _compute_slice_metrics(
            y_true, y_pred, y_prob, num_classes,
            name=f"{split_name}/{col}"
        )
        metrics["n_segments"] = int(mask.sum())
        per_type_metrics[col] = metrics
 
    # ── Per audio type GROUP (vowels, sustained, speech, tdu) ─────────────
    for group, group_cols in AUDIO_TYPE_GROUPS.items():
        mask = np.isin(cols, group_cols)
        if mask.sum() < 5:
            continue
 
        y_true = labels[mask]
        y_pred = preds[mask]
        y_prob = probs[mask]
 
        metrics = _compute_slice_metrics(
            y_true, y_pred, y_prob, num_classes,
            name=f"{split_name}/{group}"
        )
        metrics["n_segments"]  = int(mask.sum())
        metrics["cols_in_group"] = group_cols
        per_type_metrics[group] = metrics
 
    # ── Overall (sanity check, same as existing evaluate()) ───────────────
    overall = _compute_slice_metrics(
        labels, preds, probs, num_classes,
        name=f"{split_name}/overall"
    )
    overall["n_segments"] = len(labels)
    per_type_metrics["overall"] = overall
 
    # ── Print comparison table ────────────────────────────────────────────
    _print_audio_type_table(per_type_metrics, split_name)
 
    return per_type_metrics
 
 
def _compute_slice_metrics(y_true, y_pred, y_prob,
                            num_classes, name) -> Dict:
    """Compute metrics for a single slice of the test set."""
    if len(np.unique(y_true)) < 2:
        # Only one class present in this slice — metrics are degenerate
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": 0.0,
            "roc_auc":  None,
            "note":     "single class in slice",
        }
 
    acc      = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred,
                               average="macro", zero_division=0))
    f1_per   = f1_score(y_true, y_pred,
                         average=None, zero_division=0).tolist()
    cm       = confusion_matrix(y_true, y_pred).tolist()
 
    result = {
        "accuracy":        acc,
        "f1_macro":        f1_macro,
        "f1_per_class":    f1_per,
        "confusion_matrix": cm,
    }
 
    try:
        if num_classes == 2:
            result["roc_auc"] = float(
                roc_auc_score(y_true, y_prob[:, 1])
            )
        else:
            result["roc_auc"] = float(
                roc_auc_score(y_true, y_prob,
                               multi_class="ovr", average="macro")
            )
    except Exception:
        result["roc_auc"] = None
 
    return result
 
 
def _print_audio_type_table(per_type_metrics: Dict, split_name: str):
    """Print a formatted comparison table to the console."""
    log_metrics.info(f"\n{'═'*72}")
    log_metrics.info(f"  PER AUDIO TYPE RESULTS  [{split_name.upper()}]")
    log_metrics.info(f"{'═'*72}")
    log_metrics.info(
        f"  {'TYPE':<14} {'N':>6}  {'ACC':>6}  {'F1':>6}  {'AUC':>6}"
    )
    log_metrics.info(f"  {'─'*60}")
 
    # Print groups first
    group_order = ["overall"] + list(AUDIO_TYPE_GROUPS.keys())
    printed = set()
 
    for key in group_order:
        if key not in per_type_metrics:
            continue
        m = per_type_metrics[key]
        auc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "  N/A "
        log_metrics.info(
            f"  {key.upper():<14} {m['n_segments']:>6}  "
            f"{m['accuracy']:>6.4f}  {m['f1_macro']:>6.4f}  {auc_str:>6}"
        )
        printed.add(key)
 
    # Individual columns below a separator
    log_metrics.info(f"  {'─'*60}")
    log_metrics.info(f"  {'  (per column)'}")
 
    for col in sorted(per_type_metrics.keys()):
        if col in printed:
            continue
        m = per_type_metrics[col]
        auc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "  N/A "
        group   = COL_TO_GROUP.get(col, "")
        log_metrics.info(
            f"  {col:<14} {m['n_segments']:>6}  "
            f"{m['accuracy']:>6.4f}  {m['f1_macro']:>6.4f}  {auc_str:>6}"
            f"  [{group}]"
        )
 
    log_metrics.info(f"{'═'*72}\n")
 