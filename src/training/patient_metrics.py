"""
src/training/patient_metrics.py
─────────────────────────────────────────────────────────────────────────────
Shared segment-level → patient-level aggregation utilities.

Why this exists
────────────────
Every metric produced by src/training/metrics.py::compute_metrics() inside
Trainer._evaluate() is pooled over SEGMENTS, not patients. A patient with
30 easy segments and a patient with 2 hard segments contribute unequally
to a segment-pooled accuracy number, and segments from the same patient
are not independent samples — so segment-level accuracy is a different
(and typically more optimistic-looking) statistic than patient-level
accuracy. This was flagged by the examiner and was previously only
addressed after the fact via a standalone per-patient-breakdown script.
It is now computed inline for Exp1 (see BaseExperiment.run()) and is the
PRIMARY metric reported by the Exp5 cross-cohort specificity script.

Used by
────────
  - src/experiments/base.py::BaseExperiment.run()   (Exp1 test set)
  - scripts/run_cross_cohort_specificity.py          (Exp5)

Patient identity is NOT wired into SinusitisDataset.__getitem__ (a known
architectural gap — see per-patient-breakdown notes: the same gap is
called out in a trainer.py comment). Patient ID is instead recovered by
parsing the "ID{subject}_ses{session}_{col}_seg{idx}.pt" filename
convention off `SinusitisDataset.samples`, in the exact order a
shuffle=False DataLoader iterates them — this is why every caller of
these functions must come from a shuffle=False loader/dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.training.metrics import compute_metrics


def patient_ids_from_samples(samples) -> List[str]:
    """
    Parameters
    ----------
    samples : SinusitisDataset.samples
        A list of (filepath, label, audio_col) tuples, in the exact order
        a shuffle=False DataLoader over that dataset iterates them.

    Returns
    -------
    List[str] of patient IDs, same length and order as `samples`.
    """
    ids = []
    for fpath, _label, _col in samples:
        stem = Path(fpath).stem              # "ID12_ses1_a_seg0003"
        ids.append(stem.split("_")[0].replace("ID", ""))
    return ids


def aggregate_to_patient_level(probs: np.ndarray,
                                patient_ids: List[str],
                                labels: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Mean-pools segment-level predicted probabilities per patient.

    Mean pooling (rather than majority vote on hard predictions) is used
    for consistency with the aggregation convention already established
    in lund_mackay_correlation.py, and because patient_level_vote() in
    eval_utils.py is dead code that was never wired up to real predictions.

    Parameters
    ----------
    probs       : [N, num_classes] segment-level predicted probabilities.
    patient_ids : length-N list, aligned index-for-index with `probs`.
    labels      : optional length-N ground-truth labels. If provided, the
                  label is asserted CONSTANT within each patient (it must
                  be, since label is a per-patient clinical property) —
                  a mismatch raises rather than silently picking one.

    Returns
    -------
    One row per patient: ID, p_class{0..k}, n_segments, [label].
    """
    n_classes = probs.shape[1]
    data = {"ID": patient_ids}
    for c in range(n_classes):
        data[f"p_class{c}"] = probs[:, c]
    if labels is not None:
        data["label"] = np.asarray(labels)
    rec = pd.DataFrame(data)

    agg = rec.groupby("ID").agg({f"p_class{c}": "mean" for c in range(n_classes)})
    agg["n_segments"] = rec.groupby("ID").size()

    if labels is not None:
        label_nunique = rec.groupby("ID")["label"].nunique()
        inconsistent = label_nunique[label_nunique > 1]
        if len(inconsistent) > 0:
            raise ValueError(
                f"Patient(s) with inconsistent segment-level labels — "
                f"cannot aggregate to patient level safely: "
                f"{list(inconsistent.index)}"
            )
        agg["label"] = rec.groupby("ID")["label"].first()

    return agg.reset_index()


def compute_patient_level_metrics(patient_df: pd.DataFrame,
                                   num_classes: int,
                                   split_name: str = "test") -> Dict:
    """
    Derives patient-level accuracy / F1 / AUC / confusion-matrix from a
    per-patient aggregated probability table (see
    aggregate_to_patient_level), reusing compute_metrics() so the output
    schema matches the rest of the repo (metrics always derived from a
    confusion matrix, per repo convention).

    Requires ground-truth `label` in patient_df — for cohorts with no
    reliable ground truth (e.g. Exp5's Sept/Tonsill cohorts), use
    scripts/run_cross_cohort_specificity.py::specificity_metrics() instead,
    which computes the assumed-negative specificity variant of this.
    """
    if len(patient_df) == 0 or "label" not in patient_df.columns:
        return {}

    prob_cols = sorted([c for c in patient_df.columns if c.startswith("p_class")])
    probs  = patient_df[prob_cols].values
    preds  = probs.argmax(axis=1).tolist()
    labels = patient_df["label"].astype(int).tolist()

    key = f"{split_name}/patient_level"
    metrics = compute_metrics(labels, preds, probs, num_classes, split_name=key)
    metrics[f"{key}/n_patients"] = int(len(patient_df))
    return metrics
