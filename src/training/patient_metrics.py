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

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.training.metrics import compute_metrics

log = logging.getLogger(__name__)


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
                                labels: Optional[np.ndarray] = None,
                                recording_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    TWO-STAGE aggregation, per examiner instruction: mean-pool segment-
    level probabilities within each RECORDING first, then mean-pool
    those recording-level means within each PATIENT — rather than a
    flat segment-to-patient mean. These are NOT equivalent whenever
    segment counts differ across recordings for the same patient, which
    is the normal case here: e.g. the free-speech task yields ~115
    segments per recording vs. ~1 for a sustained-vowel task (confirmed
    from run_diagnostic_pipeline.py output). A flat mean lets
    high-yield recordings numerically dominate; this doesn't.

    "Recording" = (patient_id, recording_id) pair. Every current caller
    passes recording_id = the audio task/column (via
    audio_types_from_samples()) — valid because both Exp1 and Exp5's
    populations are Session-1-only (Exp1CRSvsControl filters session==1;
    run_cross_cohort_specificity.py's load_and_filter_csv does too for
    Sept/Tonsill), so (patient, session, audio_col) reduces to (patient,
    audio_col) with session held constant. This assumption does NOT
    extend to multi-session cohorts (Exp2/3/4) without also including
    session in recording_ids — don't reuse this for those without
    revisiting that.

    recording_ids is OPTIONAL, defaulting to None, for one reason only:
    scripts/run_nested_cv.py's three call sites are not yet updated to
    pass it (deliberately deferred — the fix there would invalidate
    already-completed nested CV folds for wav2vec2_scratch and
    wav2vec2_finetune, whose per-fold trained models are already deleted
    -- see module history -- so applying it means a full retrain, not a
    rerun). Passing recording_ids=None preserves the OLD flat one-stage
    mean EXACTLY (byte-for-byte — see the branch below), with a warning
    logged every time, so nested_cv.py keeps working unmodified until
    that retrain is deliberately scheduled. Every OTHER caller (Exp1,
    Exp5) has been updated to always pass real recording_ids — treat
    seeing that warning fire from anywhere other than run_nested_cv.py
    as a bug to fix, not something to silence.

    Mean pooling (rather than majority vote on hard predictions) is
    still used at both stages, for consistency with the aggregation
    convention already established in lund_mackay_correlation.py, and
    because patient_level_vote() in eval_utils.py is dead code that was
    never wired up to real predictions.

    Parameters
    ----------
    probs         : [N, num_classes] segment-level predicted probabilities.
    patient_ids   : length-N list, aligned index-for-index with `probs`.
    labels        : optional length-N ground-truth labels. Asserted CONSTANT
                    within each patient (a mismatch raises) — checked at
                    the recording level too when recording_ids is given,
                    as a stricter version of the same invariant. Kept in
                    its ORIGINAL 3rd positional slot (unlike recording_ids,
                    which is new and comes after it) specifically so every
                    existing positional call — including run_nested_cv.py's
                    three, deliberately left unmodified — keeps working
                    unchanged; putting recording_ids before labels would
                    have silently reinterpreted those callers' labels
                    arrays as recording_ids instead, breaking every
                    downstream label-dependent metric with no error.
    recording_ids : length-N list, aligned index-for-index with `probs` —
                    identifies which recording each segment came from.
                    None = old flat behavior (deprecated, nested_cv.py only).

    Returns
    -------
    One row per patient: ID, p_class{0..k}, n_segments, [n_recordings],
    [label]. n_recordings is only present when recording_ids is given.
    """
    n_classes = probs.shape[1]
    prob_cols = [f"p_class{c}" for c in range(n_classes)]

    if recording_ids is None:
        log.warning(
            "  [aggregate_to_patient_level] recording_ids not provided — "
            "using the OLD flat segment-to-patient mean, not the two-stage "
            "(recording-then-patient) aggregation. Expected only from "
            "scripts/run_nested_cv.py right now (deliberately deferred — "
            "see docstring); anywhere else, this is a bug."
        )
        data = {"ID": patient_ids}
        for c in range(n_classes):
            data[f"p_class{c}"] = probs[:, c]
        if labels is not None:
            data["label"] = np.asarray(labels)
        rec = pd.DataFrame(data)

        agg = rec.groupby("ID").agg({c: "mean" for c in prob_cols})
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

    # ── Two-stage: segment -> recording -> patient ──────────────────────────
    data = {"ID": patient_ids, "_recording": recording_ids}
    for c in range(n_classes):
        data[f"p_class{c}"] = probs[:, c]
    if labels is not None:
        data["label"] = np.asarray(labels)
    rec = pd.DataFrame(data)

    # Stage 1: segment -> recording. One row per (ID, recording) after this.
    stage1 = rec.groupby(["ID", "_recording"]).agg({c: "mean" for c in prob_cols})
    if labels is not None:
        label_nunique_rec = rec.groupby(["ID", "_recording"])["label"].nunique()
        inconsistent_rec = label_nunique_rec[label_nunique_rec > 1]
        if len(inconsistent_rec) > 0:
            raise ValueError(
                f"(Patient, recording) pair(s) with inconsistent segment-"
                f"level labels — cannot aggregate safely: "
                f"{list(inconsistent_rec.index)}"
            )
        stage1["label"] = rec.groupby(["ID", "_recording"])["label"].first()
    stage1 = stage1.reset_index()

    # Stage 2: recording -> patient. Each recording contributes ONE value
    # here regardless of how many segments it had — this is the actual fix.
    agg = stage1.groupby("ID").agg({c: "mean" for c in prob_cols})
    agg["n_segments"]   = rec.groupby("ID").size()      # total raw segments — context only
    agg["n_recordings"] = stage1.groupby("ID").size()   # how many recordings actually contributed

    if labels is not None:
        label_nunique = stage1.groupby("ID")["label"].nunique()
        inconsistent = label_nunique[label_nunique > 1]
        if len(inconsistent) > 0:
            raise ValueError(
                f"Patient(s) with inconsistent segment-level labels — "
                f"cannot aggregate to patient level safely: "
                f"{list(inconsistent.index)}"
            )
        agg["label"] = stage1.groupby("ID")["label"].first()

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


def audio_types_from_samples(samples) -> List[str]:
    """
    Unlike patient ID (which needs filename parsing — see module docstring),
    the audio column is already the third element of every
    SinusitisDataset.samples tuple: (filepath, label, audio_col) — see
    src/pipeline/dataset.py::SinusitisDataset.__init__. No parsing needed,
    just pull it straight out, in the same shuffle=False order everything
    else here depends on.
    """
    return [col for _fpath, _label, col in samples]


def aggregate_to_patient_audiotype_level(probs: np.ndarray,
                                          patient_ids: List[str],
                                          audio_types: List[str],
                                          labels: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Patient-level aggregation, broken down BY AUDIO TYPE — the patient-
    level analogue of src/training/metrics.py::evaluate_by_audio_type(),
    which does this at the segment level only. Answers "for THIS
    patient's vowel-only segments, what's the mean predicted
    probability?" rather than pooling every segment together regardless
    of which recording task it came from.

    Groups by (ID, audio_col) for the individual-column breakdown (a, a1,
    a2, a3, agua, brasero, dia, e, i, mesa, o, u — matching the segment-
    level report's rows exactly), by (ID, group) for the grouped
    breakdown (vowels/sustained/speech/tdu, via COL_TO_GROUP), and adds
    an "overall" pseudo-type per patient pooling everything — same three-
    tier structure the segment-level per-audio-type table already uses,
    so a patient-level report can mirror it row-for-row.

    Returns a long-format DataFrame: one row per (ID, audio_type), with
    audio_type covering individual columns AND groups AND "overall" —
    filter on the `level` column ("column" | "group" | "overall") to get
    just one tier.
    """
    from src.pipeline.dataset import COL_TO_GROUP

    n_classes = probs.shape[1]
    base = {"ID": patient_ids, "audio_col": audio_types}
    for c in range(n_classes):
        base[f"p_class{c}"] = probs[:, c]
    if labels is not None:
        base["label"] = np.asarray(labels)
    rec = pd.DataFrame(base)
    rec["group"] = rec["audio_col"].map(lambda c: COL_TO_GROUP.get(c, "other"))

    prob_cols = [f"p_class{c}" for c in range(n_classes)]

    def _agg(df: pd.DataFrame, group_col: str, level: str) -> pd.DataFrame:
        agg_dict = {c: "mean" for c in prob_cols}
        out = df.groupby(["ID", group_col]).agg(agg_dict)
        out["n_segments"] = df.groupby(["ID", group_col]).size()
        if labels is not None:
            label_nunique = df.groupby(["ID", group_col])["label"].nunique()
            inconsistent = label_nunique[label_nunique > 1]
            if len(inconsistent) > 0:
                raise ValueError(
                    f"Patient/audio-type combo(s) with inconsistent labels — "
                    f"cannot aggregate safely: {list(inconsistent.index)}"
                )
            out["label"] = df.groupby(["ID", group_col])["label"].first()
        out = out.reset_index().rename(columns={group_col: "audio_type"})
        out["level"] = level
        return out

    # "column" tier is correct as a flat mean — by construction, one
    # audio_col IS one recording, so pooling its segments flat already
    # gives the correct per-recording mean (this IS Stage 1).
    by_column = _agg(rec, "audio_col", "column")

    # "group" and "overall" both aggregate ACROSS MULTIPLE recordings
    # (e.g. "vowels" = the a/e/i/o/u recordings; "overall" = every
    # recording a patient has) — so both need the SAME two-stage fix as
    # aggregate_to_patient_level(): mean of the per-recording ("column"
    # tier) means, not a flat re-mean of raw segments. A flat mean here
    # lets whichever recording within the group happens to have more
    # segments dominate — the identical bug being fixed everywhere else
    # in this module, just one level up (group-of-recordings rather than
    # patient-of-recordings).
    by_column_grouped = by_column.copy()
    col_to_group = rec.drop_duplicates("audio_col").set_index("audio_col")["group"].to_dict()
    by_column_grouped["group"] = by_column_grouped["audio_type"].map(col_to_group)

    def _stage2_from_recordings(col_df: pd.DataFrame, group_key: Optional[str], level: str) -> pd.DataFrame:
        """
        Stage 2 applied to the (already Stage-1-correct) column tier:
        mean the per-recording means, grouped by `group_key` within ID
        (or just by ID alone, for the 'overall' tier, when group_key is
        None). This is exactly aggregate_to_patient_level()'s Stage 2,
        reused here rather than duplicated with different variable names.
        """
        keys = ["ID"] if group_key is None else ["ID", group_key]
        out = col_df.groupby(keys).agg({c: "mean" for c in prob_cols})
        out["n_recordings"] = col_df.groupby(keys).size()
        out["n_segments"]   = col_df.groupby(keys)["n_segments"].sum()  # total raw segments — context only
        if "label" in col_df.columns:
            label_nunique = col_df.groupby(keys)["label"].nunique()
            inconsistent = label_nunique[label_nunique > 1]
            if len(inconsistent) > 0:
                raise ValueError(
                    f"Patient(s) with inconsistent labels across recordings — "
                    f"cannot aggregate '{level}' tier safely: "
                    f"{list(inconsistent.index)}"
                )
            out["label"] = col_df.groupby(keys)["label"].first()
        out = out.reset_index()
        if group_key is not None:
            out = out.rename(columns={group_key: "audio_type"})
        else:
            out["audio_type"] = "overall"
        out["level"] = level
        return out

    by_group   = _stage2_from_recordings(by_column_grouped, "group", "group")
    by_overall = _stage2_from_recordings(by_column_grouped, None, "overall")

    return pd.concat([by_column, by_group, by_overall], ignore_index=True)


def compute_patient_audiotype_metrics(patient_audiotype_df: pd.DataFrame,
                                       num_classes: int,
                                       split_name: str = "test") -> pd.DataFrame:
    """
    Per-audio-type patient-level accuracy/F1/AUC, one row per audio_type
    — the patient-level analogue of the per-audio-type table in the
    segment-level PDF report. Requires ground-truth `label` (present when
    aggregate_to_patient_audiotype_level was called with labels — i.e.
    Exp1, not Exp5's Sept/Tonsill).
    """
    if "label" not in patient_audiotype_df.columns:
        return pd.DataFrame()

    prob_cols = sorted([c for c in patient_audiotype_df.columns if c.startswith("p_class")])
    rows = []
    for audio_type, sub in patient_audiotype_df.groupby("audio_type"):
        probs  = sub[prob_cols].values
        preds  = probs.argmax(axis=1).tolist()
        labels = sub["label"].astype(int).tolist()
        m = compute_metrics(labels, preds, probs, num_classes,
                             split_name=f"{split_name}/patient_level/{audio_type}")
        rows.append({
            "audio_type":  audio_type,
            "level":       sub["level"].iloc[0],
            "n_patients":  len(sub),
            "accuracy":    m.get(f"{split_name}/patient_level/{audio_type}/accuracy"),
            "f1_macro":    m.get(f"{split_name}/patient_level/{audio_type}/f1_macro"),
            "roc_auc":     m.get(f"{split_name}/patient_level/{audio_type}/roc_auc"),
        })
    return pd.DataFrame(rows)
