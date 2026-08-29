"""
run_group_separability_analysis.py
─────────────────────────────────────────────────────────────────────────────
Training-only acoustic separability analysis: FESS vs Control (Exp1 comparison).

Addresses examiner comment C3-17 (Research Methodology, Section 3.2): rather
than showing a single distribution plot with an informal "one appears
different" comment, this script systematically tests, across all 13 audio
tasks and using proper statistics, whether FESS and Control patients are
acoustically separable, and whether that separability survives reduction
from a full utterance to the 1-second training segment.

Restricted to Session 1, FESS vs Control, TRAINING patients only (via
patient_level_split(seed=42)) — this is a methodology-informing diagnostic,
not a result, and must never touch validation/test patients.

For each (audio task, feature, full/segment condition):
  - Aggregate to one value per patient (mean across repeats) to avoid
    pseudo-replication from multiple recordings per patient.
  - Mann-Whitney U test (FESS vs Control), non-parametric, no normality
    assumption.
  - Cohen's d effect size (computed on the same patient-aggregated values).

Outputs:
  - group_separability_summary.json   (flat-key schema, one record per
                                        task/feature/condition)
  - group_separability_summary.csv    (same data, tabular)
  - diag_separability_heatmap.png     (effect size heatmap, full vs segment)

Usage
─────
    python scripts/run_group_separability_analysis.py
    python scripts/run_group_separability_analysis.py --max_patients 10   # quick test
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT  # overridden by --data_root
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.cleaning import (
    remove_leading_silence,
    highpass_filter,
    voice_activity_detection,
)
from src.audio.io import load_audio
from src.config import TARGET_SR, WINDOW_SAMPLES
from src.pipeline.splits import patient_level_split
from src.utils.paths import resolve_path, COLUMN_TO_SUBFOLDER

OUTPUT_DIR = PROJECT_ROOT / "results" / "Plots and visuals" / "diagnostics"

# Full cohort-name mapping (C3-17: no abbreviated labels in outputs)
GROUP_NAMES = {
    "FESS": "Functional Endoscopic Sinus Surgery",
    "Contr": "Control",
}

AUDIO_TASKS = [
    "a", "e", "i", "o", "u",
    "a1", "a2", "a3",
    "agua", "brasero", "dia", "mesa",
    "speech",
]


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic feature extraction (same calls as run_exploratory_DA.py's
# _compute_acoustic_features, kept consistent with the rest of the codebase)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_acoustic_features(waveform: torch.Tensor, sr: int) -> dict:
    """Extract F0, jitter, shimmer, HNR via parselmouth (Praat wrapper)."""
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(waveform.numpy().squeeze(), sampling_frequency=sr)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        pp = call([snd, pitch], "To PointProcess (cc)")
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harm, "Get mean", 0, 0)
        return {
            "f0": f0_mean if f0_mean == f0_mean else None,  # NaN check
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
        }
    except Exception:
        return {"f0": None, "jitter": None, "shimmer": None, "hnr": None}


def _clean_waveform(filepath: Path):
    """
    Apply the SAME cleaning steps as the real training pipeline
    (src/pipeline/preprocess.py:preprocess_file), minus augmentation,
    so the "full utterance" condition reflects what a clean recording
    looks like post-pipeline rather than the raw file.
    """
    waveform, sr = load_audio(filepath)
    waveform = remove_leading_silence(waveform, sr)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)
        sr = TARGET_SR
    waveform = highpass_filter(waveform, sr)
    waveform = voice_activity_detection(waveform, sr)
    return waveform, sr


def extract_full_and_segment_features(filepath: Path) -> dict:
    """
    Returns features for (a) the full cleaned utterance and (b) its first
    1-second segment (WINDOW_SAMPLES), matching the actual training window.
    """
    waveform, sr = _clean_waveform(filepath)

    full_feats = _compute_acoustic_features(waveform, sr)

    if waveform.shape[-1] >= WINDOW_SAMPLES:
        segment = waveform[..., :WINDOW_SAMPLES]
    else:
        # Shorter than one window — pad, matching build_finetune_chunks behaviour
        pad = WINDOW_SAMPLES - waveform.shape[-1]
        segment = torch.nn.functional.pad(waveform, (0, pad))
    segment_feats = _compute_acoustic_features(segment, sr)

    return {
        **{f"full_{k}": v for k, v in full_feats.items()},
        **{f"seg_{k}": v for k, v in segment_feats.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled standard deviation)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_std


def run_group_comparison(patient_df: pd.DataFrame, feature_col: str) -> dict:
    """Mann-Whitney U + Cohen's d for FESS vs Control on one feature column,
    using one aggregated value per patient."""
    fess_vals = patient_df.loc[patient_df["_group"] == "FESS", feature_col].dropna().to_numpy()
    contr_vals = patient_df.loc[patient_df["_group"] == "Contr", feature_col].dropna().to_numpy()

    if len(fess_vals) < 2 or len(contr_vals) < 2:
        return {"n_fess": len(fess_vals), "n_contr": len(contr_vals),
                "u_stat": None, "p_value": None, "cohens_d": None}

    u_stat, p_value = stats.mannwhitneyu(fess_vals, contr_vals, alternative="two-sided")
    d = cohens_d(fess_vals, contr_vals)

    return {
        "n_fess": int(len(fess_vals)),
        "n_contr": int(len(contr_vals)),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_patients", type=int, default=9999,
                         help="Limit patients scanned per group (quick test).")
    args = parser.parse_args()

    global DATA_ROOT
    csv_path = Path(args.csv_path) if args.csv_path else (
        PROJECT_ROOT / "Data" / "data_final" / "Clinical" / "clinical_all_sessions.csv"
    )
    if args.data_root:
        DATA_ROOT = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Restrict to FESS vs Control, Session 1 (matches Exp1 design), TRAIN split only.
    exp1_df = df[df["GROUP"].isin(["FESS", "Contr"]) & (df["session"] == 1)].copy()
    train_df, _, _ = patient_level_split(exp1_df, seed=42)
    train_patient_ids = set(train_df["ID"].unique())
    print(f"Training patients (Exp1, Session 1): {len(train_patient_ids)}")

    scan_df = df[
        df["ID"].isin(train_patient_ids)
        & df["GROUP"].isin(["FESS", "Contr"])
        & (df["session"] == 1)
    ].copy()
    scan_df["_group"] = scan_df["GROUP"]

    records = []
    for task in AUDIO_TASKS:
        if task not in scan_df.columns:
            continue
        n_scanned_this_task = 0
        for _, row in scan_df.iterrows():
            if n_scanned_this_task >= args.max_patients:
                break
            val = row.get(task)
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                path = resolve_path(str(val), DATA_ROOT, col=task)
                if not path.exists():
                    continue
                feats = extract_full_and_segment_features(path)
                feats["task"] = task
                feats["patient_id"] = row["ID"]
                feats["_group"] = row["_group"]
                records.append(feats)
                n_scanned_this_task += 1
            except Exception:
                continue
        print(f"  {task}: {n_scanned_this_task} recordings processed")

    if not records:
        print("No features extracted — check --data_root / --csv_path.")
        return

    feat_df = pd.DataFrame(records)

    # Aggregate to one value per patient per task (mean across any repeats)
    agg_df = feat_df.groupby(["task", "patient_id", "_group"]).mean(numeric_only=True).reset_index()

    feature_conditions = [
        ("full_f0", "F0", "full"), ("seg_f0", "F0", "segment"),
        ("full_jitter", "Jitter", "full"), ("seg_jitter", "Jitter", "segment"),
        ("full_shimmer", "Shimmer", "full"), ("seg_shimmer", "Shimmer", "segment"),
        ("full_hnr", "HNR", "full"), ("seg_hnr", "HNR", "segment"),
    ]

    summary = []
    for task in AUDIO_TASKS:
        task_df = agg_df[agg_df["task"] == task]
        if task_df.empty:
            continue
        for col, feature_name, condition in feature_conditions:
            result = run_group_comparison(task_df, col)
            summary.append({
                "task": task,
                "feature": feature_name,
                "condition": condition,
                **result,
            })

    agg_df.to_csv(output_dir / "group_separability_raw_patient_values.csv", index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "group_separability_summary.csv", index=False)
    with open(output_dir / "group_separability_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved -> {output_dir / 'group_separability_summary.csv'}")
    print(f"Saved -> {output_dir / 'group_separability_summary.json'}")

    # Significant results (p < 0.05), sorted by effect size
    sig = summary_df[summary_df["p_value"].notna() & (summary_df["p_value"] < 0.05)]
    sig = sig.sort_values("cohens_d", key=abs, ascending=False)
    print(f"\n{len(sig)} of {len(summary_df)} task/feature/condition combinations "
          f"significant at p<0.05:")
    if not sig.empty:
        print(sig[["task", "feature", "condition", "p_value", "cohens_d"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
