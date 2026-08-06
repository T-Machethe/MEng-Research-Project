"""
scripts/run_cross_cohort_specificity.py
─────────────────────────────────────────────────────────────────────────────
Cross-cohort specificity analysis — the reframed Experiment 5.

Examiner feedback addressed
────────────────────────────
The old Exp5 (`Exp5Generalisation` in src/experiments/all_experiments.py)
TRAINED A NEW MODEL on a session-based pre/post label and then evaluated it
on Septoplasty/Tonsillectomy. The examiner's objection: strong performance
there only shows the new classifier can tell sessions apart — it says
nothing about whether the CRS-detection signal learned in Experiment 1
generalises, because it is neither the same classifier nor the same label.

This script does NOT train anything. It:
  1. Takes the SIX already-trained Exp1 checkpoints (CRS vs Control,
     Session 1 only — see src/experiments/all_experiments.py::Exp1CRSvsControl)
     completely as-is, frozen, no gradient updates.
  2. Applies each one to Session 1 recordings from the Septoplasty and
     Tonsillectomy cohorts (pre-treatment, matching the Exp1 protocol of
     using only a comparable, single timepoint).
  3. Reports what fraction of each cohort the FIXED Exp1 classifier still
     calls "CRS" — i.e. whether the signal is CRS-specific or whether it
     fires on other post-surgical / upper-airway populations too.

This is framed as a cross-cohort SPECIFICITY analysis, not a new
disease-classification experiment. Septoplasty and Tonsillectomy are
surgical cohorts/procedures, not diagnoses of CRS — see the important
caveat below before interpreting results.

── IMPORTANT: CRS-status assumption ────────────────────────────────────────
Per the CUCO dataset paper (Hernández-García et al., Scientific Data 11:746,
2024): Septoplasty corrects septal cartilage deformities (a structural
issue, distinct from CRS mucosal inflammation), and Tonsillectomy patients
are recruited for recurrent tonsillitis (unrelated to the sinuses). Neither
cohort's inclusion criteria mention CRS. This is reasonable published-level
support for treating them as non-CRS, but it is NOT a per-patient clinical
confirmation. If you have access to the raw CUCO per-patient "diagnosis"
metadata field (separate from clinical_all_sessions.csv — see
lund_mackay_correlation.py for where that lives), pass it via
--metadata_csv / --diagnosis_col so this script can flag, per patient,
anything that explicitly mentions CRS/rhinosinusitis/nasal polyps before
you treat that patient as a negative case. Without that file, the script
proceeds using the cohort-level assumption above and prints a loud warning
saying so — check this before writing results into the thesis.

Two-step workflow (mirrors run_augmentation_leak_audit.py's pattern)
─────────────────────────────────────────────────────────────────────
Segments already extracted by run_preprocessing.py were built with
augment=True baked in at the file level (see augmentation-leak-audit
notes) — not appropriate for a clean specificity read-out. So:

  Step 1 (regenerate): re-preprocess ONLY Session 1, Sept + Tonsill,
      straight from the raw WAVs, with augment=False, into fresh
      cohort-specific segment directories (one per training "mode",
      since scratch/finetune segment the waveform differently).

  Step 2 (evaluate): for each of the 6 Exp1 backbones, load
      best_model.pt (frozen), run inference over the clean segments,
      aggregate to patient level, and report specificity metrics.

CLI
───
    --step             identify | regenerate | evaluate | all   (default: all)
    --csv_path         path to clinical_all_sessions.csv
    --metadata_csv     OPTIONAL path to a per-patient diagnosis/metadata
                       CSV (raw CUCO metadata), for the CRS-status check
    --diagnosis_col    column name in --metadata_csv holding free-text
                       diagnosis notes (default: "diagnosis")
    --segment_dir      root dir to write/read the clean Sept/Tonsill
                       Session-1 segments (default:
                       <project_root>/Data/data_final/clean_audio_exp5_specificity)
    --checkpoints_dir  root dir containing exp1_backbone_comparison/
                       (default: <project_root>/MSc_Sinusitis_results)
    --output_dir       where to write results JSON (default:
                       <checkpoints_dir>/exp5_cross_cohort_specificity)
    --project_root     project root for audio path resolution
    --batch_size       inference batch size (default: 16)
    --device           cuda | cpu | auto (default: auto)
    --threshold        decision threshold on P(CRS) for the positive-rate
                       metric (default: 0.5)

Usage examples
──────────────
    # Full pipeline (regenerate clean segments, then evaluate all 6 backbones)
    python scripts/run_cross_cohort_specificity.py --step all \\
        --csv_path /content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv \\
        --checkpoints_dir /content/drive/MyDrive/MSc_Sinusitis_results

    # With the optional per-patient diagnosis cross-check
    python scripts/run_cross_cohort_specificity.py --step all \\
        --metadata_csv /content/drive/MyDrive/Data/raw_cuco_metadata.csv \\
        --diagnosis_col diagnosis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.pipeline.preprocess import process_from_csv
from src.pipeline.dataset import SinusitisDataset
from src.pipeline.dataloader import collate_standard
from src.training.checkpoint import load_checkpoint
from src.training.metrics import compute_metrics
from src.training.patient_metrics import (
    patient_ids_from_samples, aggregate_to_patient_level,
)

# ═════════════════════════════════════════════════════════════════════════
# Fixed identity of the 6 Exp1 backbones — must match how they were trained
# (src/experiments/base.py::ExperimentConfig defaults / run_experiment.py CLI)
# ═════════════════════════════════════════════════════════════════════════

# 3 backbones × 2 training modes = 6 combinations, matching the fixed
# `for backbone in ["wav2vec2","wavlm","xlsr"]: for mode in ["scratch","finetune"]`
# loop in scripts/run_experiment.py (run_key = f"{backbone}_{mode}"). This is
# NOT a curated subset — Exp1 trains and saves all six.
BACKBONE_JOBS = [
    # (output_dir_name, backbone_type, training_mode, pretrained_id)
    ("wav2vec2_scratch",  "wav2vec2", "scratch",  None),
    ("wav2vec2_finetune", "wav2vec2", "finetune", "facebook/wav2vec2-base-960h"),
    ("wavlm_scratch",     "wavlm",    "scratch",  None),
    ("wavlm_finetune",    "wavlm",    "finetune", "microsoft/wavlm-base"),
    ("xlsr_scratch",      "xlsr",     "scratch",  None),
    ("xlsr_finetune",     "xlsr",     "finetune", "facebook/wav2vec2-xls-r-300m"),
]

TEST_GROUPS = ["Sept", "Tonsill"]

CRS_KEYWORDS = [
    "crs", "chronic rhinosinusitis", "rhinosinusitis", "sinusitis",
    "nasal polyp", "nasal polyposis",
]


class _NumpyEncoder(json.JSONEncoder):
    """Same convention as src/experiments/base.py::BaseExperiment.report()."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


# ═════════════════════════════════════════════════════════════════════════
# Step 0 — identify: filter to Session 1, Sept + Tonsill, and run the
# (best-effort) CRS-status cross-check
# ═════════════════════════════════════════════════════════════════════════

def load_and_filter_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["GROUP"] = df["GROUP"].str.strip()
    df["ID"]    = df["ID"].astype(str).str.strip()

    filtered = df[(df["GROUP"].isin(TEST_GROUPS)) & (df["session"] == 1)].copy()

    # Hard assertion, matching the same defensive pattern used in
    # Exp1CRSvsControl: Session 1 is a hard requirement here, not a
    # best-effort filter — if this ever fails, that's a real bug, not a
    # noisy warning to skim past.
    assert (filtered["session"] == 1).all(), (
        "run_cross_cohort_specificity: found non-Session-1 rows after "
        "filtering — Sept/Tonsill must be Session 1 only (comparable "
        "pre-treatment timepoint to Exp1's FESS/Control Session 1)."
    )

    log.info(
        f"\n  [identify] Session-1 rows for {TEST_GROUPS}: {len(filtered)} "
        f"rows across {filtered['ID'].nunique()} patients."
    )
    for grp in TEST_GROUPS:
        sub = filtered[filtered["GROUP"] == grp]
        log.info(f"    {grp:<10} : {sub['ID'].nunique()} patients")

    return filtered


def crs_status_check(filtered: pd.DataFrame,
                      metadata_csv: Optional[str],
                      diagnosis_col: str) -> List[str]:
    """
    Best-effort per-patient check that Sept/Tonsill patients are not
    documented as having CRS. Returns a list of patient IDs flagged for
    manual review (NOT auto-excluded — the examiner asked for confirmation,
    not a silent filter).
    """
    log.info(f"\n  [CRS-status check] {'='*60}")
    if not metadata_csv:
        log.warning(
            "  No --metadata_csv provided. Proceeding on the CUCO-paper-level "
            "assumption that Septoplasty (septal deformity) and "
            "Tonsillectomy (recurrent tonsillitis) patients do not have CRS. "
            "This is a COHORT-level assumption, not a per-patient clinical "
            "confirmation. Confirm against the raw CUCO diagnosis metadata "
            "before reporting results from this analysis as evidence of "
            "CRS specificity."
        )
        return []

    meta = pd.read_csv(metadata_csv)
    if "ID" not in meta.columns or diagnosis_col not in meta.columns:
        log.warning(
            f"  --metadata_csv provided but missing 'ID' or "
            f"'{diagnosis_col}' column — skipping automated check. "
            f"Columns found: {list(meta.columns)}"
        )
        return []

    meta["ID"] = meta["ID"].astype(str).str.strip()
    ids_of_interest = set(filtered["ID"].unique())
    meta = meta[meta["ID"].isin(ids_of_interest)]

    flagged = []
    for _, row in meta.iterrows():
        text = str(row.get(diagnosis_col, "")).lower()
        if any(kw in text for kw in CRS_KEYWORDS):
            flagged.append(row["ID"])

    if flagged:
        log.warning(
            f"  {len(flagged)} patient(s) in {TEST_GROUPS} have a diagnosis "
            f"note mentioning CRS/rhinosinusitis/polyps: {flagged}. "
            f"REVIEW these before including them as negative (non-CRS) "
            f"cases — consider excluding them from the specificity "
            f"denominator or reporting them separately."
        )
    else:
        log.info(
            f"  No CRS/rhinosinusitis/polyp mentions found in diagnosis "
            f"notes for {len(meta)} matched patients."
        )
    return flagged


# ═════════════════════════════════════════════════════════════════════════
# Step 1 — regenerate: clean (unaugmented), Session-1-only segments for
# Sept + Tonsill, built straight from raw WAVs
# ═════════════════════════════════════════════════════════════════════════

def regenerate_clean_segments(filtered: pd.DataFrame,
                               project_root: _Path,
                               segment_dir: _Path,
                               modes: List[str]) -> Dict[str, str]:
    """
    Writes a filtered CSV (Session 1, Sept+Tonsill only) to a temp file and
    calls process_from_csv(augment=False) once per training mode, since
    "scratch" and "finetune" segment the waveform differently
    (sliding_window_segments vs build_finetune_chunks — see
    src/pipeline/preprocess.py::preprocess_file).

    Returns a dict {mode: output_dir_path}.
    """
    out_dirs = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_csv = _Path(tmp) / "exp5_specificity_session1.csv"
        filtered.to_csv(tmp_csv, index=False)

        for mode in modes:
            mode_dir = segment_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                f"\n  [regenerate] mode={mode} -> {mode_dir} "
                f"(augment=False, Session 1, {TEST_GROUPS} only)"
            )
            process_from_csv(
                csv_path=str(tmp_csv),
                project_root=project_root,
                output_dir=str(mode_dir),
                mode=mode,
                augment=False,
            )
            out_dirs[mode] = str(mode_dir)

    return out_dirs


# ═════════════════════════════════════════════════════════════════════════
# Step 2 — evaluate: fixed-classifier inference + specificity metrics
# ═════════════════════════════════════════════════════════════════════════

def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_inference_model(backbone_type: str, mode: str,
                           pretrained: Optional[str], num_classes: int = 2):
    """
    Mirrors src/training/trainer.py::Trainer._build_model() exactly, so the
    reconstructed architecture matches what best_model.pt's state_dict was
    saved from. Kept import-local to avoid pulling transformers at module
    import time for --step identify-only runs.
    """
    from transformers import (
        Wav2Vec2Model, Wav2Vec2Config, WavLMModel, WavLMConfig,
    )
    from src.training.trainer import Wav2Vec2Classifier

    if mode == "scratch":
        if backbone_type == "xlsr":
            config = Wav2Vec2Config(
                hidden_size=1024, num_hidden_layers=24,
                num_attention_heads=16, intermediate_size=4096,
            )
            backbone = Wav2Vec2Model(config)
        elif backbone_type == "wavlm":
            config = WavLMConfig(
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
            )
            backbone = WavLMModel(config)
        else:
            config = Wav2Vec2Config(
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
            )
            backbone = Wav2Vec2Model(config)
    else:
        if backbone_type == "wavlm":
            backbone = WavLMModel.from_pretrained(
                pretrained, mask_time_prob=0.0, mask_feature_prob=0.0)
        else:
            backbone = Wav2Vec2Model.from_pretrained(
                pretrained, mask_time_prob=0.0, mask_feature_prob=0.0)

    hidden_size = backbone.config.hidden_size
    model = Wav2Vec2Classifier(backbone, hidden_size, num_classes)
    return model


def run_inference(model, dataset: SinusitisDataset, device: torch.device,
                   batch_size: int) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (all_probs [N,2], patient_ids [N]) in matching order.
    shuffle=False is load-bearing here — it's what keeps patient_ids
    aligned with model outputs. Patient-ID recovery uses the same shared
    helper as Exp1's patient-level test metrics
    (src/training/patient_metrics.py) so both experiments derive patient
    identity the same way.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_standard,
    )
    patient_ids = patient_ids_from_samples(dataset.samples)

    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_values = torch.nan_to_num(
                batch["input_values"].to(device),
                nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_values=input_values,
                            attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    all_probs_np = np.concatenate(all_probs, axis=0) if all_probs else \
        np.zeros((0, 2), dtype=np.float32)

    assert len(patient_ids) == len(all_probs_np), (
        f"Patient ID / prediction count mismatch: "
        f"{len(patient_ids)} ids vs {len(all_probs_np)} predictions — "
        f"dataset iteration order must have changed."
    )
    return all_probs_np, patient_ids


def aggregate_per_patient(probs: np.ndarray, patient_ids: List[str],
                           df_group_lookup: Dict[str, str]) -> pd.DataFrame:
    """
    Segment-level -> patient-level aggregation via MEAN predicted P(CRS)
    (class 1). Thin wrapper around the shared
    src/training/patient_metrics.py::aggregate_to_patient_level(), which
    is also what Exp1 now uses for its patient-level test metrics — same
    aggregation convention on both sides of the comparison. No ground-
    truth label is passed here (Sept/Tonsill have no confirmed CRS label
    — see crs_status_check()), so this is PATIENT-LEVEL but not scored
    against a label; scoring against the "assumed non-CRS" working
    hypothesis happens in specificity_metrics() below.
    """
    if probs.shape[0] == 0:
        return pd.DataFrame(columns=["ID", "p_crs_mean", "n_segments", "GROUP"])

    agg = aggregate_to_patient_level(probs, patient_ids, labels=None)
    agg = agg.rename(columns={"p_class1": "p_crs_mean"}).drop(columns=["p_class0"])
    agg["GROUP"] = agg["ID"].map(df_group_lookup)
    return agg


def specificity_metrics(patient_df: pd.DataFrame, threshold: float) -> Dict:
    """
    Since these cohorts are assumed CRS-negative (see crs_status_check()),
    "accuracy" against an all-zero label vector is exactly the specificity
    /(1 - false-positive rate). Reused compute_metrics() for schema
    consistency with the rest of the repo (confusion-matrix-derived
    metrics, per repo convention), rather than hand-rolling a new metric
    shape.
    """
    if len(patient_df) == 0:
        return {"n_patients": 0}

    assumed_labels = [0] * len(patient_df)
    preds = (patient_df["p_crs_mean"] >= threshold).astype(int).tolist()
    probs = np.stack([
        1 - patient_df["p_crs_mean"].values,
        patient_df["p_crs_mean"].values,
    ], axis=1)

    m = compute_metrics(assumed_labels, preds, probs, num_classes=2,
                         split_name="specificity")
    m["specificity/n_patients"]        = int(len(patient_df))
    m["specificity/mean_p_crs"]        = float(patient_df["p_crs_mean"].mean())
    m["specificity/flagged_crs_rate"]  = float(np.mean(preds))
    m["specificity/threshold"]         = float(threshold)
    # accuracy under an all-negative assumed label IS specificity (TNR)
    m["specificity/specificity_pct"]   = m.get("specificity/accuracy", 0.0) * 100
    return m


def load_exp1_baseline(checkpoints_dir: _Path, job_name: str,
                        threshold: float = 0.5) -> Dict:
    """
    Pulls the FESS/Control patient-level test-set rates from Exp1's own
    results_summary.json for this backbone, so the Sept/Tonsill
    specificity numbers below are read next to what "flagged as CRS"
    already looks like on data this exact model WAS trained/tested on.
    Without this anchor, a flagging rate on Sept/Tonsill is hard to judge
    in isolation — e.g. is 30% "flagged CRS" high or low? That depends on
    how often this same model flags its own held-out Control patients.

    In addition to the pooled (FESS+Control) accuracy/F1, this also
    derives the CONTROL-ONLY false-positive rate directly from Exp1's
    stored per_patient predictions (label==0 is Control per
    Exp1CRSvsControl's labelling) — that's the number that's actually
    comparable to the Sept/Tonsill "flagged as CRS" rate below, since
    both are "rate of positive predictions on patients assumed CRS-
    negative". Pooled accuracy mixes in FESS true-positive performance
    and is a looser comparison.

    Returns {} (with a log message) if Exp1 hasn't been run with the
    patient-level metrics addition yet, or the file isn't there.
    """
    path = checkpoints_dir / "exp1_backbone_comparison" / job_name / "results_summary.json"
    if not path.exists():
        log.warning(f"  No Exp1 results_summary.json at {path} — "
                    f"baseline comparison unavailable for {job_name}.")
        return {}

    with open(path) as f:
        exp1_results = json.load(f)

    baseline = {
        k: v for k, v in exp1_results.items()
        if k.startswith("test/patient_level") and k != "test/patient_level/per_patient"
    }
    if not baseline:
        log.warning(
            f"  {path} has no 'test/patient_level/*' keys — this Exp1 run "
            f"predates the patient-level metrics addition. Re-run Exp1 "
            f"for {job_name} to get a like-for-like baseline, or fall back "
            f"to segment-level 'test/accuracy' for now."
        )
        return baseline

    per_patient = exp1_results.get("test/patient_level/per_patient", [])
    if per_patient:
        pp = pd.DataFrame(per_patient)
        control = pp[pp["label"] == 0]   # label 0 = Control per Exp1CRSvsControl
        if len(control) > 0:
            control_flagged = (control["p_class1"] >= threshold).mean()
            baseline["test/patient_level/control_false_positive_rate"] = float(control_flagged)
            baseline["test/patient_level/control_n_patients"] = int(len(control))

    return baseline


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════

def do_evaluate(filtered: pd.DataFrame, segment_dirs: Dict[str, str],
                 checkpoints_dir: _Path, output_dir: _Path,
                 device: torch.device, batch_size: int, threshold: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    group_lookup = dict(zip(filtered["ID"], filtered["GROUP"]))

    combined_summary = {}

    for job_name, backbone_type, mode, pretrained in BACKBONE_JOBS:
        log.info(f"\n{'─'*70}\n  Evaluating fixed Exp1 checkpoint: {job_name}\n{'─'*70}")

        ckpt_path = checkpoints_dir / "exp1_backbone_comparison" / job_name / "best_model.pt"
        if not ckpt_path.exists():
            log.warning(f"  Checkpoint not found, skipping: {ckpt_path}")
            continue

        seg_dir = segment_dirs.get(mode)
        if seg_dir is None or not _Path(seg_dir).exists():
            log.warning(
                f"  No clean segment dir for mode={mode} — run --step "
                f"regenerate first. Skipping {job_name}."
            )
            continue

        model = build_inference_model(backbone_type, mode, pretrained)
        load_checkpoint(str(ckpt_path), model)

        exp1_baseline = load_exp1_baseline(checkpoints_dir, job_name, threshold)

        job_result = {"exp1_baseline": exp1_baseline}
        for grp in TEST_GROUPS:
            grp_df = filtered[filtered["GROUP"] == grp]
            if grp_df.empty:
                continue
            dataset = SinusitisDataset(
                grp_df, segment_dir=seg_dir,
                label_fn=lambda row: 0,   # placeholder; not used for metrics
            )
            if len(dataset) == 0:
                log.warning(f"  0 segments found for {grp} in {seg_dir} — skipping.")
                continue

            probs, patient_ids = run_inference(model, dataset, device, batch_size)
            patient_df = aggregate_per_patient(probs, patient_ids, group_lookup)
            metrics = specificity_metrics(patient_df, threshold)

            job_result[grp] = {
                "metrics": metrics,
                "per_patient": patient_df.to_dict(orient="records"),
            }
            log.info(
                f"    {grp:<10} n_patients={metrics.get('specificity/n_patients')}  "
                f"mean P(CRS)={metrics.get('specificity/mean_p_crs', 0):.3f}  "
                f"flagged-CRS rate={metrics.get('specificity/flagged_crs_rate', 0):.3f}  "
                f"specificity={metrics.get('specificity/specificity_pct', 0):.1f}%"
            )

        if exp1_baseline:
            ctrl_fpr = exp1_baseline.get("test/patient_level/control_false_positive_rate")
            ctrl_n   = exp1_baseline.get("test/patient_level/control_n_patients")
            if ctrl_fpr is not None:
                log.info(
                    f"    (Exp1's own held-out CONTROL patients, n={ctrl_n}: "
                    f"flagged-as-CRS rate={ctrl_fpr:.3f} — THIS is the "
                    f"like-for-like comparison for the Sept/Tonsill "
                    f"flagged-CRS rates above. Pooled FESS+Control "
                    f"patient-level accuracy was "
                    f"{exp1_baseline.get('test/patient_level/accuracy', float('nan')):.3f}.)"
                )
            else:
                log.info(
                    f"    (Exp1 pooled patient-level accuracy="
                    f"{exp1_baseline.get('test/patient_level/accuracy', float('nan')):.3f} "
                    f"— per-patient predictions unavailable for a "
                    f"Control-only comparison; re-run Exp1 to get one.)"
                )

        combined_summary[job_name] = job_result

        with open(output_dir / f"{job_name}_specificity.json", "w") as f:
            json.dump(job_result, f, indent=2, cls=_NumpyEncoder)

    with open(output_dir / "cross_cohort_specificity_summary.json", "w") as f:
        json.dump(combined_summary, f, indent=2, cls=_NumpyEncoder)
    log.info(f"\n  Full results saved -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", choices=["identify", "regenerate", "evaluate", "all"],
                         default="all")
    parser.add_argument("--csv_path", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" /
                                     "Clinical" / "clinical_all_sessions.csv"))
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--diagnosis_col", type=str, default="diagnosis")
    parser.add_argument("--segment_dir", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" /
                                     "clean_audio_exp5_specificity"))
    parser.add_argument("--checkpoints_dir", type=str,
                         default=str(PROJECT_ROOT / "MSc_Sinusitis_results"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    project_root   = _Path(args.project_root)
    segment_dir    = _Path(args.segment_dir)
    checkpoints_dir = _Path(args.checkpoints_dir)
    output_dir     = _Path(args.output_dir) if args.output_dir else \
        checkpoints_dir / "exp5_cross_cohort_specificity"
    device         = _resolve_device(args.device)

    filtered = load_and_filter_csv(args.csv_path)
    crs_status_check(filtered, args.metadata_csv, args.diagnosis_col)

    modes_needed = sorted({m for _, _, m, _ in BACKBONE_JOBS})  # ["finetune","scratch"]

    if args.step in ("regenerate", "all"):
        regenerate_clean_segments(filtered, project_root, segment_dir, modes_needed)

    if args.step in ("evaluate", "all"):
        segment_dirs = {m: str(segment_dir / m) for m in modes_needed}
        do_evaluate(filtered, segment_dirs, checkpoints_dir, output_dir,
                    device, args.batch_size, args.threshold)


if __name__ == "__main__":
    main()
