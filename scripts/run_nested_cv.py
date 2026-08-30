"""
scripts/run_nested_cv.py
─────────────────────────────────────────────────────────────────────────────
Nested, patient-grouped cross-validation for Exp1 (CRS vs Control,
Session 1 both arms).

Why nested CV, and why it's expensive here
─────────────────────────────────────────────
An OUTER loop (K folds) gives an unbiased performance estimate: for each
outer fold, everything else is used to pick a model (INNER loop), and
that model is scored ONLY on the outer fold it never saw — including
never having any influence on hyperparameter choice. This is what
prevents the optimistic bias of picking hyperparameters and estimating
performance from the same held-out data.

The patient-grouping constraint (every recording/segment from one
patient stays in the same fold, at BOTH the outer and inner level) is
non-negotiable here for the same reason it's non-negotiable in
src/pipeline/splits.py::patient_level_split() — session-to-session voice
identity leakage would otherwise inflate every number. See
src/pipeline/splits.py::patient_group_kfold(), which this script uses
for every fold at every level.

Cost, stated plainly: for the NEURAL (MLP) head, this means
outer_folds × inner_folds × len(hyperparameter_grid) inner training runs,
PLUS outer_folds final retrains, PER BACKBONE. With the defaults below
(5 outer, 3 inner, a 4-combination grid) that's 5×(3×4 + 1) = 65 full
training runs per backbone — × 6 backbones = 390 total. ALWAYS run
--dry_run first and look at the printed estimate before committing GPU
time; trim --backbones, the grid (--grid_json), or --outer_folds/
--inner_folds if that number is too large for your budget.

The SVM head is comparatively cheap: it reuses each outer fold's
already-trained neural backbone (no extra neural training) purely to
extract frozen embeddings once, then does its own inner-CV hyperparameter
search over C/kernel entirely at the embedding level (a handful of
sklearn fits, not neural training) — negligible additional cost on top
of the neural stage.

What this does NOT touch
──────────────────────────
Nothing here overwrites your existing Exp1 checkpoints
(exp1_backbone_comparison/<backbone>/best_model.pt) or their
results_summary.json — this is a completely separate analysis, writing
to its own nested_cv/ output directory. Your single train/val/test Exp1
results remain the primary trained models used elsewhere (Exp5, ablation,
statistical tests); nested CV here produces an additional, more robust
generalisation estimate to report alongside them, not a replacement.

Inner-loop training runs are written to a LOCAL scratch directory
(--scratch_dir, default /content/nested_cv_scratch), not Drive — Trainer
always writes a best_model.pt on every val-metric improvement regardless
of --save_every, so pointing throwaway inner runs at Drive would be slow
and fill your Drive with disposable checkpoints. Only the outer-fold
FINAL model per fold is written to --output_dir (Drive), if
--save_outer_models is passed.

Usage
──────
    # ALWAYS do this first
    python scripts/run_nested_cv.py --dry_run \\
        --backbones wav2vec2_finetune \\
        --outer_folds 5 --inner_folds 3

    # Then, once the estimate is acceptable:
    python scripts/run_nested_cv.py \\
        --csv_path     /content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv \\
        --segment_dir  /content/clean_audio_3s \\
        --output_dir   /content/drive/MyDrive/MSc_Sinusitis_results_examiner_feedback/nested_cv \\
        --backbones    wav2vec2_finetune \\
        --outer_folds  5 --inner_folds 3 \\
        --inner_epochs 15 --outer_epochs 30 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.experiments.base import ExperimentConfig
from src.experiments.all_experiments import Exp1CRSvsControl
from src.pipeline.splits import patient_group_kfold
from src.pipeline.dataloader import build_experiment_loaders
from src.training.imbalance import compute_class_weights
from src.training.patient_metrics import (
    patient_ids_from_samples, aggregate_to_patient_level,
    compute_patient_level_metrics,
)

BACKBONE_JOBS = [
    # (job_name, backbone_type, mode, pretrained_id)
    ("wav2vec2_scratch",  "wav2vec2", "scratch",  None),
    ("wav2vec2_finetune", "wav2vec2", "finetune", "facebook/wav2vec2-base-960h"),
    ("wavlm_scratch",     "wavlm",    "scratch",  None),
    ("wavlm_finetune",    "wavlm",    "finetune", "microsoft/wavlm-base"),
    ("xlsr_scratch",      "xlsr",     "scratch",  None),
    ("xlsr_finetune",     "xlsr",     "finetune", "facebook/wav2vec2-xls-r-300m"),
]

DEFAULT_NEURAL_GRID = [
    {"freeze_layers": 2, "learning_rate": 1e-5},
    {"freeze_layers": 2, "learning_rate": 5e-5},
    {"freeze_layers": 6, "learning_rate": 1e-5},
    {"freeze_layers": 6, "learning_rate": 5e-5},
]

DEFAULT_SVM_GRID = [
    {"C": 0.1, "kernel": "linear"},
    {"C": 1.0, "kernel": "linear"},
    {"C": 1.0, "kernel": "rbf"},
    {"C": 10.0, "kernel": "rbf"},
]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ═════════════════════════════════════════════════════════════════════════
# Neural (MLP) head — one training run, given fixed hyperparameters
# ═════════════════════════════════════════════════════════════════════════

def train_and_score_neural(
    hp: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
    segment_dir: str, project_root: _Path, label_fn, backbone_type: str,
    mode: str, pretrained: Optional[str], batch_size: int, num_epochs: int,
    device: torch.device, output_dir: _Path, seed: int,
) -> Tuple[Dict, float, "torch.nn.Module"]:
    """
    Trains ONE model with the given hyperparameters on train_df, scored
    against val_df (patient-level F1, falling back to segment-level if
    patient-ID recovery isn't possible for some reason). val_df doubles
    as both the "val" and "test" loader — Trainer.fit() needs both
    arguments, but there's no separate held-out set needed at this level;
    the true held-out set is the OUTER fold, one level up.

    Returns (raw results dict, patient-level macro-F1 score, trained
    Trainer.model — the caller may want the model itself, e.g. the
    OUTER-fold final run needs its trained backbone for the SVM stage).
    """
    from src.training.trainer import Trainer

    cfg = ExperimentConfig(
        project_root=str(project_root), segment_dir=segment_dir,
        csv_path="unused", output_dir=str(output_dir),
        mode=mode, backbone=backbone_type,
        pretrained=pretrained or "facebook/wav2vec2-base-960h",
        freeze_layers=hp.get("freeze_layers", 6),
        learning_rate=hp.get("learning_rate", 1e-4),
        batch_size=batch_size, num_epochs=num_epochs,
        imbalance_strategy="weights", seed=seed,
        save_every=num_epochs + 100,   # effectively disables periodic saves;
                                        # best_model.pt still saves on every
                                        # val improvement (Trainer behaviour,
                                        # not controlled by save_every)
        device=str(device),
    )

    class_weights = compute_class_weights(train_df, label_fn, num_classes=2)
    train_loader, val_loader, test_loader = build_experiment_loaders(
        train_df=train_df, val_df=val_df, test_df=val_df,
        segment_dir=segment_dir, label_fn=label_fn, batch_size=cfg.batch_size,
        imbalance_strategy=cfg.imbalance_strategy, class_weights=class_weights,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )

    trainer = Trainer(cfg=cfg, num_classes=2, class_weights=class_weights,
                       output_dir=str(output_dir))
    results = trainer.fit(train_loader=train_loader, val_loader=val_loader,
                           test_loader=test_loader)

    score = results.get("test/f1_macro", 0.0)   # segment-level fallback
    test_ds = test_loader.dataset
    if hasattr(test_ds, "samples") and "test/all_probs" in results:
        patient_ids = patient_ids_from_samples(test_ds.samples)
        probs  = np.asarray(results["test/all_probs"])
        labels = np.asarray(results["test/all_labels"])
        if len(patient_ids) == len(probs):
            pdf = aggregate_to_patient_level(probs, patient_ids, labels)
            pm  = compute_patient_level_metrics(pdf, num_classes=2, split_name="test")
            score = pm.get("test/patient_level/f1_macro", score)
            results.update(pm)

    return results, float(score), trainer.model


# ═════════════════════════════════════════════════════════════════════════
# SVM head — cheap inner CV at the embedding level, reusing a trained backbone
# ═════════════════════════════════════════════════════════════════════════

def svm_nested_fold(
    model, outer_train_df: pd.DataFrame, outer_test_df: pd.DataFrame,
    segment_dir: str, label_fn, device: torch.device, batch_size: int,
    svm_grid: List[Dict], inner_folds: int, seed: int,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Reuses `model`'s already-trained (this outer fold's) backbone purely
    as a frozen feature extractor — no additional neural training. Inner
    CV for (C, kernel) selection happens entirely on the extracted
    embeddings (sklearn fits only, no GPU forward passes beyond the one
    embedding-extraction pass per df), so this is cheap regardless of
    grid size or inner fold count.
    """
    from src.training.svm_classifier import extract_embeddings
    from src.pipeline.dataset import SinusitisDataset
    from src.pipeline.dataloader import collate_standard
    from torch.utils.data import DataLoader
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    def embed(df):
        ds = SinusitisDataset(df, segment_dir, label_fn)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_standard)
        emb, labels = extract_embeddings(model, loader, device)
        pids = patient_ids_from_samples(ds.samples)
        return np.asarray(emb), np.asarray(labels), pids

    train_emb, train_labels, train_pids = embed(outer_train_df)
    test_emb,  test_labels,  test_pids  = embed(outer_test_df)

    # Patient-grouped inner folds for SVM hyperparameter selection, driven
    # by patient_group_kfold — same leakage guarantee as everywhere else,
    # applied here to a synthetic per-segment dataframe since embeddings
    # are already extracted (no audio/DataFrame needed beyond ID+label).
    inner_df = pd.DataFrame({"ID": train_pids, "_label": train_labels})
    inner_fold_ids = patient_group_kfold(inner_df, n_splits=inner_folds,
                                          seed=seed, stratify_col="_label")

    best_score, best_hp = -1.0, svm_grid[0]
    for hp in svm_grid:
        fold_scores = []
        for inner_train_ids, inner_val_ids in inner_fold_ids:
            tr_mask = np.isin(train_pids, inner_train_ids)
            va_mask = np.isin(train_pids, inner_val_ids)
            if tr_mask.sum() == 0 or va_mask.sum() == 0:
                continue
            scaler = StandardScaler().fit(train_emb[tr_mask])
            svc = SVC(C=hp["C"], kernel=hp["kernel"], probability=True,
                      random_state=seed, class_weight="balanced")
            svc.fit(scaler.transform(train_emb[tr_mask]), train_labels[tr_mask])
            val_pred = svc.predict(scaler.transform(train_emb[va_mask]))
            fold_scores.append(f1_score(train_labels[va_mask], val_pred,
                                         average="macro", zero_division=0))
        mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
        if mean_score > best_score:
            best_score, best_hp = mean_score, hp

    # Refit on the FULL outer_train, evaluate on the untouched outer_test.
    scaler = StandardScaler().fit(train_emb)
    svc = SVC(C=best_hp["C"], kernel=best_hp["kernel"], probability=True,
              random_state=seed, class_weight="balanced")
    svc.fit(scaler.transform(train_emb), train_labels)
    test_probs = svc.predict_proba(scaler.transform(test_emb))

    patient_df = aggregate_to_patient_level(test_probs, test_pids, test_labels)
    patient_metrics = compute_patient_level_metrics(patient_df, num_classes=2, split_name="test")

    return best_hp, patient_metrics, patient_df


# ═════════════════════════════════════════════════════════════════════════
# Outer loop orchestration, per backbone
# ═════════════════════════════════════════════════════════════════════════

def two_way_patient_split(df: pd.DataFrame, val_size: float, seed: int,
                           stratify_col: str = "GROUP") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/val-only split at the patient level, stratified. Deliberately
    separate from patient_level_split() (which always carves out a third,
    "test" portion with a forced minimum of 1 patient via
    max(1, int(n*test_size)) — passing a near-zero test_size to that
    function to fake a 2-way split would silently strand at least one
    patient in a discarded split for every stratum, which is wasteful for
    the final outer-fold model's training set here). Same per-stratum
    shuffle-and-partition approach as patient_level_split(), just two-way.
    """
    rng = np.random.default_rng(seed)
    patient_groups = df.groupby("ID")[stratify_col].first().reset_index()
    train_ids, val_ids = [], []
    for group in patient_groups[stratify_col].unique():
        group_ids = patient_groups[patient_groups[stratify_col] == group]["ID"].to_numpy(dtype=object).copy()
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_val = max(1, int(n * val_size)) if n > 1 else 0
        val_ids.extend(group_ids[:n_val])
        train_ids.extend(group_ids[n_val:])
    return df[df["ID"].isin(train_ids)].copy(), df[df["ID"].isin(val_ids)].copy()


def config_fingerprint(neural_grid: List[Dict], svm_grid: List[Dict],
                        outer_folds: int, inner_folds: int, seed: int,
                        inner_epochs: int, outer_epochs: int) -> Dict:
    """
    Captures every setting that determines whether a checkpointed fold is
    actually reusable for THIS run, vs. a different run that happens to
    share the same --output_dir. Resume only checks file existence — it
    has no way to know whether --grid_json, --outer_folds, --inner_folds,
    or --seed changed since that checkpoint was written (e.g. across
    Colab sessions where the grid gets narrowed further to cut cost).
    Silently reusing a fold computed under different settings would mix
    incompatible folds into one aggregated estimate with no warning —
    this fingerprint is compared on every resume so that mismatch is
    caught and made loud instead.
    """
    return {
        "neural_grid": neural_grid, "svm_grid": svm_grid,
        "outer_folds": outer_folds, "inner_folds": inner_folds, "seed": seed,
        "inner_epochs": inner_epochs, "outer_epochs": outer_epochs,
    }


def run_nested_cv_for_backbone(
    job_name: str, backbone_type: str, mode: str, pretrained: Optional[str],
    filtered_df: pd.DataFrame, label_fn, segment_dir: str, project_root: _Path,
    scratch_dir: _Path, output_dir: _Path, outer_folds: int, inner_folds: int,
    neural_grid: List[Dict], svm_grid: List[Dict], batch_size: int,
    inner_epochs: int, outer_epochs: int, device: torch.device, seed: int,
    save_outer_models: bool, run_svm: bool, overwrite: bool = False,
) -> Dict:
    log.info(f"\n{'='*70}\n  NESTED CV — {job_name}\n{'='*70}")

    fingerprint = config_fingerprint(neural_grid, svm_grid, outer_folds,
                                      inner_folds, seed, inner_epochs, outer_epochs)

    # ── Resume: whole-backbone short-circuit ────────────────────────────────
    # Nested CV is the most expensive step in this pipeline — a multi-hour
    # Colab run WILL eventually hit a disconnect. Without this, re-running
    # the cell restarts every backbone from outer fold 0, discarding
    # everything already computed. --overwrite forces a clean rerun.
    final_summary_path = output_dir / f"{job_name}_nested_cv.json"
    if final_summary_path.exists() and not overwrite:
        with open(final_summary_path) as f:
            existing = json.load(f)
        if existing.get("_config_fingerprint") == fingerprint:
            log.info(f"  ↩  SKIP (already complete, same config): {final_summary_path}")
            log.info(f"     Pass --overwrite to force a full rerun.")
            return existing
        log.warning(
            f"  ⚠ {final_summary_path} exists but was computed under DIFFERENT "
            f"settings (grid/fold-count/seed/epochs changed since that run). "
            f"Treating as stale and recomputing this backbone from scratch — "
            f"reusing it would silently mix incompatible folds into one "
            f"estimate. Pass --overwrite to suppress this check entirely."
        )

    outer_fold_ids = patient_group_kfold(filtered_df, n_splits=outer_folds,
                                          seed=seed, stratify_col="GROUP")
    fold_results = []

    for fold_i, (outer_train_ids, outer_test_ids) in enumerate(outer_fold_ids):
        # ── Resume: per-outer-fold ───────────────────────────────────────────
        # Each outer fold is roughly total_time/outer_folds — checkpointing
        # here means a disconnect partway through fold 3 of 5 only loses
        # fold 3's in-progress work, not folds 0-2's completed results.
        fold_path = output_dir / f"{job_name}_fold{fold_i}.json"
        if fold_path.exists() and not overwrite:
            with open(fold_path) as f:
                cached_fold = json.load(f)
            if cached_fold.get("_config_fingerprint") == fingerprint:
                log.info(f"\n  ↩  SKIP outer fold {fold_i+1}/{outer_folds} "
                         f"(already complete, same config): {fold_path}")
                fold_results.append(cached_fold)
                continue
            log.warning(
                f"\n  ⚠ {fold_path} exists but was computed under DIFFERENT "
                f"settings — recomputing fold {fold_i+1} instead of silently "
                f"reusing it. Pass --overwrite to suppress this check."
            )

        t0 = time.time()
        log.info(f"\n{'─'*70}\n  {job_name} — OUTER FOLD {fold_i+1}/{outer_folds}\n{'─'*70}")


        outer_train_df = filtered_df[filtered_df["ID"].isin(outer_train_ids)]
        outer_test_df  = filtered_df[filtered_df["ID"].isin(outer_test_ids)]

        # ── Inner CV: neural hyperparameter selection on outer_train only ──
        inner_fold_ids = patient_group_kfold(outer_train_df, n_splits=inner_folds,
                                              seed=seed, stratify_col="GROUP")
        grid_scores = {}
        for combo_i, hp in enumerate(neural_grid):
            inner_scores = []
            for inner_i, (inner_train_ids, inner_val_ids) in enumerate(inner_fold_ids):
                # ── Resume: per-inner-run ────────────────────────────────────
                # The outer-fold-level checkpoint alone is too coarse for an
                # expensive backbone (e.g. XLS-R with AMP disabled — ~2 min/
                # epoch vs ~35s for the others): a single outer fold's inner
                # search is inner_folds x len(neural_grid) full training runs
                # (4 here), which can take 1.5-2+ hours before the outer-fold
                # checkpoint ever gets written. If the session doesn't survive
                # that long, EVERY restart redoes inner run 1 from scratch —
                # no progress ever persists, no matter how many sessions it
                # takes. Persisting each inner run's score the moment it's
                # computed closes that gap: a disconnect mid-search only
                # loses the one inner run in progress, not the whole fold's
                # search so far.
                inner_path = output_dir / f"{job_name}_fold{fold_i}_inner{inner_i}_combo{combo_i}.json"
                if inner_path.exists() and not overwrite:
                    with open(inner_path) as f:
                        cached_inner = json.load(f)
                    if cached_inner.get("_config_fingerprint") == fingerprint:
                        score = cached_inner["score"]
                        inner_scores.append(score)
                        log.info(f"    ↩ [inner {inner_i+1}/{inner_folds}] combo={hp} "
                                 f"SKIP (cached): patient-F1={score:.4f}")
                        continue
                    log.warning(
                        f"    ⚠ {inner_path} exists but was computed under DIFFERENT "
                        f"settings — recomputing this inner run instead of silently "
                        f"reusing it."
                    )

                inner_train_df = outer_train_df[outer_train_df["ID"].isin(inner_train_ids)]
                inner_val_df   = outer_train_df[outer_train_df["ID"].isin(inner_val_ids)]
                run_dir = scratch_dir / job_name / f"outer{fold_i}_inner{inner_i}_combo{combo_i}"
                _, score, _ = train_and_score_neural(
                    hp, inner_train_df, inner_val_df, segment_dir, project_root,
                    label_fn, backbone_type, mode, pretrained, batch_size,
                    inner_epochs, device, run_dir, seed,
                )
                inner_scores.append(score)
                shutil.rmtree(run_dir, ignore_errors=True)   # throwaway — keep local disk bounded

                # Persisted immediately — before the next inner run starts —
                # so a disconnect right after this line still preserves this
                # inner run's result. Only the score is saved (not the model:
                # inner runs are throwaway by design, only used to rank
                # hyperparameters), so this is a tiny, cheap write.
                with open(inner_path, "w") as f:
                    json.dump({"score": score, "_config_fingerprint": fingerprint}, f, indent=2)

                log.info(f"    [inner {inner_i+1}/{inner_folds}] combo={hp} "
                         f"patient-F1={score:.4f}")
            mean_score = float(np.mean(inner_scores))
            grid_scores[combo_i] = mean_score
            log.info(f"  combo {combo_i} {hp} → mean inner patient-F1={mean_score:.4f}")

        best_combo_i = max(grid_scores, key=grid_scores.get)
        best_hp = neural_grid[best_combo_i]
        log.info(f"\n  Best hyperparameters for outer fold {fold_i}: {best_hp} "
                 f"(mean inner patient-F1={grid_scores[best_combo_i]:.4f})")

        # ── Final model for this outer fold: full outer_train, best hp ─────
        # Carve a small validation slice from outer_train for early
        # stopping — this is NOT the outer_test fold, so no leakage.
        final_train_df, final_val_df = two_way_patient_split(
            outer_train_df, val_size=0.15, seed=seed, stratify_col="GROUP"
        )

        final_output_dir = (output_dir if save_outer_models else scratch_dir) / job_name / f"outer_fold_{fold_i}"
        outer_results, outer_score, outer_model = train_and_score_neural(
            best_hp, final_train_df, final_val_df, segment_dir, project_root,
            label_fn, backbone_type, mode, pretrained, batch_size,
            outer_epochs, device, final_output_dir, seed,
        )

        # Score the FINAL model on the untouched outer_test fold.
        from src.pipeline.dataset import SinusitisDataset
        from src.pipeline.dataloader import collate_standard
        from torch.utils.data import DataLoader
        outer_test_ds = SinusitisDataset(outer_test_df, segment_dir, label_fn)
        outer_test_loader = DataLoader(outer_test_ds, batch_size=batch_size,
                                        shuffle=False, collate_fn=collate_standard)
        outer_model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in outer_test_loader:
                iv = batch["input_values"].to(device)
                am = batch["attention_mask"].to(device)
                logits = outer_model(input_values=iv, attention_mask=am)
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        outer_test_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 2))
        outer_test_patient_ids = patient_ids_from_samples(outer_test_ds.samples)
        outer_test_seg_labels = np.array([lbl for _, lbl, _ in outer_test_ds.samples])
        outer_patient_df = aggregate_to_patient_level(outer_test_probs, outer_test_patient_ids, outer_test_seg_labels)
        outer_patient_metrics = compute_patient_level_metrics(outer_patient_df, num_classes=2, split_name="outer_test")

        log.info(f"\n  OUTER FOLD {fold_i} final score (on held-out outer_test, "
                 f"n={len(outer_patient_df)} patients): "
                 f"patient-F1={outer_patient_metrics.get('outer_test/patient_level/f1_macro', float('nan')):.4f}  "
                 f"accuracy={outer_patient_metrics.get('outer_test/patient_level/accuracy', float('nan')):.4f}")

        fold_record = {
            "fold": fold_i,
            "best_hp": best_hp,
            "inner_grid_scores": {str(neural_grid[i]): s for i, s in grid_scores.items()},
            "outer_test_metrics": outer_patient_metrics,
            "outer_test_per_patient": outer_patient_df.to_dict(orient="records"),
            "n_outer_train_patients": len(outer_train_ids),
            "n_outer_test_patients": len(outer_test_ids),
            "elapsed_seconds": time.time() - t0,
        }

        # ── SVM head for this fold, reusing the just-trained backbone ──────
        if run_svm:
            svm_hp, svm_metrics, svm_patient_df = svm_nested_fold(
                outer_model, final_train_df, outer_test_df, segment_dir, label_fn,
                device, batch_size, svm_grid, inner_folds, seed,
            )
            fold_record["svm_best_hp"] = svm_hp
            fold_record["svm_outer_test_metrics"] = svm_metrics
            fold_record["svm_outer_test_per_patient"] = svm_patient_df.to_dict(orient="records")
            log.info(f"  OUTER FOLD {fold_i} SVM: best_hp={svm_hp}  "
                     f"patient-F1={svm_metrics.get('test/patient_level/f1_macro', float('nan')):.4f}")

        fold_results.append(fold_record)

        # ── Persist this fold immediately — this IS the resume checkpoint.
        # Written before touching the next fold, so a disconnect right after
        # this line still preserves fold_i's completed work. Fingerprint
        # embedded so a future resume can tell whether this was computed
        # under the same settings currently in effect.
        fold_record["_config_fingerprint"] = fingerprint
        with open(fold_path, "w") as f:
            json.dump(fold_record, f, indent=2, cls=_NumpyEncoder)
        log.info(f"  Fold {fold_i} checkpointed -> {fold_path}")

        # Inner-run caches for this fold are superseded now that the whole
        # fold is checkpointed — clean them up so they don't accumulate
        # indefinitely across every backbone/fold/combo on Drive.
        for stale_cache in output_dir.glob(f"{job_name}_fold{fold_i}_inner*_combo*.json"):
            stale_cache.unlink(missing_ok=True)

        if not save_outer_models:
            shutil.rmtree(final_output_dir, ignore_errors=True)

    # ── Aggregate across outer folds — the actual nested CV estimate ───────
    neural_f1s = [f["outer_test_metrics"].get("outer_test/patient_level/f1_macro", np.nan) for f in fold_results]
    neural_accs = [f["outer_test_metrics"].get("outer_test/patient_level/accuracy", np.nan) for f in fold_results]
    summary = {
        "job_name": job_name, "outer_folds": outer_folds, "inner_folds": inner_folds,
        "_config_fingerprint": fingerprint,
        "neural_patient_f1_mean": float(np.nanmean(neural_f1s)),
        "neural_patient_f1_std":  float(np.nanstd(neural_f1s)),
        "neural_patient_acc_mean": float(np.nanmean(neural_accs)),
        "neural_patient_acc_std":  float(np.nanstd(neural_accs)),
        "folds": fold_results,
    }
    if run_svm:
        svm_f1s = [f.get("svm_outer_test_metrics", {}).get("test/patient_level/f1_macro", np.nan) for f in fold_results]
        summary["svm_patient_f1_mean"] = float(np.nanmean(svm_f1s))
        summary["svm_patient_f1_std"]  = float(np.nanstd(svm_f1s))

    log.info(f"\n{'='*70}\n  {job_name} NESTED CV RESULT: "
             f"patient-F1 = {summary['neural_patient_f1_mean']:.4f} ± {summary['neural_patient_f1_std']:.4f} "
             f"(n={outer_folds} outer folds)\n{'='*70}")

    return summary


# ═════════════════════════════════════════════════════════════════════════
# Cost estimation (--dry_run)
# ═════════════════════════════════════════════════════════════════════════

def print_dry_run_estimate(jobs, outer_folds, inner_folds, neural_grid, svm_grid,
                            est_minutes_per_inner_run, est_minutes_per_outer_run):
    n_backbones = len(jobs)
    inner_runs_per_fold = inner_folds * len(neural_grid)
    total_inner_runs = n_backbones * outer_folds * inner_runs_per_fold
    total_outer_runs = n_backbones * outer_folds

    print("═" * 70)
    print("  NESTED CV — DRY RUN COST ESTIMATE (no training will happen)")
    print("═" * 70)
    print(f"  Backbones:            {n_backbones}  ({[j[0] for j in jobs]})")
    print(f"  Outer folds:          {outer_folds}")
    print(f"  Inner folds:          {inner_folds}")
    print(f"  Neural grid size:     {len(neural_grid)}  {neural_grid}")
    print(f"  SVM grid size:        {len(svm_grid)}  (cheap — embedding-level only)")
    print()
    print(f"  Inner (throwaway) training runs: {n_backbones} × {outer_folds} × "
          f"{inner_folds} × {len(neural_grid)} = {total_inner_runs}")
    print(f"  Outer (final, kept) training runs: {n_backbones} × {outer_folds} = {total_outer_runs}")
    print(f"  TOTAL neural training runs: {total_inner_runs + total_outer_runs}")
    print()
    est_minutes = (total_inner_runs * est_minutes_per_inner_run +
                    total_outer_runs * est_minutes_per_outer_run)
    print(f"  Estimated time @ {est_minutes_per_inner_run} min/inner run, "
          f"{est_minutes_per_outer_run} min/outer run:")
    print(f"    {est_minutes:.0f} minutes  ≈  {est_minutes/60:.1f} hours")
    print()
    print("  These per-run minute estimates are a GUESS — pass --est_inner_min /")
    print("  --est_outer_min based on how long ONE of your actual training runs")
    print("  takes (check your Exp1 training log) for an accurate number.")
    print("═" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv_path", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" / "Clinical" / "clinical_all_sessions.csv"))
    parser.add_argument("--segment_dir", type=str,
                         default=str(PROJECT_ROOT / "Data" / "data_final" / "clean_audio"))
    parser.add_argument("--output_dir", type=str,
                         default=str(PROJECT_ROOT / "nested_cv_results"))
    parser.add_argument("--scratch_dir", type=str, default="/content/nested_cv_scratch")
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--backbones", type=str, default=None,
                         help="Comma-separated subset, e.g. wav2vec2_finetune. Default: all six.")
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--grid_json", type=str, default=None,
                         help='JSON list of neural hyperparameter combos, e.g. '
                              '\'[{"freeze_layers":2,"learning_rate":1e-5}]\'. Default: 4-combo grid.')
    parser.add_argument("--svm_grid_json", type=str, default=None,
                         help='JSON list of SVM combos, e.g. \'[{"C":1.0,"kernel":"rbf"}]\'.')
    parser.add_argument("--no_svm", action="store_true", help="Skip the SVM head entirely.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inner_epochs", type=int, default=15)
    parser.add_argument("--outer_epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_outer_models", action="store_true",
                         help="Persist each outer fold's final model to --output_dir (Drive). "
                              "Off by default — outer_folds × backbones checkpoints add up fast.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Ignore existing per-fold and per-backbone checkpoints and "
                              "rerun everything from scratch. Off by default — re-running this "
                              "cell after a disconnect resumes from the last completed outer "
                              "fold per backbone instead of restarting.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--est_inner_min", type=float, default=8.0)
    parser.add_argument("--est_outer_min", type=float, default=15.0)
    args = parser.parse_args()

    jobs = BACKBONE_JOBS
    if args.backbones:
        wanted = set(args.backbones.split(","))
        jobs = [j for j in BACKBONE_JOBS if j[0] in wanted]

    neural_grid = json.loads(args.grid_json) if args.grid_json else DEFAULT_NEURAL_GRID
    svm_grid    = json.loads(args.svm_grid_json) if args.svm_grid_json else DEFAULT_SVM_GRID

    if args.dry_run:
        print_dry_run_estimate(jobs, args.outer_folds, args.inner_folds,
                                neural_grid, svm_grid, args.est_inner_min, args.est_outer_min)
        return

    device = resolve_device(args.device)
    output_dir  = _Path(args.output_dir)
    scratch_dir = _Path(args.scratch_dir)
    project_root = _Path(args.project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # ── Persistent training log ──────────────────────────────────────────────
    # Attached to the ROOT logger (not just this module's), so it captures
    # EVERYTHING that gets logged during the run — including trainer.py's
    # per-epoch [TRAIN]/[VAL] lines, "New best val_f1" markers, per-audio-
    # type tables, and every fold/backbone boundary — not just this script's
    # own high-level messages. Written to --output_dir (Drive), so it
    # survives a disconnect the same way the JSON checkpoints do. Opened in
    # append mode: re-running across sessions keeps extending the SAME file
    # rather than overwriting the history from earlier sessions, so the full
    # multi-session narrative stays in one place. A banner is written at the
    # start of every invocation so it's easy to see where one session's
    # output ends and the next begins when reading it back.
    log_path = output_dir / "nested_cv_training_log.txt"
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    log.info(f"\n{'#'*70}")
    log.info(f"#  NEW SESSION — {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"#  backbones={[j[0] for j in jobs]}")
    log.info(f"#  outer_folds={args.outer_folds}  inner_folds={args.inner_folds}  "
             f"seed={args.seed}  overwrite={args.overwrite}")
    log.info(f"#  neural_grid={neural_grid}")
    log.info(f"{'#'*70}")
    log.info(f"  Full training log -> {log_path}")

    # Reuses Exp1CRSvsControl's OWN filtering — guarantees this matches
    # exactly what Exp1's single train/val/test run used, per
    # src/experiments/all_experiments.py::Exp1CRSvsControl.get_filtered_labeled_df().
    cfg_for_filtering = ExperimentConfig(project_root=str(project_root),
                                          segment_dir=args.segment_dir, csv_path=args.csv_path)
    filtered_df, label_fn = Exp1CRSvsControl(cfg_for_filtering).get_filtered_labeled_df()
    log.info(f"\nFiltered Exp1 population: {len(filtered_df)} rows, "
             f"{filtered_df['ID'].nunique()} patients")

    all_summaries = {}
    for job_name, backbone_type, mode, pretrained in jobs:
        summary = run_nested_cv_for_backbone(
            job_name, backbone_type, mode, pretrained, filtered_df, label_fn,
            args.segment_dir, project_root, scratch_dir, output_dir,
            args.outer_folds, args.inner_folds, neural_grid, svm_grid,
            args.batch_size, args.inner_epochs, args.outer_epochs, device,
            args.seed, args.save_outer_models, not args.no_svm, args.overwrite,
        )
        all_summaries[job_name] = summary

        with open(output_dir / f"{job_name}_nested_cv.json", "w") as f:
            json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    with open(output_dir / "nested_cv_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, cls=_NumpyEncoder)

    print("\n" + "=" * 70)
    print("  NESTED CV — FINAL SUMMARY (unbiased, patient-grouped estimate)")
    print("=" * 70)
    for job_name, s in all_summaries.items():
        line = (f"  {job_name:<20} neural patient-F1 = "
                f"{s['neural_patient_f1_mean']:.4f} ± {s['neural_patient_f1_std']:.4f}")
        if "svm_patient_f1_mean" in s:
            line += f"   |   SVM patient-F1 = {s['svm_patient_f1_mean']:.4f} ± {s['svm_patient_f1_std']:.4f}"
        print(line)
    print(f"\nFull results → {output_dir}")


if __name__ == "__main__":
    main()
