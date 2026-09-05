"""
─────────────────────────────────────────────────────────────────────────────
Abstract base class that every experiment inherits from.

Enforces a consistent interface:
    experiment.prepare()   → build train/val/test DataLoaders
    experiment.run()       → train + evaluate, return results dict
    experiment.report()    → print + save a human-readable summary

Design principles
─────────────────
• Patient-level splitting is MANDATORY — no patient leaks across splits.
• Class distribution is logged before and after any balancing step.
• All randomness is seeded from a single config value.
• Results are saved to a per-experiment subdirectory under output_dir.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    project_root:  str = "."
    segment_dir:   str = "./Data/data_final/clean_audio"
    csv_path:      str = "./Data/data_final/Clinical/clinical_all_sessions.csv"
    output_dir:    str = "./results"

    # ── Model ──────────────────────────────────────────────────────────────
    mode:          str  = "finetune"          # "scratch" | "finetune"
    pretrained:    str  = "facebook/wav2vec2-base-960h"
    backbone:      str  = "wav2vec2"  # "wav2vec2" | "wavlm"
    freeze_layers: int  = 6
    freeze_encoder: bool = True

    # ── Training ───────────────────────────────────────────────────────────
    batch_size:          int   = 16
    num_epochs:          int   = 30
    learning_rate:       float = 1e-4
    warmup_steps:        int   = 200
    weight_decay:        float = 1e-2
    label_smoothing:     float = 0.1    # prevents class collapse in finetune
    layerwise_lr_decay:  float = 0.8    # lower transformer layers get smaller LR
    use_focal_loss:       bool  = True    # use Focal Loss instead of CrossEntropy
    focal_gamma:          float = 2.0     # Focal Loss focusing parameter
    use_svm:              bool  = False   # run SVM on finetune embeddings
    svm_C:               float = 1.0    # SVM regularisation
    svm_kernel:          str   = "rbf"  # 'rbf' or 'linear'
    max_grad_norm:       float = 1.0
    early_stop_patience:  int   = 5
    early_stop_metric:    str   = "val_f1"   # "val_f1" (recommended) or "val_loss"
    head_warmup_epochs:   int   = 1           # finetune: train classifier head only for N epochs first

    # ── Split ──────────────────────────────────────────────────────────────
    test_size:     float = 0.20
    val_size:      float = 0.10
    seed:          int   = 42

    # ── Class imbalance ────────────────────────────────────────────────────
    # "none"     → no correction
    # "weights"  → pass class_weight to loss function
    # "oversample" → oversample minority class in DataLoader
    imbalance_strategy: str = "weights"

    # ── Logging ────────────────────────────────────────────────────────────
    log_every:   int = 20
    save_every:  int = 5
    keep_last_n: int = 2
    num_workers: int = 2

    # ── Device (auto-detected if left as "auto") ───────────────────────────
    device: str = field(default_factory=lambda: "auto")


class BaseExperiment(ABC):
    """
    All five experiments inherit from this class.

    Subclasses must implement:
        name         : str property  — used for output folder naming
        num_classes  : int property  — 2 for binary, 3 for trajectory
        prepare_data() → (train_df, val_df, test_df, label_fn)
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root)
        self.segment_dir  = Path(cfg.segment_dir)
        self.csv_path     = Path(cfg.csv_path)
        self.output_dir   = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Populated by prepare()
        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None
        self.class_weights = None
        self.results: Dict = {}

    # ── Abstract interface ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'exp1_crs_vs_control'."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-paragraph description of the experiment."""

    @abstractmethod
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame,
                                    pd.DataFrame, callable]:
        """
        Return (train_df, val_df, test_df, label_fn).

        label_fn : callable(row) -> int
            Maps a CSV row to its integer class label.

        Subclasses must:
          1. Filter the full CSV to the relevant rows.
          2. Apply patient-level splitting.
          3. Log class distribution before and after any balancing.
        """

    # ── Shared orchestration ───────────────────────────────────────────────

    def prepare(self):
        """Build DataLoaders. Called before run()."""
        from src.pipeline.dataloader import build_experiment_loaders
        from src.training.imbalance import compute_class_weights

        log.info(f"\n{'═'*60}")
        log.info(f"  {self.name}")
        log.info(f"{'═'*60}")
        log.info(self.description)

        train_df, val_df, test_df, label_fn = self.prepare_data()

        # Log split sizes
        log.info(f"\n  Split sizes:")
        log.info(f"    Train : {len(train_df)} rows  "
                 f"({train_df['ID'].nunique()} patients)")
        log.info(f"    Val   : {len(val_df)} rows  "
                 f"({val_df['ID'].nunique()} patients)")
        log.info(f"    Test  : {len(test_df)} rows  "
                 f"({test_df['ID'].nunique()} patients)")

        # Compute class weights from training set only
        self.class_weights = compute_class_weights(
            train_df, label_fn, self.num_classes
        )

        (self.train_loader,
         self.val_loader,
         self.test_loader) = build_experiment_loaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            segment_dir=str(self.segment_dir),
            label_fn=label_fn,
            batch_size=self.cfg.batch_size,
            imbalance_strategy=self.cfg.imbalance_strategy,
            class_weights=self.class_weights,
            num_workers=self.cfg.num_workers,
            seed=self.cfg.seed,
        )

    def run(self) -> Dict:
        """Train model and evaluate on test set. Returns results dict."""
        from src.training.trainer import Trainer

        assert self.train_loader is not None, \
            "Call prepare() before run()."

        trainer = Trainer(
            cfg=self.cfg,
            num_classes=self.num_classes,
            class_weights=self.class_weights,
            output_dir=str(self.output_dir),
        )

        self.results = trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
        )

        self._add_patient_level_metrics()

        return self.results

    def _add_patient_level_metrics(self):
        """
        Examiner feedback: segment-pooled metrics alone are misleading —
        patients with more segments dominate the pooled number, and
        segments from one patient are not independent. Adds a PATIENT-
        level view alongside the existing segment-level one, computed
        from predictions Trainer already produced (no re-inference
        needed here — see scripts/run_patient_level_aggregation.py for
        why VAL specifically needed a trainer.py fix first: val/all_probs
        used to reflect whatever epoch early stopping happened to land
        on, not necessarily the best checkpoint; test/all_probs was
        already correctly reloaded-checkpoint-aligned).

        Runs for BOTH val and test — same logic, same loader/dataset
        pairing per split — plus the SVM head for both, when present, and
        a patient-level, per-audio-type breakdown for both. Silently
        skipped (with a log message) for dataset types that don't expose
        `.samples` in the "ID{n}_ses{s}_{col}_seg{i}.pt" filename-
        parseable form — currently just PairedDataset (Exp4), out of
        scope for this addition.
        """
        from src.training.patient_metrics import (
            patient_ids_from_samples, audio_types_from_samples,
            aggregate_to_patient_level, compute_patient_level_metrics,
            aggregate_to_patient_audiotype_level, compute_patient_audiotype_metrics,
        )

        for split_name, loader in (("test", self.test_loader), ("val", self.val_loader)):
            ds = getattr(loader, "dataset", None)
            has_probs = f"{split_name}/all_probs" in self.results and f"{split_name}/all_labels" in self.results

            if ds is None or not hasattr(ds, "samples") or not has_probs:
                log.info(
                    f"  [patient-level][{split_name}] Skipped — dataset type or "
                    f"stored predictions don't support filename-based patient-ID "
                    f"recovery (expected for Exp4/PairedDataset)."
                )
                continue

            probs  = np.asarray(self.results[f"{split_name}/all_probs"])
            labels = np.asarray(self.results[f"{split_name}/all_labels"])
            patient_ids = patient_ids_from_samples(ds.samples)
            audio_types = audio_types_from_samples(ds.samples)

            assert len(patient_ids) == len(probs) == len(labels), (
                f"[patient-level][{split_name}] length mismatch: "
                f"{len(patient_ids)} patient_ids vs {len(probs)} probs vs "
                f"{len(labels)} labels — loader must be shuffle=False "
                f"and iterated exactly once for this alignment to hold."
            )

            patient_df = aggregate_to_patient_level(probs, patient_ids, labels, recording_ids=audio_types)
            patient_metrics = compute_patient_level_metrics(
                patient_df, self.num_classes, split_name=split_name
            )

            log.info(f"\n  [patient-level][{split_name}] {len(patient_df)} patients "
                      f"(vs {len(probs)} segments)")
            for k in (f"{split_name}/patient_level/accuracy", f"{split_name}/patient_level/f1_macro"):
                if k in patient_metrics:
                    log.info(f"    {k:<35} {patient_metrics[k]:.4f}")

            self.results.update(patient_metrics)
            self.results[f"{split_name}/patient_level/per_patient"] = patient_df.to_dict(orient="records")

            # ── Patient-level, per-audio-type breakdown ─────────────────────
            at_df = aggregate_to_patient_audiotype_level(probs, patient_ids, audio_types, labels)
            at_metrics = compute_patient_audiotype_metrics(at_df, self.num_classes, split_name=split_name)
            self.results[f"{split_name}/patient_level/by_audio_type"] = at_df.to_dict(orient="records")
            self.results[f"{split_name}/patient_level/by_audio_type_metrics"] = at_metrics.to_dict(orient="records")

            # ── Same treatment for the SVM head, when present ───────────────
            # SVM evaluates the exact same loader (same order, same
            # patient_ids/audio_types computed above) — see
            # src/training/svm_classifier.py::train_svm(), called with the
            # shared train/val/test_loader from this same run(). No
            # separate ID recovery needed.
            svm_results = self.results.get("svm")
            if svm_results and f"{split_name}/all_probs" in svm_results and f"{split_name}/all_labels" in svm_results:
                svm_probs  = np.asarray(svm_results[f"{split_name}/all_probs"])
                svm_labels = np.asarray(svm_results[f"{split_name}/all_labels"])
                if len(svm_probs) == len(patient_ids):
                    svm_patient_df = aggregate_to_patient_level(svm_probs, patient_ids, svm_labels, recording_ids=audio_types)
                    svm_patient_metrics = compute_patient_level_metrics(
                        svm_patient_df, self.num_classes, split_name=split_name
                    )
                    log.info(f"\n  [patient-level][SVM][{split_name}] {len(svm_patient_df)} patients")
                    for k in (f"{split_name}/patient_level/accuracy", f"{split_name}/patient_level/f1_macro"):
                        if k in svm_patient_metrics:
                            log.info(f"    {k:<35} {svm_patient_metrics[k]:.4f}")
                    svm_results.update(svm_patient_metrics)
                    svm_results[f"{split_name}/patient_level/per_patient"] = svm_patient_df.to_dict(orient="records")

                    svm_at_df = aggregate_to_patient_audiotype_level(svm_probs, patient_ids, audio_types, svm_labels)
                    svm_at_metrics = compute_patient_audiotype_metrics(svm_at_df, self.num_classes, split_name=split_name)
                    svm_results[f"{split_name}/patient_level/by_audio_type"] = svm_at_df.to_dict(orient="records")
                    svm_results[f"{split_name}/patient_level/by_audio_type_metrics"] = svm_at_metrics.to_dict(orient="records")
                else:
                    log.warning(
                        f"  [patient-level][SVM][{split_name}] Skipped — {len(svm_probs)} SVM "
                        f"predictions vs {len(patient_ids)} patient_ids "
                        f"(unexpected length mismatch)."
                    )

    def report(self):
        """Save and print a human-readable results summary."""
        if not self.results:
            log.warning("No results to report. Run run() first.")
            return
 
        class _NumpyEncoder(json.JSONEncoder):
            """Converts numpy arrays/scalars to JSON-serialisable types."""
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return super().default(obj)
 
        report_path = self.output_dir / "results_summary.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, cls=_NumpyEncoder)
 
        log.info(f"\n{'─'*60}")
        log.info(f"  RESULTS — {self.name}")
        log.info(f"{'─'*60}")
        for k, v in self.results.items():
            if isinstance(v, float):
                log.info(f"  {k:<30} {v:.4f}")
            else:
                log.info(f"  {k:<30} {v}")
        log.info(f"\n  Full results saved -> {report_path}")
 
    # ── Shared utilities ───────────────────────────────────────────────────

    def load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df["GROUP"] = df["GROUP"].str.strip()
        df["ID"]    = df["ID"].astype(str).str.strip()
        return df

    def log_class_distribution(self, df: pd.DataFrame,
                                label_fn: callable,
                                split_name: str):
        """Log how many samples per class are in a given split."""
        labels = df.apply(label_fn, axis=1)
        counts = labels.value_counts().sort_index()
        log.info(f"\n  Class distribution [{split_name}]:")
        for cls, cnt in counts.items():
            log.info(f"    Class {cls} : {cnt} rows  "
                     f"({100*cnt/len(df):.1f}%)")
        # Imbalance ratio
        if len(counts) >= 2:
            ratio = counts.max() / counts.min()
            log.info(f"    Imbalance ratio : {ratio:.2f}:1")