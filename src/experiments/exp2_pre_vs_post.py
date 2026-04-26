from __future__ import annotations
 
import logging
from typing import Callable, Tuple
 
import pandas as pd
 
from src.experiments.base import BaseExperiment, ExperimentConfig
from src.pipeline.splits import (
    patient_level_split,
    generalisation_split,
    paired_patient_split,
)
 
log = logging.getLogger(__name__)
 
 
 
# ═════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Pre vs Post Surgery
# ═════════════════════════════════════════════════════════════════════════════
 
class Exp2PreVsPost(BaseExperiment):
    """
    Binary classification: Pre-op (Session 1) vs Post-op (Session 2 & 3).
 
    Clinical question
    ─────────────────
    Does voice change detectably after FESS surgery?
    Can we isolate the treatment effect from voice acoustics?
 
    Critical constraint
    ───────────────────
    Patient-level split ONLY. A patient's pre-op and post-op recordings
    must land in the SAME split. If session 1 is in train and session 2
    is in test, the model trivially learns voice identity → leakage.
 
    Data
    ────
    FESS Session 1        →  Class 0 (Pre-op)
    FESS Session 2 & 3    →  Class 1 (Post-op)
 
    Risks
    ─────
    • Natural recovery variation: not all patients recover equally.
      Post-op voice may still sound pathological for slow-recovery patients.
    • Session imbalance: 1 pre-op recording vs 2 post-op per patient.
      → Class weights applied to compensate.
    """
 
    @property
    def name(self) -> str:
        return "exp2_pre_vs_post"
 
    @property
    def num_classes(self) -> int:
        return 2
 
    @property
    def description(self) -> str:
        return (
            "Binary: Pre-op (FESS Session 1) vs Post-op (Sessions 2 & 3).\n"
            "Detects surgery-induced physiological change in speech."
        )
 
    def prepare_data(self):
        df = self.load_csv()
 
        # FESS only
        fess = df[df["GROUP"] == "FESS"].copy()
        fess["label"] = fess["session"].apply(
            lambda s: 0 if s == 1 else 1
        )
 
        label_fn: Callable = lambda row: int(row["label"])
 
        log.info("\n  [Exp2] Raw class distribution:")
        self.log_class_distribution(fess, label_fn, "all FESS data")
 
        # Patient-level split — all sessions of one patient stay together
        train_df, val_df, test_df = patient_level_split(
            fess,
            test_size=self.cfg.test_size,
            val_size=self.cfg.val_size,
            seed=self.cfg.seed,
            stratify_col="GROUP",
        )
 
        for name, split in [("train", train_df),
                             ("val",   val_df),
                             ("test",  test_df)]:
            self.log_class_distribution(split, label_fn, name)
 
        return train_df, val_df, test_df, label_fn
 
 