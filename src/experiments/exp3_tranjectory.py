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
# Experiment 3 — Recovery Trajectory (Multi-Class)
# ═════════════════════════════════════════════════════════════════════════════
 
class Exp3Trajectory(BaseExperiment):
    """
    3-class classification: Pre / Early Post / Late Post surgery.
 
    Clinical question
    ─────────────────
    Can voice acoustics track the temporal progression of recovery
    from CRS through early and late post-surgical stages?
 
    Data
    ────
    FESS Session 1  →  Class 0 (Pre-op / active CRS)
    FESS Session 2  →  Class 1 (Early recovery, ~weeks post-surgery)
    FESS Session 3  →  Class 2 (Late recovery, ~months post-surgery)
 
    Risks
    ─────
    • Label noise: recovery rate varies per patient. A patient in Session 2
      may sound identical to Session 1 if recovery is slow.
    • Class overlap: the acoustic boundary between classes is likely gradual,
      not discrete.
    • Class imbalance: equal sessions per patient but fewer late-recovery
      recordings if some patients dropped out.
    """
 
    @property
    def name(self) -> str:
        return "exp3_trajectory"
 
    @property
    def num_classes(self) -> int:
        return 3
 
    @property
    def description(self) -> str:
        return (
            "3-class: Pre-op (0) / Early Post (1) / Late Post (2).\n"
            "Models temporal recovery trajectory from voice acoustics."
        )
 
    def prepare_data(self):
        df = self.load_csv()
 
        fess = df[df["GROUP"] == "FESS"].copy()
 
        session_to_label = {1: 0, 2: 1, 3: 2}
        fess["label"] = fess["session"].map(session_to_label)
        fess = fess.dropna(subset=["label"])
        fess["label"] = fess["label"].astype(int)
 
        label_fn: Callable = lambda row: int(row["label"])
 
        log.info("\n  [Exp3] Raw class distribution:")
        self.log_class_distribution(fess, label_fn, "all FESS data")
 
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
 
 