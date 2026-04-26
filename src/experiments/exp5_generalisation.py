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
# Experiment 5 — Generalisation Test (Out-of-Distribution)
# ═════════════════════════════════════════════════════════════════════════════
 
class Exp5Generalisation(BaseExperiment):
    """
    Out-of-distribution evaluation: Train on FESS, Test on other groups.
 
    Clinical question
    ─────────────────
    Does a model trained to detect CRS/surgery effects in FESS patients
    generalise to other clinical populations (septoplasty, tonsillitis)?
 
    This is the most honest test of real-world deployment readiness.
    A model that only works on the training cohort is not clinically useful.
 
    Data
    ────
    Train / Val : FESS patients (sessions 1, 2, 3)
    Test        : Sept (septoplasty) and Tonsill (tonsillitis) patients
 
    Label strategy
    ──────────────
    The test labels follow the same session-based mapping as Exp 2:
    Session 1 → 0 (pre-intervention), Session 2/3 → 1 (post-intervention).
    This tests whether the pre/post voice change signal is surgery-specific
    or reflects a more general intervention effect.
 
    Risks
    ─────
    • Different pathologies: septoplasty and tonsillitis affect different
      anatomical structures. Low generalisation is an expected and
      scientifically informative finding.
    • Different recording protocols across groups.
    """
 
    @property
    def name(self) -> str:
        return "exp5_generalisation"
 
    @property
    def num_classes(self) -> int:
        return 2
 
    @property
    def description(self) -> str:
        return (
            "OOD generalisation: Train on FESS, Test on Sept + Tonsill.\n"
            "Assesses deployment readiness across clinical populations."
        )
 
    def prepare_data(self):
        df = self.load_csv()
 
        # Session-based label (same mapping as Exp 2)
        label_fn: Callable = lambda row: (
            0 if row["session"] == 1 else 1
        )
 
        train_df, val_df, test_df = generalisation_split(
            df,
            train_groups=["FESS"],
            test_groups=["Sept", "Tonsill"],
            val_size=self.cfg.val_size,
            seed=self.cfg.seed,
        )
 
        for name, split in [("train (FESS)", train_df),
                             ("val (FESS)",   val_df),
                             ("test (Sept+Tonsill)", test_df)]:
            self.log_class_distribution(split, label_fn, name)
 
        return train_df, val_df, test_df, label_fn