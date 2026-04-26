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
# Experiment 1 — CRS vs Control (Baseline Clinical Classifier)
# ═════════════════════════════════════════════════════════════════════════════
 
class Exp1CRSvsControl(BaseExperiment):
    """
    Binary classification: CRS (pre-surgery FESS) vs Healthy (Control).
 
    Clinical question
    ─────────────────
    Does voice contain a detectable disease signal that distinguishes
    Chronic Rhinosinusitis from a healthy larynx?
 
    Data
    ────
    FESS Session 1  →  Class 1 (CRS, confirmed pre-surgery)
    Control group   →  Class 0 (Healthy baseline)
 
    Risks and mitigations
    ─────────────────────
    • Class imbalance: FESS cohort may outnumber controls.
      → Handled via class_weights (logged before and after).
    • Identity bias: same patient voice across experiments.
      → Patient-level split with GROUP-stratification.
    • Confound: age/gender differences between groups.
      → Documented in interpretation notes; not corrected here.
    """
 
    @property
    def name(self) -> str:
        return "exp1_crs_vs_control"
 
    @property
    def num_classes(self) -> int:
        return 2
 
    @property
    def description(self) -> str:
        return (
            "Binary classification: CRS (FESS Session 1) vs Healthy (Control).\n"
            "Establishes whether voice contains a detectable sinusitis signal."
        )
 
    def prepare_data(self):
        df = self.load_csv()
 
        # ── Filter to relevant groups and sessions ─────────────────────────
        fess_s1 = df[(df["GROUP"] == "FESS") & (df["session"] == 1)].copy()
        control = df[df["GROUP"] == "Contr"].copy()
 
        fess_s1["label"] = 1   # CRS
        control["label"] = 0   # Healthy
 
        combined = pd.concat([fess_s1, control], ignore_index=True)
        combined = combined.dropna(subset=["label"])
 
        label_fn: Callable = lambda row: int(row["label"])
 
        # ── Log raw class distribution ─────────────────────────────────────
        log.info("\n  [Exp1] Raw class distribution (before split):")
        self.log_class_distribution(combined, label_fn, "all data")
 
        # ── Patient-level split stratified by GROUP ─────────────────────────
        train_df, val_df, test_df = patient_level_split(
            combined,
            test_size=self.cfg.test_size,
            val_size=self.cfg.val_size,
            seed=self.cfg.seed,
            stratify_col="GROUP",
        )
 
        # Log per-split class distribution to detect imbalance per split
        for name, split in [("train", train_df),
                             ("val",   val_df),
                             ("test",  test_df)]:
            self.log_class_distribution(split, label_fn, name)
 
        return train_df, val_df, test_df, label_fn
 