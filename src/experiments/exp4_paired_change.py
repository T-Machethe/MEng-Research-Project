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
# Experiment 4 — Within-Patient Change Detection (Paired)
# ═════════════════════════════════════════════════════════════════════════════
 
class Exp4PairedChange(BaseExperiment):
    """
    Binary classification on PAIRED inputs from the same patient.
 
    Clinical question
    ─────────────────
    Can the model detect acoustic change between two recordings from the
    SAME patient, eliminating voice identity as a confound entirely?
 
    Input format
    ────────────
    Each sample is a PAIR (pre_segment, post_segment) from one patient.
    Label = 1 (change) for session1 vs session2/3 pairs.
    Label = 0 (no change) for session1 vs session1 pairs (negatives).
 
    Why this matters scientifically
    ────────────────────────────────
    In Experiments 1–3 the model could learn "this voice sounds like
    patient X who has CRS" rather than "this voice has CRS-like features".
    Pairing recordings from the same speaker forces the model to learn
    DELTA features — what changed in the voice — which is the true
    clinical signal of interest.
 
    Risks
    ─────
    • Smaller effective dataset (pairing reduces sample count).
    • Negative pairs (same session) may be too easy if recordings are
      very similar, giving inflated accuracy.
    """
 
    @property
    def name(self) -> str:
        return "exp4_paired_change"
 
    @property
    def num_classes(self) -> int:
        return 2
 
    @property
    def description(self) -> str:
        return (
            "Within-patient paired change detection.\n"
            "Eliminates identity bias by pairing pre/post recordings "
            "from the same speaker."
        )
 
    def prepare_data(self):
        df = self.load_csv()
        fess = df[df["GROUP"] == "FESS"].copy()
 
        # For paired experiments the label_fn is not used in the standard
        # way — PairedDataset handles labelling internally.
        # We return a dummy label_fn for the base class interface.
        label_fn: Callable = lambda row: 0   # placeholder
 
        train_df, val_df, test_df = paired_patient_split(
            fess,
            test_size=self.cfg.test_size,
            val_size=self.cfg.val_size,
            seed=self.cfg.seed,
        )
 
        log.info("\n  [Exp4] Paired dataset will be built from segment files.")
        log.info(f"    Train patients: {train_df['ID'].nunique()}")
        log.info(f"    Val patients  : {val_df['ID'].nunique()}")
        log.info(f"    Test patients : {test_df['ID'].nunique()}")
 
        return train_df, val_df, test_df, label_fn
 
    def prepare(self):
        """
        Override prepare() to use PairedDataset + paired DataLoaders.
        """
        from src.pipeline.dataloader import build_experiment_loaders
        from src.training.imbalance import compute_class_weights
 
        train_df, val_df, test_df, label_fn = self.prepare_data()
 
        # Class weights for paired task (roughly balanced by construction)
        self.class_weights = None   # PairedDataset balances via neg_ratio
 
        (self.train_loader,
         self.val_loader,
         self.test_loader) = build_experiment_loaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            segment_dir=str(self.segment_dir),
            label_fn=label_fn,
            batch_size=self.cfg.batch_size,
            imbalance_strategy="none",   # paired dataset self-balances
            class_weights=None,
            num_workers=self.cfg.num_workers,
            seed=self.cfg.seed,
            paired=True,               # ← use PairedDataset
        )
 