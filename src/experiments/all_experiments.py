"""
src/experiments/exp1_crs_vs_control.py  — Experiment 1
src/experiments/exp2_pre_vs_post.py     — Experiment 2
src/experiments/exp3_trajectory.py      — Experiment 3
src/experiments/exp4_paired_change.py   — Experiment 4
src/experiments/exp5_generalisation.py  — Experiment 5
─────────────────────────────────────────────────────────────────────────────
All five experiment classes in one file for delivery.
In production split into separate files as named above.
"""

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

    def _patch_cfg_for_multiclass(self):
        """
        Override training settings for the 3-class task.

        Class 1 (early post-op) sits acoustically between pre-op and
        late recovery, making it the hardest class to learn. Standard
        focal gamma=2 is insufficient — gamma=3 more aggressively
        forces the model to attend to ambiguous intermediate samples.
        """
        if getattr(self.cfg, "focal_gamma", 2.0) <= 2.0:
            self.cfg.focal_gamma = 3.0
        # Ensure class weights are always used alongside oversampling
        # for the 3-class task (belt-and-suspenders for the minority class)
        if getattr(self.cfg, "imbalance_strategy", "oversample") == "oversample":
            self.cfg.imbalance_strategy = "both"

    def prepare_data(self):
        self._patch_cfg_for_multiclass()
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