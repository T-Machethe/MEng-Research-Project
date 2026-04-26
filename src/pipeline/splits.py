"""
Patient-level splitting utilities.

Critical design constraint
──────────────────────────
Every split function here guarantees that NO patient (identified by "ID")
appears in more than one of: train / val / test.

This is non-negotiable for longitudinal clinical data because:
  • The same patient has recordings across sessions 1, 2, 3.
  • If patient X appears in both train and test, the model can memorise
    voice identity rather than learning pathology features.
  • This would produce misleadingly high accuracy that collapses at
    deployment on unseen patients.

Session ordering is preserved — sessions are NEVER shuffled within a patient.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def patient_level_split(
    df: pd.DataFrame,
    test_size:  float = 0.20,
    val_size:   float = 0.10,
    seed:       int   = 42,
    stratify_col: str = "GROUP",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe at the patient level with optional stratification.

    Stratification ensures each split has a proportional representation
    of clinical groups (FESS / Contr / Sept / Tonsill), preventing a
    scenario where all FESS patients land in test by chance.

    Parameters
    ----------
    df            : full filtered dataframe (one row per recording session).
    test_size     : fraction of patients held out for test.
    val_size      : fraction of patients held out for validation.
    seed          : random seed for reproducibility.
    stratify_col  : column used for stratified sampling (default: GROUP).

    Returns
    -------
    train_df, val_df, test_df
    """
    rng = np.random.default_rng(seed)

    # Get unique patients with their group for stratification
    patient_groups = (
        df.groupby("ID")[stratify_col]
        .first()
        .reset_index()
    )

    groups = patient_groups[stratify_col].unique()
    train_ids, val_ids, test_ids = [], [], []

    for group in groups:
        group_patients = patient_groups[
            patient_groups[stratify_col] == group
        ]["ID"].values.copy()

        rng.shuffle(group_patients)

        n       = len(group_patients)
        n_test  = max(1, int(n * test_size))
        n_val   = max(1, int(n * val_size))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Too few patients in this group — put at least 1 in train
            log.warning(
                f"Group '{group}' has only {n} patients. "
                f"Adjusting split to 1 train / 1 val / rest test."
            )
            train_ids.extend(group_patients[:1])
            val_ids.extend(group_patients[1:2])
            test_ids.extend(group_patients[2:])
        else:
            test_ids.extend(group_patients[:n_test])
            val_ids.extend(group_patients[n_test:n_test + n_val])
            train_ids.extend(group_patients[n_test + n_val:])

    train_df = df[df["ID"].isin(train_ids)].copy()
    val_df   = df[df["ID"].isin(val_ids)].copy()
    test_df  = df[df["ID"].isin(test_ids)].copy()

    _verify_no_leakage(train_ids, val_ids, test_ids)
    _log_split_summary(train_df, val_df, test_df, stratify_col)

    return train_df, val_df, test_df


def paired_patient_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size:  float = 0.10,
    seed:      int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split for Experiment 4 (paired within-patient change detection).

    Each patient contributes PAIRS of rows (session 1 vs session 2/3).
    The split is done at the patient level so all pairs from one patient
    land in the same split — preventing identity leakage.

    Returns the same patient-level split but the pairing is done later
    in the PairedDataset class.
    """
    return patient_level_split(
        df, test_size=test_size, val_size=val_size,
        seed=seed, stratify_col="GROUP"
    )


def generalisation_split(
    df: pd.DataFrame,
    train_groups: List[str],
    test_groups:  List[str],
    val_size: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split for Experiment 5 (out-of-distribution generalisation).

    Train set : patients from train_groups only.
    Test set  : patients from test_groups only (different clinical groups).
    Val set   : held-out patients from train_groups (same distribution
                as train, used for early stopping).

    This simulates real-world deployment: train on one cohort, test on
    a different clinical population to assess generalisation.
    """
    train_pool = df[df["GROUP"].isin(train_groups)]
    test_df    = df[df["GROUP"].isin(test_groups)].copy()

    # Val is a patient-level holdout from the training groups
    train_patients = train_pool["ID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(train_patients)

    n_val    = max(1, int(len(train_patients) * val_size))
    val_ids  = train_patients[:n_val]
    train_ids = train_patients[n_val:]

    train_df = train_pool[train_pool["ID"].isin(train_ids)].copy()
    val_df   = train_pool[train_pool["ID"].isin(val_ids)].copy()

    _verify_no_leakage(
        list(train_ids), list(val_ids),
        list(test_df["ID"].unique())
    )
    _log_split_summary(train_df, val_df, test_df, "GROUP")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _verify_no_leakage(train_ids, val_ids, test_ids):
    """Assert zero overlap across splits. Raises if any leakage found."""
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set   & test_set

    assert not tv, f"LEAKAGE: {len(tv)} patients in both train and val: {tv}"
    assert not tt, f"LEAKAGE: {len(tt)} patients in both train and test: {tt}"
    assert not vt, f"LEAKAGE: {len(vt)} patients in both val and test: {vt}"

    log.info("  No patient leakage detected across splits.")


def _log_split_summary(train_df, val_df, test_df, stratify_col):
    log.info(f"\n  Patient-level split summary:")
    for name, split in [("Train", train_df),
                         ("Val",   val_df),
                         ("Test",  test_df)]:
        n_patients = split["ID"].nunique()
        n_rows     = len(split)
        groups     = split[stratify_col].value_counts().to_dict()
        log.info(f"    {name:<6}: {n_patients} patients, "
                 f"{n_rows} rows  |  {groups}")