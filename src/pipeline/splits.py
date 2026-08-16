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

# ─────────────────────────────────────────────────────────────────────────────
# Test split demographic description
# ─────────────────────────────────────────────────────────────────────────────

def describe_test_split(
    full_df:  "pd.DataFrame",
    train_df: "pd.DataFrame",
    val_df:   "pd.DataFrame",
    test_df:  "pd.DataFrame",
    audio_cols: list = None,
) -> tuple:
    """
    Build a demographic profile of the test split patients and return:
        (demo_df, flags, latex_table)

    Parameters
    ----------
    full_df   : full clinical CSV (all patients, all sessions)
    train_df  : training partition (output of patient_level_split)
    val_df    : validation partition
    test_df   : test partition
    audio_cols: list of audio channel column names to count recordings

    Returns
    -------
    demo_df     : pd.DataFrame — one row per test patient
    flags       : list[str]   — demographic imbalance warnings
    latex_table : str         — booktabs LaTeX table for methodology chapter
    """
    import numpy as np
    import pandas as pd

    GROUP_LABEL = {
        "FESS":    "FESS (CRS)",
        "Contr":   "Control",
        "Sept":    "Septoplasty",
        "Tonsill": "Tonsillectomy",
    }

    if audio_cols is None:
        # Fallback: use any column that looks like an audio path column
        audio_cols = [c for c in full_df.columns
                      if c in ("a","e","i","o","u","a1","a2","a3",
                               "agua","brasero","dia","mesa","speech","un")]

    records  = []
    test_ids = sorted(test_df["ID"].unique())

    for pid in test_ids:
        rows  = full_df[full_df["ID"] == pid]
        if rows.empty:
            continue
        group    = rows["GROUP"].iloc[0].strip() if "GROUP" in rows.columns else "?"
        group_lbl = GROUP_LABEL.get(group, group)
        sessions  = sorted(rows["session"].dropna().unique().astype(int).tolist()) \
                    if "session" in rows.columns else []
        ses_str   = "/".join(str(s) for s in sessions)

        age = None
        for col in ("Age","AGE","age","edad"):
            if col in rows.columns:
                v = rows[col].dropna()
                if not v.empty:
                    age = float(v.iloc[0]); break

        gender = None
        for col in ("Gender","GENDER","gender","sexo","Sex","SEX"):
            if col in rows.columns:
                v = rows[col].dropna()
                if not v.empty:
                    gender = str(v.iloc[0]).strip(); break

        n_recs = int(sum(rows[c].notna().sum() for c in audio_cols if c in rows.columns))

        records.append({
            "Patient ID": pid,
            "Group":      group_lbl,
            "Sessions":   ses_str,
            "Age":        f"{age:.0f}" if age is not None else "N/A",
            "Gender":     gender if gender else "N/A",
            "Recordings": n_recs,
        })

    demo_df = pd.DataFrame(records)

    # ── Imbalance flags ────────────────────────────────────────────────────────
    flags = []
    fess_n = demo_df["Group"].str.contains("FESS").sum()
    ctrl_n = demo_df["Group"].str.contains("Control").sum()
    if fess_n == 0:
        flags.append("CRITICAL: No FESS patients in test set — cannot estimate sensitivity.")
    if ctrl_n == 0:
        flags.append("CRITICAL: No Control patients in test set — cannot estimate specificity.")
    if fess_n != ctrl_n:
        flags.append(
            f"WARN: Unequal class balance ({fess_n} FESS vs {ctrl_n} Control) — "
            "macro-F1 and AUC do not reflect a balanced operating point."
        )
    ages = pd.to_numeric(demo_df["Age"], errors="coerce").dropna()
    if len(ages) > 1 and ages.std() < 5:
        flags.append(
            f"WARN: Low age variance in test set (σ={ages.std():.1f}) — "
            "results may not generalise across age groups."
        )
    genders = demo_df[demo_df["Gender"] != "N/A"]["Gender"].str.upper()
    if len(genders) > 0 and genders.nunique() == 1:
        flags.append(
            f"WARN: All test patients are {genders.iloc[0]} — "
            "gender-specific vocal tract differences may affect generalisation."
        )
    if not any("1" in s for s in demo_df["Sessions"].tolist()):
        flags.append(
            "WARN: No Session 1 (pre-op) recordings in test — "
            "cannot evaluate pre-operative detection performance."
        )
    if not flags:
        flags.append("No critical imbalances detected.")

    # ── LaTeX table ────────────────────────────────────────────────────────────
    n_tr = train_df["ID"].nunique()
    n_va = val_df["ID"].nunique()
    n_te = test_df["ID"].nunique()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Demographic profile of the %d test-split patients "
        r"(seed=42, stratified patient-level split: %d train / %d val / %d test). "
        r"Session numbers: 1\,=\,pre-operative, 2\,=\,2-week post-operative, "
        r"3\,=\,3-month post-operative. "
        r"Recordings: non-empty audio channel cells available for the patient.}" % (n_te, n_tr, n_va, n_te),
        r"\label{tab:test_split_demographics}",
        r"\begin{tabular}{llcccr}",
        r"\toprule",
        r"Patient & Group & Sessions & Age & Gender & Recordings \\",
        r"\midrule",
    ]

    fess_rows  = demo_df[demo_df["Group"].str.contains("FESS")]
    other_rows = demo_df[~demo_df["Group"].str.contains("FESS")]
    for df_part, first in [(fess_rows, True), (other_rows, False)]:
        if not df_part.empty and not first:
            lines.append(r"\addlinespace")
        for _, r in df_part.iterrows():
            lines.append(
                f"  {r['Patient ID']} & {r['Group']} & {r['Sessions']} & "
                f"{r['Age']} & {r['Gender']} & {r['Recordings']} \\\\"
            )

    lines += [
        r"\midrule",
        rf"  \multicolumn{{2}}{{l}}{{Total}} & & & & {demo_df['Recordings'].sum()} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_table = "\n".join(lines)

    return demo_df, flags, latex_table


def patient_group_kfold(
    df: pd.DataFrame,
    n_splits: int,
    seed: int = 42,
    stratify_col: str = "GROUP",
) -> List[Tuple[List[str], List[str]]]:
    """
    Patient-grouped K-fold splitting for (nested) cross-validation.

    Same non-negotiable guarantee as patient_level_split(): no patient
    (by "ID") ever appears in both the train and held-out portion of a
    fold — every row for a given patient, across every session, moves
    together. This is the CV analogue of patient_level_split()'s single
    train/val/test split; use this one when you need K rotating folds
    (nested CV outer loop, nested CV inner loop, or plain K-fold) instead
    of one fixed partition.

    Stratification: uses sklearn's StratifiedGroupKFold when available
    (sklearn >= 1.1), which balances both class/group proportions AND
    fold sizes as well as the group constraint allows. Falls back to a
    manual per-stratum round-robin assignment (same spirit as
    patient_level_split()'s per-group loop) if StratifiedGroupKFold isn't
    importable, so this doesn't hard-fail on an older sklearn build (e.g.
    an older Colab image) — the manual fallback still guarantees the
    patient-grouping constraint; it's the class-balance-across-folds
    property that's best-effort in that path, not the leakage guarantee.

    Parameters
    ----------
    df            : full filtered dataframe (one row per recording/session).
    n_splits      : number of folds (K).
    seed          : random seed for reproducibility.
    stratify_col  : column used for stratified fold assignment.

    Returns
    -------
    List of (train_ids, test_ids) tuples, one per fold, each a list of
    patient ID strings. Caller filters df by df["ID"].isin(ids) to get
    the actual fold dataframes — kept as ID lists (not dataframes) so the
    same fold assignment can be reused against different segment_dirs
    or re-applied to a differently-filtered dataframe with the same
    patient population (e.g. inner folds re-splitting an outer-fold's
    training patients).
    """
    patient_groups = df.groupby("ID")[stratify_col].first().reset_index()
    ids     = patient_groups["ID"].values
    strata  = patient_groups[stratify_col].values

    smallest_stratum = pd.Series(strata).value_counts().min()
    if smallest_stratum < n_splits:
        log.warning(
            f"  [patient_group_kfold] Smallest stratum in {stratify_col!r} "
            f"has only {smallest_stratum} patient(s), fewer than "
            f"n_splits={n_splits}. Some folds may end up without a "
            f"patient from that stratum — check fold sizes below."
        )

    folds: List[Tuple[List[str], List[str]]] = []
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(ids, strata, groups=ids):
            folds.append((ids[train_idx].tolist(), ids[test_idx].tolist()))
    except ImportError:
        log.warning(
            "  [patient_group_kfold] sklearn.model_selection."
            "StratifiedGroupKFold not importable (sklearn < 1.1?) — "
            "falling back to manual per-stratum round-robin assignment. "
            "Patient-grouping guarantee still holds; fold class-balance "
            "is best-effort in this path."
        )
        rng = np.random.default_rng(seed)
        fold_ids: List[List[str]] = [[] for _ in range(n_splits)]
        for group in pd.unique(strata):
            # .to_numpy(dtype=object) rather than .values: pandas' newer
            # StringDtype backs .values with a StringArray, which
            # np.random.Generator.shuffle only shuffles safely for plain
            # ndarrays/sequences — silently risking duplicated entries
            # otherwise (this is exactly the kind of bug that would
            # silently reintroduce patient leakage, so worth being
            # explicit here rather than trusting the array type).
            group_ids = (patient_groups[patient_groups[stratify_col] == group]["ID"]
                         .to_numpy(dtype=object).copy())
            rng.shuffle(group_ids)
            for i, pid in enumerate(group_ids):
                fold_ids[i % n_splits].append(pid)
        for k in range(n_splits):
            test_ids  = fold_ids[k]
            train_ids = [pid for j in range(n_splits) if j != k for pid in fold_ids[j]]
            folds.append((train_ids, test_ids))

    # Verify the one guarantee that must never break, regardless of path taken.
    for k, (train_ids, test_ids) in enumerate(folds):
        overlap = set(train_ids) & set(test_ids)
        assert not overlap, (
            f"[patient_group_kfold] Fold {k}: patient(s) {overlap} appear in "
            f"both train and test — this must never happen."
        )

    log.info(f"\n  [patient_group_kfold] {n_splits} folds over "
             f"{len(ids)} patients, stratified by {stratify_col!r}:")
    for k, (train_ids, test_ids) in enumerate(folds):
        log.info(f"    Fold {k}: train={len(train_ids)} patients, "
                 f"test={len(test_ids)} patients")

    return folds


# ── CLI entry point ─────────────────────────────────────────────────────────-
if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Describe the demographic profile of the test split."
    )
    parser.add_argument("--csv",         required=True,
                        help="Path to clinical_all_sessions.csv")
    parser.add_argument("--output_tex",  default=None,
                        help="Path to write the LaTeX table (optional)")
    args = parser.parse_args()

    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.pipeline.splits import patient_level_split, describe_test_split

    full_df = pd.read_csv(args.csv)
    full_df["GROUP"] = full_df["GROUP"].str.strip()
    exp1_df = full_df[full_df["GROUP"].isin(["FESS","Contr"])].copy()

    train_df, val_df, test_df = patient_level_split(
        exp1_df, test_size=0.20, val_size=0.10, seed=42, stratify_col="GROUP"
    )

    demo_df, flags, latex = describe_test_split(
        full_df, train_df, val_df, test_df
    )

    print("\n" + "═"*60)
    print("  TEST SPLIT DEMOGRAPHIC PROFILE")
    print("═"*60)
    print(demo_df.to_string(index=False))
    print("\n  Flags:")
    for f in flags:
        print(f"  • {f}")
    print("\n  LaTeX Table:\n")
    print(latex)

    if args.output_tex:
        Path(args.output_tex).write_text(latex)
        print(f"\nLaTeX written → {args.output_tex}")