"""
Audio type groups used throughout
──────────────────────────────────
  VOWELS     : a, e, i, o, u
  SUSTAINED  : a1, a2, a3
  SPEECH     : speech
  TDU        : agua, brasero, dia, mesa

─────────────────────────────────────────────────────────────────────────────
Dataset classes for all five experiments.

SinusitisDataset     — standard segment-level dataset (Exp 1, 2, 3, 5)
PairedDataset        — within-patient paired recordings (Exp 4)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Canonical groupings of audio columns by clinical category
AUDIO_TYPE_GROUPS = {
    "vowels":    ["a", "e", "i", "o", "u"],
    "sustained": ["a1", "a2", "a3"],
    "speech":    ["speech"],
    "tdu":       ["agua", "brasero", "dia", "mesa"],
}

ALL_AUDIO_COLS = [col for cols in AUDIO_TYPE_GROUPS.values() for col in cols]

COL_TO_GROUP = {
    col: group
    for group, cols in AUDIO_TYPE_GROUPS.items()
    for col in cols
}


# ------------------------------------------------------------------
# 🔧 Shared index builder (used by both datasets)
# ------------------------------------------------------------------
def build_file_index(segment_dir: Path):
    """
    Builds:
    1. prefix_index:  IDxxx_sesX_col → [Path, Path, ...]
    2. structured_index: ID → session → col → [Path, ...]
    """
    prefix_index = defaultdict(list)
    structured_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in segment_dir.rglob("*.pt"):
        f = Path(f)
        parts = f.name.split("_")

        if len(parts) < 3:
            continue

        id_part, ses_part, col = parts[:3]

        key = f"{id_part}_{ses_part}_{col}"
        prefix_index[key].append(f)

        # structured index
        subject_id = id_part.replace("ID", "")
        session = int(ses_part.replace("ses", ""))

        structured_index[subject_id][session][col].append(f)

    return prefix_index, structured_index


# ------------------------------------------------------------------
# 🧠 SinusitisDataset (FAST)
# ------------------------------------------------------------------
class SinusitisDataset(Dataset):

    def __init__(self,
                 df,
                 segment_dir: str,
                 label_fn: Callable,
                 audio_cols: Optional[List[str]] = None,
                 transform: Optional[Callable] = None):

        self.segment_dir = Path(segment_dir)
        self.transform   = transform
        self.audio_cols  = audio_cols or ALL_AUDIO_COLS
        self.samples     = []

        # Build index once
        prefix_index, _ = build_file_index(self.segment_dir)
        self._file_index = prefix_index

        for _, row in df.iterrows():
            subject_id = str(row["ID"]).strip()
            session    = int(row["session"])

            try:
                label = label_fn(row)
            except Exception:
                continue

            if label is None:
                continue

            for col in self.audio_cols:
                key = f"ID{subject_id}_ses{session}_{col}"
                matches = self._file_index.get(key, [])

                for fpath in matches:
                    self.samples.append((fpath, int(label), col))

        if not self.samples:
            log.warning(
                f"SinusitisDataset: 0 segments found. "
                f"audio_cols={self.audio_cols}, "
                f"segment_dir={self.segment_dir}"
            )
        else:
            log.info(
                f"Dataset: {len(self.samples)} segments | "
                f"{df['ID'].nunique()} patients | "
                f"cols: {self.audio_cols}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, audio_col = self.samples[idx]
        x = torch.load(str(filepath), weights_only=True)

        if self.transform:
            x = self.transform(x)

        return x, label, audio_col

    def class_counts(self):
        counts = {}
        for _, label, _ in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def col_counts(self):
        counts = {}
        for _, _, col in self.samples:
            counts[col] = counts.get(col, 0) + 1
        return counts

    def type_group_counts(self):
        counts = {}
        for _, _, col in self.samples:
            group = COL_TO_GROUP.get(col, "other")
            counts[group] = counts.get(group, 0) + 1
        return counts


# ------------------------------------------------------------------
# 🔗 PairedDataset (NOW ALSO FAST)
# ------------------------------------------------------------------
class PairedDataset(Dataset):

    def __init__(self,
                 df,
                 segment_dir: str,
                 pre_session: int = 1,
                 post_sessions: List[int] = (2, 3),
                 include_negatives: bool = True,
                 neg_ratio: float = 1.0,
                 audio_cols: Optional[List[str]] = None,
                 transform: Optional[Callable] = None):

        self.segment_dir = Path(segment_dir)
        self.transform   = transform
        self.audio_cols  = audio_cols or ALL_AUDIO_COLS

        self.pairs: List[Tuple[Path, Path, int, str]] = []

        # Build structured index once
        _, structured_index = build_file_index(self.segment_dir)

        for patient_id in df["ID"].unique():
            patient_id = str(patient_id).strip()

            patient_data = structured_index.get(patient_id, {})

            for col in self.audio_cols:

                pre_segs = patient_data.get(pre_session, {}).get(col, [])

                if not pre_segs:
                    continue

                # -------------------
                # Positive pairs
                # -------------------
                for post_ses in post_sessions:
                    post_segs = patient_data.get(post_ses, {}).get(col, [])

                    for i, pre_seg in enumerate(pre_segs):
                        if post_segs:
                            post_seg = post_segs[i % len(post_segs)]
                            self.pairs.append((pre_seg, post_seg, 1, col))

                # -------------------
                # Negative pairs
                # -------------------
                if include_negatives and len(pre_segs) >= 2:
                    n_neg = int(len(pre_segs) * neg_ratio)

                    for i in range(min(n_neg, len(pre_segs) - 1)):
                        self.pairs.append(
                            (pre_segs[i], pre_segs[i + 1], 0, col)
                        )

        pos = sum(1 for *_, l, _ in self.pairs if l == 1)
        neg = sum(1 for *_, l, _ in self.pairs if l == 0)

        log.info(
            f"PairedDataset: {len(self.pairs)} pairs "
            f"(pos={pos}, neg={neg})"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label, col = self.pairs[idx]

        x1 = torch.load(str(path1), weights_only=True)
        x2 = torch.load(str(path2), weights_only=True)

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1, x2, label, col

    def class_counts(self):
        counts = {}
        for *_, label, _ in self.pairs:
            counts[label] = counts.get(label, 0) + 1
        return counts