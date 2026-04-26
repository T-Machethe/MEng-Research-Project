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
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
 
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
 
# Flat list of all known audio columns
ALL_AUDIO_COLS = [col for cols in AUDIO_TYPE_GROUPS.values() for col in cols]
 
# Reverse map: individual col → group name
COL_TO_GROUP = {
    col: group
    for group, cols in AUDIO_TYPE_GROUPS.items()
    for col in cols
}
 
 
class SinusitisDataset(Dataset):
    """
    Segment-level dataset with audio type tagging.
 
    Each sample returns:
        waveform  : torch.Tensor [1, T]
        label     : int
        audio_col : str  (e.g. 'a', 'speech', 'a1', 'agua')
 
    The audio_col tag is used at evaluation time to slice predictions
    by audio type without needing separate datasets.
 
    audio_cols parameter
    ────────────────────
    None        → include all available audio types (mixed training)
    ['a','e']   → include only those columns (specialist training)
    """
                
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

        # Scan directory ONCE — build {filename: full_path} lookup
        all_files = {
            f.name: str(f)
            for f in self.segment_dir.rglob("*.pt")
        }

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
                prefix = f"ID{subject_id}_ses{session}_{col}"
                for fname, fpath in all_files.items():
                    if fname.startswith(prefix):
                        self.samples.append((fpath, int(label), col))

        if not self.samples:
            log.warning(
                f"SinusitisDataset: 0 segments found. "
                f"audio_cols={self.audio_cols}, "
                f"segment_dir={self.segment_dir}"
            )
        else:
            log.info(
                f"  Dataset: {len(self.samples)} segments | "
                f"{df['ID'].nunique()} patients |"
                f"cols: {self.audio_cols}"
            )
 
    def __len__(self) -> int:
        return len(self.samples)
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        filepath, label, audio_col = self.samples[idx]
        x = torch.load(filepath, weights_only=True)   # [1, T]
        if self.transform:
            x = self.transform(x)
        return x, label, audio_col
 
    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for _, label, *_ in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts
 
    def col_counts(self) -> Dict[str, int]:
        """How many segments per audio column."""
        counts: Dict[str, int] = {}
        for _, _, col in self.samples:
            counts[col] = counts.get(col, 0) + 1
        return counts
 
    def type_group_counts(self) -> Dict[str, int]:
        """How many segments per audio type GROUP."""
        counts: Dict[str, int] = {}
        for _, _, col in self.samples:
            group = COL_TO_GROUP.get(col, "other")
            counts[group] = counts.get(group, 0) + 1
        return counts
 
 
class PairedDataset(Dataset):
    """
    Within-patient paired dataset for Experiment 4.
    Returns (x1, x2, label, audio_col) — col from the pre-session segment.
    """
 
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
 
        # (path1, path2, label, audio_col)
        self.pairs: List[Tuple[str, str, int, str]] = []
 
        for patient_id in df["ID"].unique():
            patient_id = str(patient_id).strip()
 
            for col in self.audio_cols:
                pre_segs = sorted([
                    str(f) for f in Path(self.segment_dir).rglob("*.pt")
                    if f"ID{patient_id}_ses{pre_session}_{col}" in f.name
                ])

                if not pre_segs:
                    continue
 
                # Positive pairs: pre vs post
                for post_ses in post_sessions:
                    post_segs = sorted([
                        str(f) for f in Path(self.segment_dir).rglob("*.pt")
                        if f"ID{patient_id}_ses{post_ses}_{col}" in f.name
                    ])
                    for i, pre_seg in enumerate(pre_segs):
                        if post_segs:
                            post_seg = post_segs[i % len(post_segs)]
                            self.pairs.append((pre_seg, post_seg, 1, col))
 
                # Negative pairs: pre vs pre
                if include_negatives and len(pre_segs) >= 2:
                    n_neg = int(len(pre_segs) * neg_ratio)
                    for i in range(min(n_neg, len(pre_segs) - 1)):
                        self.pairs.append(
                            (pre_segs[i], pre_segs[i + 1], 0, col)
                        )
 
        pos = sum(1 for *_, l, c in self.pairs if l == 1)
        neg = sum(1 for *_, l, c in self.pairs if l == 0)
        log.info(f"  PairedDataset: {len(self.pairs)} pairs "
                 f"(pos={pos}, neg={neg})")
 
    def __len__(self) -> int:
        return len(self.pairs)
 
    def __getitem__(self, idx):
        path1, path2, label, col = self.pairs[idx]
        x1 = torch.load(path1, weights_only=True)
        x2 = torch.load(path2, weights_only=True)
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return x1, x2, label, col
 
    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for *_, label, col in self.pairs:
            counts[label] = counts.get(label, 0) + 1
        return counts
 