"""
─────────────────────────────────────────────────────────────────────────────
Class imbalance handling for clinical audio classification.

Why imbalance matters here
──────────────────────────
In Experiment 1 (CRS vs Control), FESS Session 1 patients may outnumber
control patients or vice versa.  An unbalanced model silently learns to
predict the majority class, producing misleadingly high accuracy while
completely failing on the minority class — exactly the class that
contains the clinically important signal.

We provide three strategies and show before/after metrics for each:
  1. "none"        → raw counts, no correction (baseline for comparison)
  2. "weights"     → inverse-frequency class weights passed to the loss
  3. "oversample"  → WeightedRandomSampler resamples minority at train time
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

log = logging.getLogger(__name__)


def compute_class_weights(df: pd.DataFrame,
                           label_fn: Callable,
                           num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training dataframe.

    weight[c] = total_samples / (num_classes * count[c])

    These are passed to nn.CrossEntropyLoss(weight=...) so rare classes
    receive proportionally larger gradient updates.

    Returns float32 Tensor of shape [num_classes].
    """
    labels = df.apply(label_fn, axis=1).dropna().astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)

    # Guard against empty classes
    counts = np.where(counts == 0, 1e-6, counts)
    weights = len(labels) / (num_classes * counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    log.info(f"\n  Class weights (inverse frequency):")
    for i, (cnt, w) in enumerate(zip(counts, weights)):
        log.info(f"    Class {i}: {int(cnt):>5} samples  ->  weight {w:.4f}")

    return weights_tensor


def build_weighted_sampler(dataset,
                            label_fn: Callable = None) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each training batch has balanced classes.

    This is an ALTERNATIVE to loss weighting — it resamples the dataset
    at the DataLoader level so the model sees equal class frequencies
    per batch, which can be more effective when imbalance is severe (>5:1).

    Works with SinusitisDataset (has .samples attribute with labels).
    """
    # Extract labels from dataset
    if hasattr(dataset, "samples"):
        labels = [label for _, label, *_ in dataset.samples]
    elif hasattr(dataset, "pairs"):
        labels = [label for _, _, label in dataset.pairs]
    else:
        raise ValueError("Dataset must have .samples or .pairs attribute.")

    class_counts = np.bincount(labels).astype(float)
    class_counts = np.where(class_counts == 0, 1e-6, class_counts)

    # Weight per sample = inverse of its class frequency
    sample_weights = torch.tensor(
        [1.0 / class_counts[l] for l in labels],
        dtype=torch.float32
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    log.info(f"  WeightedRandomSampler built: {len(sample_weights)} samples, "
             f"class counts: {class_counts.astype(int).tolist()}")
    return sampler


def log_imbalance_report(dataset, split_name: str):
    """Log class distribution for a dataset split."""
    if hasattr(dataset, "class_counts"):
        counts = dataset.class_counts()
    else:
        return

    total = sum(counts.values())
    log.info(f"\n  Class distribution [{split_name}]:")
    for cls in sorted(counts):
        cnt = counts[cls]
        log.info(f"    Class {cls}: {cnt:>5} segments  ({100*cnt/total:.1f}%)")

    if len(counts) >= 2:
        vals = list(counts.values())
        ratio = max(vals) / max(min(vals), 1)
        status = "IMBALANCED" if ratio > 2 else "Balanced"
        log.info(f"    Imbalance ratio: {ratio:.2f}:1  {status}")