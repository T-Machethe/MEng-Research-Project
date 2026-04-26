import torch
from functools import partial
import torch.nn.functional as TF
from torch.utils.data import DataLoader
import numpy as np

from src.pipeline.dataset import SinusitisDataset, PairedDataset
from src.training.imbalance import (
    build_weighted_sampler, log_imbalance_report
)

 
 
def collate_standard(batch):
    """
    Collate (waveform, label, audio_col) batches.
    Pads to longest waveform in batch.
    """
    waveforms, labels, cols = zip(*batch)
    max_len = max(w.shape[-1] for w in waveforms)
 
    padded, masks = [], []
    for w in waveforms:
        pad_len = max_len - w.shape[-1]
        mask    = torch.ones(w.shape[-1], dtype=torch.long)
        if pad_len > 0:
            w    = TF.pad(w, (0, pad_len))
            mask = TF.pad(mask, (0, pad_len))
        padded.append(w.squeeze(0))
        masks.append(mask)
 
    return {
        "input_values":   torch.stack(padded),
        "attention_mask": torch.stack(masks),
        "labels":         torch.tensor(labels, dtype=torch.long),
        "audio_cols":     list(cols),        # ← kept as list of strings
    }
 
 
def collate_paired(batch):
    """Collate (x1, x2, label, audio_col) paired batches."""
    x1s, x2s, labels, cols = zip(*batch)
    max_len = max(
        max(x.shape[-1] for x in x1s),
        max(x.shape[-1] for x in x2s),
    )
 
    def pad_seq(seqs):
        out, masks = [], []
        for w in seqs:
            pad_len = max_len - w.shape[-1]
            mask    = torch.ones(w.shape[-1], dtype=torch.long)
            if pad_len > 0:
                w    = TF.pad(w, (0, pad_len))
                mask = TF.pad(mask, (0, pad_len))
            out.append(w.squeeze(0))
            masks.append(mask)
        return torch.stack(out), torch.stack(masks)
 
    x1_padded, mask1 = pad_seq(x1s)
    x2_padded, mask2 = pad_seq(x2s)
 
    return {
        "input_values_1":   x1_padded,
        "attention_mask_1": mask1,
        "input_values_2":   x2_padded,
        "attention_mask_2": mask2,
        "labels":           torch.tensor(labels, dtype=torch.long),
        "audio_cols":       list(cols),
    }
 
def seed_worker(worker_id):
    """
    Module-level worker init function — must be defined at module level
    to be picklable on Windows (which uses 'spawn' multiprocessing).
    """
    np.random.seed(torch.initial_seed() % 2**32) 
 
def build_experiment_loaders(
    train_df, val_df, test_df,
    segment_dir: str,
    label_fn,
    batch_size: int = 8,
    imbalance_strategy: str = "weights",
    class_weights=None,
    num_workers: int = 2,
    seed: int = 42,
    paired: bool = False,
    audio_cols=None,
):
    """
    Build train / val / test DataLoaders for any experiment.
 
    Parameters
    ----------
    imbalance_strategy : "none" | "weights" | "oversample"
        "weights"    → class_weights passed to loss (handled in trainer)
        "oversample" → WeightedRandomSampler used in train DataLoader
    paired             → use PairedDataset + paired collate (Exp 4)
    audio_cols         → restrict to specific task columns (None = all)
    """
    DatasetClass = PairedDataset if paired else SinusitisDataset
    collate_fn   = collate_paired if paired else collate_standard
 
    if paired:
        train_ds = PairedDataset(train_df, segment_dir)
        val_ds   = PairedDataset(val_df,   segment_dir)
        test_ds  = PairedDataset(test_df,  segment_dir)
    else:
        train_ds = SinusitisDataset(train_df, segment_dir,
                                    label_fn, audio_cols)
        val_ds   = SinusitisDataset(val_df,   segment_dir,
                                    label_fn, audio_cols)
        test_ds  = SinusitisDataset(test_df,  segment_dir,
                                    label_fn, audio_cols)
 
    # Log class distribution for every split
    log_imbalance_report(train_ds, "train")
    log_imbalance_report(val_ds,   "val")
    log_imbalance_report(test_ds,  "test")
 
    # Sampler for training
    train_sampler = None
    train_shuffle = True
    if imbalance_strategy == "oversample":
        train_sampler = build_weighted_sampler(train_ds)
        train_shuffle = False   # sampler and shuffle are mutually exclusive
 
    pin = torch.cuda.is_available()
 
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        worker_init_fn=seed_worker,
    )
    
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )
 
    return train_loader, val_loader, test_loader
 