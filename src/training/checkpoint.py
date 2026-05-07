import json
import torch
import numpy as np
from pathlib import Path as _Path
import logging
 
 
def save_checkpoint(model, optimizer, scheduler,
                    epoch: int, val_loss: float,
                    output_dir: str,
                    filename: str = "checkpoint.pt"):
    _Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    torch.save(ckpt, str(_Path(output_dir) / filename))
    log.info(f"  Checkpoint saved -> {filename}")
 
 
def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    torch.serialization.add_safe_globals([
            np.ndarray,
            np._core.multiarray._reconstruct,
            np.dtype,
            np.dtypes.Float32DType,
        ])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"  Resumed from {path}  "
             f"(epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    return ckpt["epoch"], ckpt["val_loss"]
 
 