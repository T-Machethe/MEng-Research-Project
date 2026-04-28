import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path as _Path
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    get_linear_schedule_with_warmup,
)

import logging
log = logging.getLogger(__name__)
log_epoch = logging.getLogger("epoch_summary")   # always visible on console

from src.training.metrics import compute_metrics

class Wav2Vec2Classifier(nn.Module):
    """
    wav2vec 2.0 backbone with a custom classification head.

    Defined at module level (not inside _build_model) so it is
    picklable on Windows multiprocessing and can be saved/loaded
    cleanly with torch.save / torch.load.
    """

    def __init__(self, backbone, hidden_size: int, num_classes: int):
        super().__init__()
        self.backbone   = backbone
        self.dropout    = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        # Mean-pool over latent time frames [B, T', H] → [B, H]
        hidden = outputs.last_hidden_state.mean(dim=1)
        hidden = self.dropout(hidden)
        return self.classifier(hidden)   # [B, num_classes]


class Trainer:
    """
    Shared training loop for all five experiments.

    Supports both scratch and fine-tuning modes.
    Handles class-weighted loss, gradient clipping,
    LR scheduling, early stopping, and checkpointing.
    """

    def __init__(self, cfg, num_classes: int,
                 class_weights=None, output_dir: str = "."):
        self.cfg         = cfg
        self.num_classes = num_classes
        self.output_dir  = _Path(output_dir)
        self.device      = self._resolve_device(cfg.device)
        self.model       = self._build_model()
        self.optimizer   = self._build_optimizer()
        self.scheduler   = None
        self.loss_fn     = self._build_loss(class_weights)
        self.writer      = SummaryWriter(
            log_dir=str(self.output_dir / "tensorboard")
        )

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return torch.device(device_str)

    def _build_model(self) -> Wav2Vec2Classifier:
        cfg = self.cfg

        if cfg.mode == "scratch":
            config   = Wav2Vec2Config(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
            )
            backbone = Wav2Vec2Model(config)

        else:
            # Load backbone only — no lm_head, no vocabulary-size mismatch.
            # mask_time_prob / mask_feature_prob MUST be 0.0 for fine-tuning:
            # masked_spec_embed is randomly initialised and produces NaN
            # outputs when masking is active, causing loss=NaN from step 0.
            backbone = Wav2Vec2Model.from_pretrained(
                cfg.pretrained,
                mask_time_prob=0.0,      # ← disable time masking (NaN fix)
                mask_feature_prob=0.0,   # ← disable feature masking (NaN fix)
            )

            if cfg.freeze_encoder:
                backbone.feature_extractor.requires_grad_(False)
                backbone.feature_projection.requires_grad_(False)

            for i, layer in enumerate(backbone.encoder.layers):
                if i < cfg.freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        hidden_size = backbone.config.hidden_size
        model       = Wav2Vec2Classifier(backbone, hidden_size,
                                         self.num_classes)

        n_train = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log.info(f"  Model: {n_train/1e6:.2f}M / "
                 f"{n_total/1e6:.2f}M trainable.")
        return model.to(self.device)

    def _build_optimizer(self) -> AdamW:
        trainable = [p for p in self.model.parameters()
                     if p.requires_grad]
        return AdamW(
            trainable,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

    def _build_loss(self, class_weights) -> nn.CrossEntropyLoss:
        if class_weights is not None:
            weights = class_weights.to(self.device)
            log.info(f"  Using weighted loss: {weights.tolist()}")
        else:
            weights = None
        return nn.CrossEntropyLoss(weight=weights)

    def fit(self, train_loader, val_loader, test_loader) -> dict:
        cfg         = self.cfg
        total_steps = len(train_loader) * cfg.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_loss    = float("inf")
        patience_counter = 0
        global_step      = 0
        val_metrics      = {}
        all_train_metrics = []   # ← collect per-epoch train metrics
         
         # ── Resume from checkpoint if one exists ──────────────────────────────
        resume_path = self.output_dir / "latest_checkpoint.pt"
        if resume_path.exists():
            log.info(f"\n  Resuming from {resume_path}")
            ckpt = torch.load(
                resume_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") and self.scheduler:
                self.scheduler.load_state_dict(ckpt["scheduler"])

            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            global_step      = ckpt.get("global_step", 0)
            all_train_metrics = ckpt.get("train_history", [])

            log.info(
                f"  Resumed at epoch {start_epoch} | "
                f"best_val_loss so far: {best_val_loss:.4f} | "
                f"global_step: {global_step}"
            )
        else:
            log.info("  No checkpoint found — starting from epoch 1.")
            
            
        for epoch in range(1, cfg.num_epochs + 1):

            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            epoch_loss  = 0.0
            t0          = time.time()

            # Collect predictions during training for metric computation
            train_labels, train_preds, train_probs = [], [], []

            for batch in train_loader:
                labels = batch["labels"].to(self.device)

                # ── Paired batch (Exp 4): concatenate pre/post along batch dim
                if "input_values_1" in batch:
                    iv1 = torch.nan_to_num(
                        batch["input_values_1"].to(self.device),
                        nan=0.0, posinf=1.0, neginf=-1.0)
                    iv2 = torch.nan_to_num(
                        batch["input_values_2"].to(self.device),
                        nan=0.0, posinf=1.0, neginf=-1.0)
                    am1 = batch["attention_mask_1"].to(self.device)
                    am2 = batch["attention_mask_2"].to(self.device)
                    # Mean of both embeddings → single logit vector
                    logits = (
                        self.model(input_values=iv1, attention_mask=am1) +
                        self.model(input_values=iv2, attention_mask=am2)
                    ) / 2.0
                else:
                    input_values   = batch["input_values"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    input_values = torch.nan_to_num(
                        input_values, nan=0.0, posinf=1.0, neginf=-1.0
                    )
                    logits = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask,
                    )

                loss = self.loss_fn(logits, labels)

                if not torch.isfinite(loss):
                    log.warning(f"  NaN/Inf loss at step {global_step} — skipping.")
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss  += loss.item()
                global_step += 1

                # Collect training predictions for epoch-level metrics
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = logits.argmax(dim=-1).cpu().tolist()
                train_labels.extend(labels.cpu().tolist())
                train_preds.extend(preds)
                train_probs.extend(probs)

                if global_step % cfg.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/loss", loss.item(), global_step)
                    log.debug(
                        f"  Ep {epoch:03d}  step {global_step:05d}  "
                        f"loss {loss.item():.4f}  lr {lr:.2e}"
                    )

            avg_train_loss = epoch_loss / max(len(train_loader), 1)

            # ── Compute training metrics for this epoch ────────────────────────
            train_metrics = compute_metrics(
                train_labels, train_preds,
                np.array(train_probs),
                self.num_classes,
                split_name="train",
            )
            train_metrics["train/loss"] = avg_train_loss

            # Log training metrics to TensorBoard
            self.writer.add_scalar("train/accuracy",
                                train_metrics["train/accuracy"], global_step)
            self.writer.add_scalar("train/f1_macro",
                                train_metrics["train/f1_macro"], global_step)

            log.debug(
                f"  Epoch {epoch:03d} TRAIN | "
                f"loss {avg_train_loss:.4f} | "
                f"acc {train_metrics['train/accuracy']:.4f} | "
                f"f1 {train_metrics['train/f1_macro']:.4f}"
            )

            # ── Validate ──────────────────────────────────────────────────────
            val_metrics = self._evaluate(val_loader, "val")
            val_loss    = val_metrics.get("val/loss", float("inf"))

            # Store both train AND val metrics per epoch so the reporter
            # can plot val curves (not just a flat line from the last epoch)
            all_train_metrics.append({
                "epoch": epoch,
                **train_metrics,
                **val_metrics,   # ← adds val/loss, val/accuracy, val/f1_macro
            })
            elapsed     = time.time() - t0

            self.writer.add_scalar("val/loss",     val_loss,   global_step)
            self.writer.add_scalar("val/accuracy",
                                val_metrics.get("val/accuracy", 0), global_step)

            log_epoch.info(
                f"  Ep {epoch:03d}/{self.cfg.num_epochs} | "
                f"loss {val_loss:.4f} | "
                f"acc {val_metrics.get('val/accuracy', 0):.4f} | "
                f"f1 {val_metrics.get('val/f1_macro', 0):.4f} | "
                f"auc {val_metrics.get('val/roc_auc', float('nan')):.4f} | "
                f"{elapsed:.1f}s"
            )

            # ── Checkpoint ────────────────────────────────────────────────────
            # Every-epoch checkpoint
            if epoch % cfg.save_every == 0:
                self._save(epoch, val_loss,
                        f"epoch_{epoch:03d}.pt",
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                        global_step=global_step,
                        train_history=all_train_metrics)

            # Best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                self._save(epoch, val_loss,
                        "best_model.pt",
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                        global_step=global_step,
                        train_history=all_train_metrics)
                log.info(f"  ★ New best val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                # Still save latest so resume works even on non-best epochs
                self._save(epoch, val_loss,
                        f"epoch_{epoch:03d}.pt",
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                        global_step=global_step,
                        train_history=all_train_metrics)
                if patience_counter >= cfg.early_stop_patience:
                    log.info(f"  Early stopping at epoch {epoch}.")
                    break

        # ── Final test evaluation ─────────────────────────────────────────
        log.info("\\n  Loading best model for test evaluation...")
        self._load("best_model.pt")
        test_metrics = self._evaluate(test_loader, "test")

        # ── Print epoch-by-epoch training summary ─────────────────────────
        log.info(f"\\n{'═'*72}")
        log.info("  TRAINING HISTORY (per epoch)")
        log.info(f"{'═'*72}")
        log.info(
            f"  {'Ep':>3}  {'Train Loss':>10}  {'Train Acc':>9}  "
            f"{'Train F1':>8}"
        )
        log.info(f"  {'─'*40}")
        for em in all_train_metrics:
            log.info(
                f"  {em['epoch']:>3}  "
                f"{em.get('train/loss', 0):>10.4f}  "
                f"{em.get('train/accuracy', 0):>9.4f}  "
                f"{em.get('train/f1_macro', 0):>8.4f}"
            )
        log.info(f"{'═'*72}\\n")

        self.writer.close()

        all_results = {
            "best_val_loss":    best_val_loss,
            "training_history": all_train_metrics,
            **val_metrics,
            **test_metrics,
        }

        # ── Generate plots and PDF report ─────────────────────────────────
        try:
            from src.training.reporter import ExperimentReporter
            reporter = ExperimentReporter(
                results         = all_results,
                output_dir      = str(self.output_dir),
                experiment_name = self.output_dir.name,
                mode            = "single",
                num_classes     = self.num_classes,
            )
            reporter.generate()
        except Exception as e:
            log.warning(f"  Reporter failed (non-fatal): {e}")

        return all_results

    @torch.no_grad()
    def _evaluate(self, loader, split_name: str) -> dict:
        from src.training.metrics import compute_metrics, evaluate_by_audio_type

        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds, all_probs, all_audio_cols = [], [], [], []

        for batch in loader:
            labels = batch["labels"].to(self.device)

            # ── Paired batch (Exp 4) ──────────────────────────────────────
            if "input_values_1" in batch:
                iv1 = torch.nan_to_num(
                    batch["input_values_1"].to(self.device),
                    nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
                iv2 = torch.nan_to_num(
                    batch["input_values_2"].to(self.device),
                    nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
                am1 = batch["attention_mask_1"].to(self.device)
                am2 = batch["attention_mask_2"].to(self.device)
                logits = (
                    self.model(input_values=iv1, attention_mask=am1) +
                    self.model(input_values=iv2, attention_mask=am2)
                ) / 2.0
            else:
                input_values   = batch["input_values"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                input_values = torch.nan_to_num(
                    input_values, nan=0.0, posinf=1.0, neginf=-1.0)
                input_values = input_values.clamp(-10.0, 10.0)
                logits = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                )   # [B, num_classes]

            loss        = self.loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().tolist()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds)
            all_probs.extend(probs)

            # Collect audio_col tags stored by the collate_fn
            batch_cols = batch.get("audio_cols", [])
            if batch_cols:
                all_audio_cols.extend(batch_cols)

        avg_loss = total_loss / max(len(loader), 1)
        metrics  = compute_metrics(
            all_labels,
            all_preds,
            np.array(all_probs),
            self.num_classes,
            split_name,
        )
        metrics[f"{split_name}/loss"] = avg_loss

        # ── Per-audio-type breakdown (test split only, Option B) ───────────
        if split_name == "test" and len(all_audio_cols) == len(all_labels):
            try:
                per_type = evaluate_by_audio_type(
                    all_labels     = all_labels,
                    all_preds      = all_preds,
                    all_probs      = np.array(all_probs),
                    all_audio_cols = all_audio_cols,
                    num_classes    = self.num_classes,
                    split_name     = split_name,
                )
                metrics["test/per_audio_type"] = per_type
            except Exception as e:
                log.warning(f"  Per-audio-type evaluation failed (non-fatal): {e}")

        return metrics

    # ── Internal checkpoint helpers ───────────────────────────────────────

    def _save(self, epoch: int, val_loss: float, filename: str,
            best_val_loss: float = None,
            patience_counter: int = 0,
            global_step: int = 0,
            train_history: list = None):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "epoch":           epoch,
            "val_loss":        val_loss,
            "model":           self.model.state_dict(),
            "optimizer":       self.optimizer.state_dict(),
            "scheduler":       self.scheduler.state_dict()
                            if self.scheduler else None,
            # ── Resume state ──────────────────────────────────────────────
            "best_val_loss":   best_val_loss or val_loss,
            "patience_counter": patience_counter,
            "global_step":     global_step,
            "train_history":   train_history or [],
        }

        torch.save(ckpt, self.output_dir / filename)

        # Always overwrite latest_checkpoint.pt so resume always picks up
        # the most recent epoch regardless of whether it was the best
        torch.save(ckpt, self.output_dir / "latest_checkpoint.pt")

        log.debug(f"  Checkpoint saved → {filename}")

        # ── Prune old epoch checkpoints to save Drive space ───────────────
        keep_n = getattr(self.cfg, "keep_last_n", 2)
        epoch_ckpts = sorted(
            self.output_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        for old_ckpt in epoch_ckpts[:-keep_n]:
            old_ckpt.unlink(missing_ok=True)
            log.debug(f"  Pruned old checkpoint → {old_ckpt.name}")
        
        
    def _load(self, filename: str):
        path = self.output_dir / filename

        if not path.exists():
            log.error(
                f"Checkpoint not found: {path}\n"
                f"This usually means all epochs produced NaN loss so no "
                f"checkpoint was ever saved. Check for NaN in your input segments."
            )
            raise FileNotFoundError(f"No checkpoint at {path}")

        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(ckpt["model"])
        log.info(f"  Loaded → {filename}  "
                f"(epoch {ckpt['epoch']}, "
                f"val_loss {ckpt['val_loss']:.4f})")