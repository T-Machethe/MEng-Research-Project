import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path as _Path
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    WavLMConfig,
    WavLMModel,
    get_linear_schedule_with_warmup,
)
 
import logging
log = logging.getLogger(__name__)
log_epoch = logging.getLogger("epoch_summary")   # always visible on console
 
from src.training.metrics import compute_metrics
 
class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
 
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
 
    gamma=0 → standard cross entropy.
    gamma=2 → standard setting; down-weights easy examples strongly.
 
    Handles both binary and multiclass. class_weights play the role
    of alpha per class. With gamma>0 the loss focuses gradient updates
    on misclassified or uncertain samples, which is exactly what we
    need given the 75/25 class split and the scratch model's tendency
    to exploit the majority class.
    """
 
    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing
 
    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Standard CE with class weights (handles imbalance baseline)
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t = exp(-CE) — probability assigned to the correct class
        pt      = torch.exp(-ce)
        # Focal term: down-weight easy examples
        focal   = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()
 
 
class Wav2Vec2Classifier(nn.Module):
    """
    wav2vec 2.0 backbone with a custom classification head.
 
    Defined at module level (not inside _build_model) so it is
    picklable on Windows multiprocessing and can be saved/loaded
    cleanly with torch.save / torch.load.
    """
 
    def __init__(self, backbone, hidden_size: int, num_classes: int,
                 proj_size: int = 256):
        super().__init__()
        self.backbone   = backbone
        # MLP head: bottleneck projection → LayerNorm → ReLU → Dropout → classify
        # Replaces the single Linear layer.
        # Benefits:
        #   Scratch: bottleneck forces the model to compress rather than memorise
        #   Finetune: more expressive head without touching pretrained backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, proj_size),
            nn.LayerNorm(proj_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(proj_size, num_classes),
        )
 
    @property
    def classifier(self):
        """Compatibility alias so existing code that reads .classifier still works."""
        return self.head
 
    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        # Mean-pool over time frames [B, T', H] → [B, H]
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.head(hidden)   # [B, num_classes]
 
 
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
        self.model       = self._build_model().to(self._resolve_device(self.cfg.device))
        self.optimizer   = self._build_optimizer()
        self.scheduler   = None
        self.loss_fn     = self._build_loss(class_weights)
        backbone_type = getattr(cfg, "backbone", "wav2vec2").lower().strip()
        init_scale    = 2 ** 14 if backbone_type == "xlsr" else 2 ** 16
        self.scaler   = GradScaler(
                                    enabled       = self.device.type == "cuda",
                                    init_scale    = init_scale,
                                    growth_interval = 200,
                                )
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
        """
        Build backbone + MLP head.

        Supports two backbone families (cfg.backbone):
          "wav2vec2"  ->  facebook/wav2vec2-base-960h
          "wavlm"     ->  microsoft/wavlm-base

        Both share identical API: raw waveform input [B, T],
        hidden_size=768, same encoder.layers for layerwise LR.
        """
        cfg           = self.cfg
        backbone_type = getattr(cfg, "backbone", "wav2vec2").lower().strip()

        BACKBONE_DEFAULTS = {
            "wav2vec2": "facebook/wav2vec2-base-960h",
            "wavlm":    "microsoft/wavlm-base",
            "xlsr":     "facebook/wav2vec2-xls-r-300m",
        }

        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_warning()

        if cfg.mode == "scratch":
            if backbone_type == "xlsr":
                raise ValueError(
                    "XLS-R scratch mode is not supported. "
                    "The value of XLS-R comes from multilingual pretraining. "
                    "Use --mode finetune for XLS-R."
                )
            if backbone_type == "wavlm":
                config   = WavLMConfig(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                )
                backbone = WavLMModel(config)
            else:
                config   = Wav2Vec2Config(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                )
                backbone = Wav2Vec2Model(config)
            log.info(f"  {backbone_type} backbone — randomly initialised (scratch).")

        else:
            # Resolve pretrained checkpoint.
            # If user passed --backbone wavlm without overriding --pretrained,
            # auto-select the WavLM checkpoint.
            pretrained = cfg.pretrained
            if pretrained == "facebook/wav2vec2-base-960h" and backbone_type == "wavlm":
                pretrained = BACKBONE_DEFAULTS["wavlm"]
            if pretrained == "facebook/wav2vec2-base-960h" and backbone_type == "xlsr":
                pretrained = BACKBONE_DEFAULTS["xlsr"]

            log.info(f"  Loading {backbone_type} weights: {pretrained}")

            if backbone_type == "wavlm":
                backbone = WavLMModel.from_pretrained(
                    pretrained,
                    mask_time_prob=0.0,
                    mask_feature_prob=0.0,
                )
            else:
                # Both wav2vec2 and xlsr use Wav2Vec2Model
                backbone = Wav2Vec2Model.from_pretrained(
                    pretrained,
                    mask_time_prob=0.0,
                    mask_feature_prob=0.0,
                )

            log.info(f"  Weights loaded: {pretrained}")

            if cfg.freeze_encoder:
                backbone.feature_extractor.requires_grad_(False)
                backbone.feature_projection.requires_grad_(False)

            for i, layer in enumerate(backbone.encoder.layers):
                if i < cfg.freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        hidden_size = backbone.config.hidden_size
        model       = Wav2Vec2Classifier(backbone, hidden_size, self.num_classes)

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log.info(
            f"  Backbone={backbone_type} | mode={cfg.mode} | "
            f"{n_train/1e6:.1f}M / {n_total/1e6:.1f}M params trainable."
        )
        return model

    def _build_optimizer(self) -> AdamW:
        cfg   = self.cfg
        lr    = cfg.learning_rate
        decay = getattr(cfg, "layerwise_lr_decay", 1.0)
        # XLS-R has 24 transformer layers vs 12 for wav2vec2/WavLM.
        # 0.8^24 ≈ 0.005 — crushes bottom layers. Auto-adjust to 0.9 (0.9^24 ≈ 0.08).
        backbone_type = getattr(cfg, "backbone", "wav2vec2").lower().strip()
        if backbone_type == "xlsr" and decay == 0.8:
            decay = 0.9
            log.info("  XLS-R: layerwise LR decay adjusted to 0.9 (24 layers).")
        wd    = cfg.weight_decay
 
        if cfg.mode == "finetune" and decay < 1.0:
            
            # Layer-wise LR decay: classifier head gets full lr,
            # each lower transformer layer gets lr * decay^n.
            # This prevents destroying pretrained representations
            # in lower layers while allowing the head to adapt freely.
            
            param_groups = []
 
            # Classifier head — full lr
            param_groups.append({
                "params": list(self.model.classifier.parameters()),
                "lr": lr,
            })
 
            # Transformer encoder layers — top to bottom
            enc_layers = self.model.backbone.encoder.layers
            n = len(enc_layers)
            for i, layer in enumerate(reversed(enc_layers)):
                param_groups.append({
                    "params": list(layer.parameters()),
                    "lr": lr * (decay ** (i + 1)),
                })
 
            # Feature projection
            param_groups.append({
                "params": list(
                    self.model.backbone.feature_projection.parameters()
                ),
                "lr": lr * (decay ** (n + 1)),
            })
 
            # Feature extractor — only if not frozen
            
            fe_params = [
                p for p in
                self.model.backbone.feature_extractor.parameters()
                if p.requires_grad
            ]
            if fe_params:
                param_groups.append({
                    "params": fe_params,
                    "lr": lr * (decay ** (n + 2)),
                })
 
            log.info(
                f"  Layer-wise LR decay={decay}: "
                f"head={lr:.2e}, "
                f"top-layer={lr*decay:.2e}, "
                f"bottom-layer={lr*(decay**n):.2e}"
            )
            return AdamW(
                param_groups, weight_decay=wd,
                betas=(0.9, 0.98), eps=1e-8
            )
 
        # Scratch or no decay — uniform lr
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(
            trainable, lr=lr, weight_decay=wd,
            betas=(0.9, 0.98), eps=1e-8
        )
 
    def _build_loss(self, class_weights):
        if class_weights is not None:
            weights = class_weights.to(self.device)
            log.info(f"  Using weighted loss: {weights.tolist()}")
        else:
            weights = None
        ls    = getattr(self.cfg, "label_smoothing", 0.0)
        gamma = getattr(self.cfg, "focal_gamma", 2.0)
        use_focal = getattr(self.cfg, "use_focal_loss", True)
 
        if use_focal:
            log.info(f"  Focal Loss: gamma={gamma}, label_smoothing={ls}")
            return FocalLoss(gamma=gamma, weight=weights,
                             label_smoothing=ls)
        else:
            if ls > 0.0:
                log.info(f"  Label smoothing: {ls}")
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=ls)
 
    def fit(self, train_loader, val_loader, test_loader) -> dict:
        cfg         = self.cfg
        total_steps = len(train_loader) * cfg.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=total_steps,
        )
 
        # Early stopping tracks val_f1 (higher=better) or val_loss (lower=better)
        _stop_metric   = getattr(cfg, "early_stop_metric", "val_f1")
        _higher_better = (_stop_metric == "val_f1")
        best_val_metric  = float("-inf") if _higher_better else float("inf")
        best_val_loss    = float("inf")   # kept for checkpoint compat
        patience_counter = 0
        global_step      = 0
        val_metrics      = {}
        all_train_metrics = []   # ← collect per-epoch train metrics
         
         # ── Resume from checkpoint if one exists ──────────────────────────────
        resume_path = self.output_dir / "latest_checkpoint.pt"
        start_epoch = 1
        if resume_path.exists():
            log.info(f"\n  Resuming from {resume_path}")
            torch.serialization.add_safe_globals([
                np.ndarray,
                np._core.multiarray._reconstruct,
                np.dtype,
                np.dtypes.Float32DType,
            ])
            ckpt = torch.load(
                resume_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(ckpt["model"],strict=False)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") and self.scheduler:
                self.scheduler.load_state_dict(ckpt["scheduler"])
 
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            best_val_metric  = ckpt.get("best_val_metric", best_val_metric)
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
            
            
        # Head warmup bookkeeping (finetune only)
        _warmup_epochs = getattr(cfg, "head_warmup_epochs", 0)
        _warmup_active = (cfg.mode == "finetune" and _warmup_epochs > 0)
        
        for epoch in range(start_epoch, cfg.num_epochs + 1):
 
            # ── Two-phase finetune: head-only warmup ───────────────────────
            # For the first head_warmup_epochs, freeze everything except the
            # classifier head. This prevents the pretrained representations
            # from being destroyed by a randomly-initialised head producing
            # large gradients from epoch 1, which causes the observed class
            # collapse (model predicts only one class from step 0).
            if _warmup_active:
                if epoch <= _warmup_epochs:
                    if epoch == 1:
                        log.info(
                            f"  [Head warmup] Freezing backbone for "
                            f"epochs 1-{_warmup_epochs}. "
                            f"Only classifier head will train."
                        )
                    for p in self.model.backbone.parameters():
                        p.requires_grad_(False)
                else:
                    if epoch == _warmup_epochs + 1:
                        log.info(
                            f"  [Head warmup] Epoch {epoch}: unfreezing backbone "
                            f"with layer-wise LR decay."
                        )
                        # Restore trainability per original freeze_layers config
                        for p in self.model.backbone.parameters():
                            p.requires_grad_(True)
                        if cfg.freeze_encoder:
                            self.model.backbone.feature_extractor.requires_grad_(False)
                            self.model.backbone.feature_projection.requires_grad_(False)
                        for i, layer in enumerate(
                            self.model.backbone.encoder.layers
                        ):
                            if i < cfg.freeze_layers:
                                for p in layer.parameters():
                                    p.requires_grad_(False)
                        # Rebuild optimizer so new param groups have correct LRs
                        self.optimizer = self._build_optimizer()
                        self.scheduler = get_linear_schedule_with_warmup(
                            self.optimizer,
                            num_warmup_steps=cfg.warmup_steps,
                            num_training_steps=total_steps,
                        )
 
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            epoch_loss  = 0.0
            t0          = time.time()

            # Collect predictions during training for metric computation
            train_labels, train_preds, train_probs = [], [], []

            # Snapshot clean weights before epoch — restore on NaN flood
            _epoch_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            nan_steps = 0
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
                    with autocast(enabled=self.device.type == "cuda"):
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
                    with autocast(enabled=self.device.type == "cuda"):
                        logits = self.model(
                            input_values=input_values,
                            attention_mask=attention_mask,
                        )

                loss = self.loss_fn(logits.float(), labels)

                if not torch.isfinite(loss):
                    nan_steps += 1
                    if nan_steps == 1 or nan_steps % 50 == 0:
                        log.warning(
                            f"  NaN/Inf loss at step {global_step} "
                            f"({nan_steps} consecutive NaN steps) — skipping."
                        )
                    self.optimizer.zero_grad()
                    if nan_steps > 50:
                        log.error(
                            f"  NaN flood ({nan_steps} steps) — model weights "
                            f"corrupted by fp16 overflow. Restoring epoch "
                            f"snapshot and halving GradScaler scale."
                        )
                        self.model.load_state_dict(_epoch_state)
                        if self.scaler.is_enabled():
                            new_scale = max(self.scaler.get_scale() / 2.0, 1.0)
                            self.scaler._scale = torch.tensor(
                                new_scale,
                                device=self.device,
                                dtype=torch.float32,
                            )
                            log.info(f"  GradScaler scale reduced to {new_scale}")
                        break
                    continue
                nan_steps = 0

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
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
            if not train_labels:
                log.warning("  No valid training steps this epoch — skipping metrics.")
                train_metrics = {
                    "train/loss": avg_train_loss,
                    "train/accuracy": 0.0,
                    "train/f1_macro": 0.0,
                    "train/roc_auc": float("nan"),
                }
            else:
                train_metrics = compute_metrics(
                    train_labels, train_preds,
                    np.array(train_probs),
                    self.num_classes,
                    split_name="train",
                )
            train_metrics["train/loss"] = avg_train_loss
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
 
            # Fallback to roc_auc_macro for multiclass (Exp3)
            _auc = (val_metrics.get("val/roc_auc")
                    or val_metrics.get("val/roc_auc_macro"))
            _auc_str = f"{_auc:.4f}" if _auc is not None else "nan"
            log_epoch.info(
                f"  Ep {epoch:03d}/{self.cfg.num_epochs} | "
                f"loss {val_loss:.4f} | "
                f"acc {val_metrics.get('val/accuracy', 0):.4f} | "
                f"f1 {val_metrics.get('val/f1_macro', 0):.4f} | "
                f"auc {_auc_str} | "
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
 
            # Best model checkpoint — track val_F1 (or val_loss if configured)
            val_f1_now  = val_metrics.get("val/f1_macro", 0.0)
            if _higher_better:
                current_metric = val_f1_now
                improved       = current_metric > best_val_metric
            else:
                current_metric = val_loss
                improved       = current_metric < best_val_metric
 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
 
            if improved:
                best_val_metric  = current_metric
                patience_counter = 0
                self._save(epoch, val_loss,
                        "best_model.pt",
                        best_val_loss=best_val_loss,
                        best_val_metric=best_val_metric,
                        patience_counter=patience_counter,
                        global_step=global_step,
                        train_history=all_train_metrics)
                metric_label = "val_f1" if _higher_better else "val_loss"
                log.info(f"  ★ New best {metric_label}: {best_val_metric:.4f}")
            else:
                patience_counter += 1
                self._save(epoch, val_loss,
                        f"epoch_{epoch:03d}.pt",
                        best_val_loss=best_val_loss,
                        best_val_metric=best_val_metric,
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
 
        # ── Step 5a: Threshold calibration (binary only) ──────────────────
        val_probs_arr  = val_metrics.get("val/all_probs")
        val_labels_arr = val_metrics.get("val/all_labels")
        test_probs_arr = test_metrics.get("test/all_probs")
        test_labels_arr = test_metrics.get("test/all_labels")
 
        if (val_probs_arr is not None and test_probs_arr is not None
                and self.num_classes == 2):
            try:
                from src.training.eval_utils import calibrate_threshold
                thresh_results = calibrate_threshold(
                    val_probs   = val_probs_arr,
                    val_labels  = val_labels_arr,
                    test_probs  = test_probs_arr,
                    test_labels = test_labels_arr,
                    num_classes = self.num_classes,
                )
                test_metrics["test/threshold_calibration"] = thresh_results
            except Exception as e:
                log.warning(f"  Threshold calibration failed (non-fatal): {e}")
 
        # ── Step 5b: Patient-level vote aggregation ───────────────────────
        # patient_ids are stored by the dataloader if present in the batch
        # For now we use a placeholder — requires patient_ids in test batches
        # (see dataloader extension note in eval_utils.py)
        # This will produce useful results once patient_id is added to batches.
 
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
            "mode":             cfg.mode,
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
        
        # ── SVM on frozen finetune embeddings (finetune mode only) ────────
        if cfg.mode == "finetune" and getattr(cfg, "use_svm", False):
            try:
                from src.training.svm_classifier import train_svm
                log.info("\n  Running SVM classifier on backbone embeddings...")
                svm_results = train_svm(
                    model       = self.model,
                    train_loader= train_loader,
                    val_loader  = val_loader,
                    test_loader = test_loader,
                    num_classes = self.num_classes,
                    output_dir  = str(self.output_dir),
                    device      = self.device,
                    C           = getattr(cfg, "svm_C", 1.0),
                    kernel      = getattr(cfg, "svm_kernel", "rbf"),
                )
                all_results["svm"] = svm_results
            except Exception as e:
                log.warning(f"  SVM failed (non-fatal): {e}")
 
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
 
            loss        = self.loss_fn(logits.float(), labels)
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
        
        all_probs_np = np.array(all_probs, dtype=np.float32)
        row_sums     = all_probs_np.sum(axis=1, keepdims=True)
        all_probs_np = all_probs_np / np.clip(row_sums, 1e-12, None)

        metrics  = compute_metrics(
            all_labels,
            all_preds,
            all_probs_np,
            self.num_classes,
            split_name,
        )
        metrics[f"{split_name}/loss"] = avg_loss

        # Store raw arrays so the reporter can draw actual ROC curves
        metrics[f"{split_name}/all_probs"]  = all_probs_np
        metrics[f"{split_name}/all_labels"] = np.array(all_labels)
 
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
            best_val_metric: float = None,
            patience_counter: int = 0,
            global_step: int = 0,
            train_history: list = None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
 
        ckpt = {
            "epoch":            epoch,
            "val_loss":         val_loss,
            "model":            self.model.state_dict(),
            "optimizer":        self.optimizer.state_dict(),
            "scheduler":        self.scheduler.state_dict()
                             if self.scheduler else None,
            # ── Resume state ─────────────────────────────────────────────
            "best_val_loss":    best_val_loss or val_loss,
            "best_val_metric":  best_val_metric,
            "patience_counter": patience_counter,
            "global_step":      global_step,
            "train_history":    train_history or [],
        }
 
        # Atomic save: write to a temp file then rename.
        # If the process is interrupted mid-write the destination file
        # stays intact and only the temp file is left corrupt.
        def _atomic_save(ckpt, dest):
            tmp = dest.with_suffix(".tmp")
            torch.save(ckpt, tmp)
            tmp.replace(dest)   # atomic on POSIX; overwrites dest only on success

        _atomic_save(ckpt, self.output_dir / filename)
        _atomic_save(ckpt, self.output_dir / "latest_checkpoint.pt")
 
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
        
        # Verify the file is not corrupted before loading.
        # A truncated write leaves a file that cannot be opened as a zip.
        try:
            import zipfile
            with zipfile.ZipFile(str(path), "r") as _:
                pass   # just checks the central directory
        except (zipfile.BadZipFile, Exception) as corrupt_err:
            log.error(
                f"Checkpoint {path} is corrupted ({corrupt_err}). "
                f"This usually means a Drive write was interrupted. "
                f"Falling back to latest_checkpoint.pt if available."
            )
            fallback = self.output_dir / "latest_checkpoint.pt"
            if fallback.exists() and fallback != path:
                try:
                    with zipfile.ZipFile(str(fallback), "r") as _:
                        pass
                    path = fallback
                    log.info(f"  Using fallback checkpoint: {fallback.name}")
                except Exception:
                    pass
            else:
                raise FileNotFoundError(
                    f"Checkpoint {filename} is corrupted and no fallback found."
                ) from corrupt_err

        torch.serialization.add_safe_globals([
            np.ndarray,
            np._core.multiarray._reconstruct,
            np.dtype,
            np.dtypes.Float32DType,
        ])
        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["model"], strict=False)
        log.info(
            f"  Loaded → {path.name}  "
            f"(epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})"
        )