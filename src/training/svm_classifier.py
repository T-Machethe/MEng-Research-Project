"""
src/training/svm_classifier.py
─────────────────────────────────────────────────────────────────────────────
SVM classifier trained on frozen wav2vec 2.0 backbone embeddings.

Only runs in finetune mode. The backbone is used purely as a feature
extractor — weights are frozen, no gradients computed. A linear SVM
is trained on the mean-pooled last hidden state ([B, 768]) from every
segment in the train set, then evaluated on val and test.

This provides a second finetune result alongside the end-to-end
neural classifier, allowing direct comparison:

    finetune (neural)  vs  finetune (SVM)  vs  scratch (neural)

If the SVM on frozen pretrained features outperforms scratch, it
validates the transfer learning hypothesis: the pretrained
representations contain clinically relevant structure that
end-to-end fine-tuning alone fails to exploit due to overfitting.

Results are returned in the same dict format as trainer.fit() and
stored under output_dir/svm/.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract mean-pooled backbone embeddings for every segment.

    Parameters
    ----------
    model   : Wav2Vec2Classifier with .backbone attribute
    loader  : DataLoader (standard or paired)
    device  : torch.device

    Returns
    -------
    embeddings : np.ndarray [N, 768]
    labels     : np.ndarray [N]
    """
    model.eval()
    all_emb, all_lbl = [], []

    for batch in loader:
        labels = batch["labels"].to(device)

        if "input_values_1" in batch:
            # Paired batch (Exp4) — average embeddings of both segments
            iv1 = torch.nan_to_num(
                batch["input_values_1"].to(device), nan=0.0).clamp(-10, 10)
            iv2 = torch.nan_to_num(
                batch["input_values_2"].to(device), nan=0.0).clamp(-10, 10)
            am1 = batch["attention_mask_1"].to(device)
            am2 = batch["attention_mask_2"].to(device)
            h1 = model.backbone(input_values=iv1,
                                 attention_mask=am1).last_hidden_state.mean(1)
            h2 = model.backbone(input_values=iv2,
                                 attention_mask=am2).last_hidden_state.mean(1)
            emb = ((h1 + h2) / 2.0).cpu().numpy()
        else:
            iv = torch.nan_to_num(
                batch["input_values"].to(device), nan=0.0).clamp(-10, 10)
            am = batch["attention_mask"].to(device)
            emb = (model.backbone(input_values=iv, attention_mask=am)
                   .last_hidden_state.mean(1).cpu().numpy())

        all_emb.append(emb)
        all_lbl.extend(labels.cpu().tolist())

    return np.vstack(all_emb), np.array(all_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# SVM training + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_svm(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_classes: int,
    output_dir: str,
    device,
    C: float = 1.0,
    kernel: str = "rbf",
) -> dict:
    """
    Extract embeddings from frozen backbone, train SVM, evaluate on all splits.

    Parameters
    ----------
    model        : Wav2Vec2Classifier — backbone used for feature extraction
    *_loader     : DataLoaders for each split
    num_classes  : 2 for binary, 3 for trajectory
    output_dir   : results saved here under svm/
    device       : torch.device
    C            : SVM regularisation (default 1.0)
    kernel       : 'rbf' or 'linear'

    Returns
    -------
    results dict compatible with trainer.fit() format
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report
    )
    from src.training.metrics import evaluate_by_audio_type

    out = Path(output_dir) / "svm"
    out.mkdir(parents=True, exist_ok=True)

    # ── Extract embeddings ─────────────────────────────────────────────────
    log.info("  [SVM] Extracting train embeddings...")
    X_train, y_train = extract_embeddings(model, train_loader, device)
    log.info(f"  [SVM] Train: {X_train.shape[0]} segments, {X_train.shape[1]}d")

    log.info("  [SVM] Extracting val embeddings...")
    X_val, y_val = extract_embeddings(model, val_loader, device)

    log.info("  [SVM] Extracting test embeddings...")
    X_test, y_test = extract_embeddings(model, test_loader, device)

    # ── Standardise ────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Train SVM ──────────────────────────────────────────────────────────
    log.info(f"  [SVM] Training SVM (kernel={kernel}, C={C}, "
             f"class_weight=balanced)...")
    svm = SVC(
        kernel=kernel,
        C=C,
        class_weight="balanced",
        probability=True,   # needed for AUC
        random_state=42,
    )
    svm.fit(X_train, y_train)
    log.info("  [SVM] Training complete.")

    # ── Evaluate each split ────────────────────────────────────────────────
    results = {"mode": "svm"}

    for split, X, y in [("train", X_train, y_train),
                         ("val",   X_val,   y_val),
                         ("test",  X_test,  y_test)]:

        preds = svm.predict(X)
        probs = svm.predict_proba(X)

        acc     = accuracy_score(y, preds)
        f1_mac  = f1_score(y, preds, average="macro", zero_division=0)
        f1_per  = f1_score(y, preds, average=None,    zero_division=0).tolist()
        cm      = confusion_matrix(y, preds).tolist()

        results[f"{split}/accuracy"]       = float(acc)
        results[f"{split}/f1_macro"]       = float(f1_mac)
        results[f"{split}/f1_per_class"]   = f1_per
        results[f"{split}/confusion_matrix"] = cm
        results[f"{split}/all_probs"]      = probs
        results[f"{split}/all_labels"]     = y

        try:
            if num_classes == 2:
                auc = roc_auc_score(y, probs[:, 1])
                results[f"{split}/roc_auc"] = float(auc)
            else:
                auc = roc_auc_score(y, probs,
                                    multi_class="ovr", average="macro")
                results[f"{split}/roc_auc_macro"] = float(auc)
        except Exception as e:
            log.warning(f"  [SVM] AUC failed for {split}: {e}")

        _auc = results.get(f"{split}/roc_auc",
                           results.get(f"{split}/roc_auc_macro"))
        _auc_str = f"{_auc:.4f}" if _auc is not None else "n/a"
        log.info(
            f"  [SVM] [{split.upper():5s}] "
            f"acc {acc:.4f} | f1 {f1_mac:.4f} | auc {_auc_str}"
        )
        log.debug(f"\n{classification_report(y, preds, zero_division=0)}")

    # ── Save results to CSV ────────────────────────────────────────────────
    import pandas as pd
    rows = []
    for split in ["train", "val", "test"]:
        rows.append({
            "split":    split,
            "accuracy": results.get(f"{split}/accuracy"),
            "f1_macro": results.get(f"{split}/f1_macro"),
            "roc_auc":  results.get(f"{split}/roc_auc",
                                     results.get(f"{split}/roc_auc_macro")),
        })
    pd.DataFrame(rows).to_csv(out / "svm_results.csv",
                               index=False, float_format="%.4f")
    log.info(f"  [SVM] Results saved → {out / 'svm_results.csv'}")

    # ── Persist fitted SVM + scaler ────────────────────────────────────────
    # Previously only metrics were saved (svm_results.csv / results_summary
    # .json) — the fitted sklearn objects themselves were never written to
    # disk, so nothing downstream (e.g. a cross-cohort specificity check)
    # could run new inference through the SVM without retraining it. Since
    # SVC(random_state=42) + the deterministic patient_level_split(seed=42)
    # train set make this fit fully reproducible, this is also always safe
    # to regenerate later if a saved artifact is ever lost.
    import joblib
    joblib.dump(
        {"svm": svm, "scaler": scaler, "kernel": kernel, "C": C},
        out / "svm_model.joblib",
    )
    log.info(f"  [SVM] Fitted model + scaler saved → {out / 'svm_model.joblib'}")

    return results