"""
src/training/reporter.py
─────────────────────────────────────────────────────────────────────────────
Results reporting module — generates all plots and a PDF report
for each experiment run.

Generated outputs per experiment run
─────────────────────────────────────
  plots/
    01_loss_curves.png          Training loss, val loss, LR schedule
    02_confusion_matrix.png     Heatmap with counts and percentages
    03_roc_auc_curve.png        ROC curve with AUC annotation
    04_f1_bar_chart.png         Per-class F1 for train / val / test
    05_scratch_vs_finetune.png  Side-by-side metric comparison bars
  tables/
    summary_table.csv           All metrics in one CSV
  report.pdf                    All plots + tables in one PDF

Usage
─────
Called automatically at the end of each experiment run via reporter.generate().
Can also be called standalone:

    from src.training.reporter import ExperimentReporter
    reporter = ExperimentReporter(results, output_dir, experiment_name)
    reporter.generate()
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Colab / server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG       = "#0D1117"
PANEL    = "#161B22"
BORDER   = "#30363D"
TEXT     = "#E6EDF3"
MUTED    = "#8B949E"
ACCENT   = "#58A6FF"
GREEN    = "#3FB950"
RED      = "#F85149"
ORANGE   = "#D29922"
PURPLE   = "#BC8CFF"

SCRATCH_COLOR  = "#F85149"
FINETUNE_COLOR = "#3FB950"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.6,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
    "legend.fontsize":   8,
    "figure.dpi":        120,
})


class ExperimentReporter:
    """
    Generates all plots and a PDF report for one experiment run.

    Parameters
    ----------
    results         : dict returned by trainer.fit() or run_compare_modes()
    output_dir      : experiment output directory (plots saved here)
    experiment_name : e.g. 'exp1_crs_vs_control'
    mode            : 'single' (one mode) or 'comparison' (scratch vs finetune)
    num_classes     : 2 for binary, 3 for trajectory
    class_names     : list of class label strings
    """

    def __init__(self,
                 results: Dict,
                 output_dir: str,
                 experiment_name: str,
                 mode: str = "single",
                 num_classes: int = 2,
                 class_names: Optional[List[str]] = None):

        self.results         = results
        self.output_dir      = Path(output_dir)
        self.experiment_name = experiment_name
        self.mode            = mode
        self.num_classes     = num_classes
        self.class_names     = class_names or [str(i) for i in range(num_classes)]

        self.plots_dir  = self.output_dir / "plots"
        self.tables_dir = self.output_dir / "tables"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.generated_plots: List[Path] = []

    def generate(self):
        """Generate all plots, tables, and PDF report."""
        log.info(f"\n  Generating results report for {self.experiment_name}...")

        if self.mode == "comparison":
            self._plot_scratch_vs_finetune()
            # Also generate individual plots for each mode
            for mode_key in ["scratch", "finetune"]:
                if mode_key in self.results:
                    sub_reporter = ExperimentReporter(
                        results         = self.results[mode_key],
                        output_dir      = str(self.output_dir / mode_key),
                        experiment_name = f"{self.experiment_name}_{mode_key}",
                        mode            = "single",
                        num_classes     = self.num_classes,
                        class_names     = self.class_names,
                    )
                    sub_reporter.generate()
        else:
            self._plot_loss_curves()
            self._plot_confusion_matrix()
            self._plot_roc_curve()
            self._plot_f1_bars()
            self._save_summary_csv()
            self._build_pdf_report()

        log.info(f"  Report saved → {self.output_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1 — Loss curves + LR schedule
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_loss_curves(self):
        history = self.results.get("training_history", [])
        if not history:
            log.warning("  No training history found — skipping loss curves.")
            return

        epochs     = [h["epoch"]          for h in history]
        train_loss = [h.get("train/loss",     np.nan) for h in history]
        train_acc  = [h.get("train/accuracy", np.nan) for h in history]
        train_f1   = [h.get("train/f1_macro", np.nan) for h in history]

        # Val metrics are now stored per-epoch in training_history
        val_loss   = [h.get("val/loss",       np.nan) for h in history]
        val_acc    = [h.get("val/accuracy",   np.nan) for h in history]
        val_f1     = [h.get("val/f1_macro",   np.nan) for h in history]
        has_val    = any(not np.isnan(v) for v in val_loss)

        fig = plt.figure(figsize=(16, 10), facecolor=BG)
        fig.suptitle(
            f"{self.experiment_name}\nTraining Curves",
            fontsize=13, color=TEXT, y=0.98, fontweight="bold"
        )

        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35,
                               figure=fig)

        # ── Loss ──────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, train_loss, color=ACCENT, linewidth=2,
                 marker="o", markersize=4, label="Train Loss")
        if has_val:
            ax1.plot(epochs, val_loss, color=ORANGE, linewidth=2,
                     marker="s", markersize=4, linestyle="--", label="Val Loss")
        else:
            ax1.axhline(self.results.get("val/loss", np.nan),
                        color=ORANGE, linewidth=1.5, linestyle="--",
                        label=f"Best Val Loss: {self.results.get('val/loss', 0):.4f}")
        ax1.set_title("Loss per Epoch", pad=8)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        ax1.set_facecolor(PANEL)

        # ── Accuracy ──────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, train_acc, color=GREEN, linewidth=2,
                 marker="s", markersize=4, label="Train Accuracy")
        if has_val:
            ax2.plot(epochs, val_acc, color=ORANGE, linewidth=2,
                     marker="^", markersize=4, linestyle="--", label="Val Accuracy")
        else:
            ax2.axhline(self.results.get("val/accuracy", np.nan),
                        color=ORANGE, linewidth=1.5, linestyle="--",
                        label=f"Val Acc: {self.results.get('val/accuracy', 0):.4f}")
        ax2.axhline(self.results.get("test/accuracy", np.nan),
                    color=PURPLE, linewidth=1.5, linestyle=":",
                    label=f"Test Acc: {self.results.get('test/accuracy', 0):.4f}")
        ax2.set_title("Accuracy per Epoch", pad=8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True)
        ax2.set_facecolor(PANEL)

        # ── F1 ────────────────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, train_f1, color=PURPLE, linewidth=2,
                 marker="^", markersize=4, label="Train F1 (macro)")
        if has_val:
            ax3.plot(epochs, val_f1, color=ORANGE, linewidth=2,
                     marker="o", markersize=4, linestyle="--", label="Val F1 (macro)")
        else:
            ax3.axhline(self.results.get("val/f1_macro", np.nan),
                        color=ORANGE, linewidth=1.5, linestyle="--",
                        label=f"Val F1: {self.results.get('val/f1_macro', 0):.4f}")
        ax3.axhline(self.results.get("test/f1_macro", np.nan),
                    color=PURPLE, linewidth=1.5, linestyle=":",
                    label=f"Test F1: {self.results.get('test/f1_macro', 0):.4f}")
        ax3.set_title("F1 Macro per Epoch", pad=8)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1 Score")
        ax3.set_ylim(0, 1.05)
        ax3.legend()
        ax3.grid(True)
        ax3.set_facecolor(PANEL)

        # ── Summary text box ──────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        ax4.set_facecolor(PANEL)

        summary_lines = [
            f"FINAL RESULTS SUMMARY",
            f"{'─'*28}",
            f"  Val  Loss    : {self.results.get('val/loss', 0):.4f}",
            f"  Val  Acc     : {self.results.get('val/accuracy', 0):.4f}",
            f"  Val  F1      : {self.results.get('val/f1_macro', 0):.4f}",
            f"  Val  AUC     : {self.results.get('val/roc_auc', 0):.4f}",
            f"{'─'*28}",
            f"  Test Loss    : {self.results.get('test/loss', 0):.4f}",
            f"  Test Acc     : {self.results.get('test/accuracy', 0):.4f}",
            f"  Test F1      : {self.results.get('test/f1_macro', 0):.4f}",
            f"  Test AUC     : {self.results.get('test/roc_auc', 0):.4f}",
            f"{'─'*28}",
            f"  Epochs       : {len(epochs)}",
            f"  Best Val Loss: {self.results.get('best_val_loss', 0):.4f}",
        ]
        ax4.text(0.05, 0.95, "\n".join(summary_lines),
                 transform=ax4.transAxes,
                 fontsize=9, verticalalignment="top",
                 fontfamily="monospace", color=TEXT,
                 bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=BG, edgecolor=BORDER))

        path = self.plots_dir / "01_loss_curves.png"
        plt.savefig(path, bbox_inches="tight", facecolor=BG)
        plt.show()
        plt.close(fig)
        self.generated_plots.append(path)
        log.info(f"  Saved → {path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2 — Confusion matrix
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_confusion_matrix(self):
        cm_test = self.results.get("test/confusion_matrix")
        cm_val  = self.results.get("val/confusion_matrix")

        if not cm_test:
            log.warning("  No confusion matrix found — skipping.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
        fig.suptitle(f"{self.experiment_name}\nConfusion Matrices",
                     fontsize=13, color=TEXT, y=1.02, fontweight="bold")

        for ax, cm_data, split in zip(axes,
                                       [cm_val, cm_test],
                                       ["Validation", "Test"]):
            if not cm_data:
                ax.set_visible(False)
                continue

            cm = np.array(cm_data)
            total = cm.sum()
            cm_pct = cm / max(total, 1) * 100

            im = ax.imshow(cm, cmap="Blues", aspect="auto")

            ax.set_xticks(range(self.num_classes))
            ax.set_yticks(range(self.num_classes))
            ax.set_xticklabels(self.class_names, fontsize=9)
            ax.set_yticklabels(self.class_names, fontsize=9)
            ax.set_xlabel("Predicted Label", fontsize=9)
            ax.set_ylabel("True Label", fontsize=9)
            ax.set_title(f"{split} Set", fontsize=10, color=TEXT, pad=8)
            ax.set_facecolor(PANEL)

            # Annotate cells with count and percentage
            thresh = cm.max() / 2.0
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    color = "white" if cm[i, j] < thresh else BG
                    ax.text(j, i,
                            f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                            ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

            plt.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(
                labelcolor=TEXT, labelsize=7
            )

        plt.tight_layout()
        path = self.plots_dir / "02_confusion_matrix.png"
        plt.savefig(path, bbox_inches="tight", facecolor=BG)
        plt.show()
        plt.close(fig)
        self.generated_plots.append(path)
        log.info(f"  Saved → {path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3 — ROC-AUC curve
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_roc_curve(self):
        """
        Plot ROC curve using stored probabilities.
        Requires all_probs and all_labels to be stored in results.
        Falls back to AUC annotation only if raw probs not available.
        """
        val_auc  = self.results.get("val/roc_auc")
        test_auc = self.results.get("test/roc_auc")

        fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
        fig.suptitle(f"{self.experiment_name}\nROC-AUC Curve",
                     fontsize=13, color=TEXT, y=1.01, fontweight="bold")

        ax.set_facecolor(PANEL)

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1,
                linestyle="--", label="Random classifier (AUC=0.50)")

        # Plot curves if raw data available, otherwise show AUC as annotation
        val_probs  = self.results.get("val/all_probs")
        val_labels = self.results.get("val/all_labels")
        test_probs = self.results.get("test/all_probs")
        test_labels = self.results.get("test/all_labels")

        plotted = False
        for probs, labels, color, split, auc in [
            (val_probs,  val_labels,  ORANGE, "Val",  val_auc),
            (test_probs, test_labels, ACCENT, "Test", test_auc),
        ]:
            if probs and labels and self.num_classes == 2:
                try:
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(labels,
                                            np.array(probs)[:, 1])
                    ax.plot(fpr, tpr, color=color, linewidth=2,
                            label=f"{split} ROC (AUC={auc:.4f})")
                    plotted = True
                except Exception:
                    pass

        if not plotted:
            # No raw probs stored — show AUC as text annotation
            y_pos = 0.65
            for split, auc, color in [
                ("Validation", val_auc,  ORANGE),
                ("Test",       test_auc, ACCENT),
            ]:
                if auc is not None:
                    ax.annotate(
                        f"{split} AUC = {auc:.4f}",
                        xy=(0.5, y_pos),
                        xycoords="axes fraction",
                        fontsize=13, color=color, fontweight="bold",
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor=BG, edgecolor=color,
                                  linewidth=1.5)
                    )
                    y_pos -= 0.12

            ax.text(0.5, 0.45,
                    "Note: Raw probability arrays not stored.\n"
                    "Store val/all_probs and test/all_probs in\n"
                    "results dict for full ROC curve.",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=8, color=MUTED,
                    style="italic")

        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
        ax.grid(True)

        path = self.plots_dir / "03_roc_auc_curve.png"
        plt.savefig(path, bbox_inches="tight", facecolor=BG)
        plt.show()
        plt.close(fig)
        self.generated_plots.append(path)
        log.info(f"  Saved → {path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 4 — Per-class F1 bar chart
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_f1_bars(self):
        splits = {}
        for split in ["train", "val", "test"]:
            key = f"{split}/f1_per_class"
            if key in self.results:
                splits[split] = self.results[key]
            elif "training_history" in self.results:
                history = self.results["training_history"]
                if history:
                    last = history[-1]
                    if key in last:
                        splits[split] = last[key]

        if not splits:
            log.warning("  No per-class F1 data — skipping F1 bar chart.")
            return

        fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 6),
                                  facecolor=BG)
        if len(splits) == 1:
            axes = [axes]

        fig.suptitle(f"{self.experiment_name}\nPer-Class F1 Score",
                     fontsize=13, color=TEXT, y=1.02, fontweight="bold")

        colors = [ACCENT, GREEN, ORANGE, PURPLE, RED]
        split_colors = {"train": ACCENT, "val": ORANGE, "test": GREEN}

        for ax, (split, f1_vals) in zip(axes, splits.items()):
            x       = np.arange(self.num_classes)
            color   = split_colors.get(split, ACCENT)
            bars    = ax.bar(x, f1_vals, color=color, alpha=0.8,
                             edgecolor=BORDER, linewidth=0.8)

            # Annotate bars
            for bar, val in zip(bars, f1_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=9, color=TEXT, fontweight="bold")

            # Macro F1 reference line
            macro = np.mean(f1_vals)
            ax.axhline(macro, color=RED, linewidth=1.5,
                       linestyle="--",
                       label=f"Macro avg: {macro:.3f}")

            ax.set_title(f"{split.upper()} Split", fontsize=10,
                         color=color, pad=6)
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, fontsize=9)
            ax.set_ylabel("F1 Score", fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y")
            ax.set_facecolor(PANEL)

        plt.tight_layout()
        path = self.plots_dir / "04_f1_bar_chart.png"
        plt.savefig(path, bbox_inches="tight", facecolor=BG)
        plt.show()
        plt.close(fig)
        self.generated_plots.append(path)
        log.info(f"  Saved → {path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 5 — Scratch vs Finetune comparison
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_scratch_vs_finetune(self):
        scratch  = self.results.get("scratch",  {})
        finetune = self.results.get("finetune", {})

        if not scratch and not finetune:
            log.warning("  No scratch/finetune results — skipping comparison.")
            return

        metrics = [
            ("Test Accuracy",  "test/accuracy"),
            ("Test F1 Macro",  "test/f1_macro"),
            ("Test ROC-AUC",   "test/roc_auc"),
            ("Val Accuracy",   "val/accuracy"),
            ("Val F1 Macro",   "val/f1_macro"),
            ("Val ROC-AUC",    "val/roc_auc"),
        ]

        labels  = [m[0] for m in metrics]
        s_vals  = [scratch.get(m[1],  0) or 0 for m in metrics]
        ft_vals = [finetune.get(m[1], 0) or 0 for m in metrics]

        x     = np.arange(len(labels))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
        fig.suptitle(
            f"{self.experiment_name}\nScratch vs Fine-tune Comparison",
            fontsize=13, color=TEXT, y=1.02, fontweight="bold"
        )

        # ── Grouped bar chart ──────────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor(PANEL)

        b1 = ax.bar(x - width/2, s_vals,  width, label="Scratch",
                    color=SCRATCH_COLOR,  alpha=0.8, edgecolor=BORDER)
        b2 = ax.bar(x + width/2, ft_vals, width, label="Fine-tune",
                    color=FINETUNE_COLOR, alpha=0.8, edgecolor=BORDER)

        for bar, val in zip(b1, s_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT)
        for bar, val in zip(b2, ft_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title("Metric Comparison", fontsize=10, color=TEXT, pad=6)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y")

        # ── Delta bar chart (finetune - scratch) ──────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor(PANEL)

        deltas = [ft - s for ft, s in zip(ft_vals, s_vals)]
        bar_colors = [GREEN if d >= 0 else RED for d in deltas]

        bars = ax2.bar(x, deltas, color=bar_colors, alpha=0.85,
                       edgecolor=BORDER)
        ax2.axhline(0, color=MUTED, linewidth=1)

        for bar, val in zip(bars, deltas):
            ypos = bar.get_height() + 0.002 if val >= 0 \
                   else bar.get_height() - 0.012
            ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                     f"{val:+.3f}", ha="center", va="bottom",
                     fontsize=8, color=TEXT, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Fine-tune − Scratch", fontsize=9)
        ax2.set_title("Delta (positive = fine-tune wins)",
                      fontsize=10, color=TEXT, pad=6)
        ax2.grid(True, axis="y")

        plt.tight_layout()
        path = self.plots_dir / "05_scratch_vs_finetune.png"
        plt.savefig(path, bbox_inches="tight", facecolor=BG)
        plt.show()
        plt.close(fig)
        self.generated_plots.append(path)
        log.info(f"  Saved → {path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary CSV
    # ─────────────────────────────────────────────────────────────────────────

    def _save_summary_csv(self):
        rows = []
        for split in ["train", "val", "test"]:
            row = {"split": split}
            for metric in ["loss", "accuracy", "f1_macro", "roc_auc"]:
                key = f"{split}/{metric}"
                val = self.results.get(key)
                if val is None and "training_history" in self.results:
                    history = self.results["training_history"]
                    if history:
                        val = history[-1].get(key)
                row[metric] = val
            rows.append(row)

        df  = pd.DataFrame(rows)
        path = self.tables_dir / "summary_table.csv"
        df.to_csv(path, index=False, float_format="%.4f")
        log.info(f"  Saved → {path.name}")

        # Also print as formatted table
        log.info(f"\n{'═'*60}")
        log.info(f"  SUMMARY TABLE — {self.experiment_name}")
        log.info(f"{'═'*60}")
        log.info(f"  {df.to_string(index=False)}")
        log.info(f"{'═'*60}\n")

        # ── Per-audio-type breakdown (if present) ─────────────────────────
        per_type = self.results.get("test/per_audio_type")
        if per_type:
            rows_pt = []
            for audio_type, m in per_type.items():
                rows_pt.append({
                    "audio_type":  audio_type,
                    "n_segments":  m.get("n_segments", ""),
                    "accuracy":    m.get("accuracy"),
                    "f1_macro":    m.get("f1_macro"),
                    "roc_auc":     m.get("roc_auc"),
                    "note":        m.get("note", ""),
                })
            df_pt = pd.DataFrame(rows_pt)
            pt_path = self.tables_dir / "per_audio_type.csv"
            df_pt.to_csv(pt_path, index=False, float_format="%.4f")
            log.info(f"  Saved → {pt_path.name}")

            log.info(f"\n{'═'*72}")
            log.info(f"  PER AUDIO TYPE — {self.experiment_name}  [TEST]")
            log.info(f"{'═'*72}")
            log.info(f"  {df_pt.to_string(index=False)}")
            log.info(f"{'═'*72}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # PDF Report
    # ─────────────────────────────────────────────────────────────────────────

    def _build_pdf_report(self):
        if not self.generated_plots:
            log.warning("  No plots to include in PDF.")
            return

        pdf_path = self.output_dir / "report.pdf"

        with PdfPages(str(pdf_path)) as pdf:

            # ── Cover page ────────────────────────────────────────────────
            fig = plt.figure(figsize=(11.7, 8.3), facecolor=BG)
            ax  = fig.add_subplot(111)
            ax.axis("off")
            ax.set_facecolor(BG)

            # Title block
            ax.text(0.5, 0.82,
                    "SINUSITIS VOICE ANALYSIS",
                    transform=ax.transAxes,
                    fontsize=22, color=ACCENT,
                    ha="center", fontweight="bold",
                    fontfamily="monospace")

            ax.text(0.5, 0.72,
                    self.experiment_name.replace("_", " ").upper(),
                    transform=ax.transAxes,
                    fontsize=14, color=TEXT,
                    ha="center", fontfamily="monospace")

            ax.text(0.5, 0.62,
                    "wav2vec 2.0 Clinical Audio Classification",
                    transform=ax.transAxes,
                    fontsize=11, color=MUTED,
                    ha="center", fontfamily="monospace")

            # Divider
            ax.axhline(0.56, color=BORDER, linewidth=1,
                       xmin=0.1, xmax=0.9)

            # Key metrics block
            metrics_text = [
                f"Test Accuracy : {self.results.get('test/accuracy', 0):.4f}",
                f"Test F1 Macro : {self.results.get('test/f1_macro',  0):.4f}",
                f"Test ROC-AUC  : {self.results.get('test/roc_auc',   0):.4f}",
                f"Val  Accuracy : {self.results.get('val/accuracy',   0):.4f}",
                f"Val  F1 Macro : {self.results.get('val/f1_macro',   0):.4f}",
                f"Val  ROC-AUC  : {self.results.get('val/roc_auc',    0):.4f}",
            ]
            ax.text(0.5, 0.48,
                    "\n".join(metrics_text),
                    transform=ax.transAxes,
                    fontsize=11, color=GREEN,
                    ha="center", fontfamily="monospace",
                    linespacing=1.8)

            pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
            plt.close(fig)

            # ── One plot per page ─────────────────────────────────────────
            plot_titles = [
                "Training Curves — Loss, Accuracy, F1 per Epoch",
                "Confusion Matrix — Validation and Test Sets",
                "ROC-AUC Curve",
                "Per-Class F1 Score — Train / Val / Test",
                "Scratch vs Fine-tune Comparison",
            ]

            for plot_path, title in zip(self.generated_plots,
                                         plot_titles):
                if not plot_path.exists():
                    continue

                img = plt.imread(str(plot_path))
                fig = plt.figure(figsize=(11.7, 8.3), facecolor=BG)
                ax  = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title, fontsize=11, color=TEXT, pad=10,
                             fontfamily="monospace")
                fig.patch.set_facecolor(BG)
                pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
                plt.close(fig)

            # ── Summary table page ────────────────────────────────────────
            csv_path = self.tables_dir / "summary_table.csv"
            if csv_path.exists():
                df  = pd.read_csv(csv_path)
                fig = plt.figure(figsize=(11.7, 4), facecolor=BG)
                ax  = fig.add_subplot(111)
                ax.axis("off")
                ax.set_facecolor(BG)
                ax.set_title("Summary Metrics Table",
                             fontsize=11, color=TEXT, pad=10,
                             fontfamily="monospace")

                col_labels = list(df.columns)
                cell_text  = df.values.tolist()

                tbl = ax.table(
                    cellText=cell_text,
                    colLabels=col_labels,
                    loc="center",
                    cellLoc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)
                tbl.scale(1.2, 2.0)

                for (row, col), cell in tbl.get_celld().items():
                    cell.set_facecolor(PANEL if row > 0 else BG)
                    cell.set_edgecolor(BORDER)
                    cell.set_text_props(color=TEXT if row > 0 else ACCENT,
                                        fontfamily="monospace",
                                        fontweight="bold" if row == 0
                                        else "normal")

                pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
                plt.close(fig)

            # ── Per-audio-type table page (if available) ─────────────────
            pt_csv = self.tables_dir / "per_audio_type.csv"
            if pt_csv.exists():
                df_pt = pd.read_csv(pt_csv)
                # Drop note column if all empty
                if "note" in df_pt.columns and df_pt["note"].isna().all():
                    df_pt = df_pt.drop(columns=["note"])

                fig = plt.figure(figsize=(11.7, max(4, len(df_pt) * 0.45 + 1.5)),
                                 facecolor=BG)
                ax  = fig.add_subplot(111)
                ax.axis("off")
                ax.set_facecolor(BG)
                ax.set_title("Per Audio Type Results — Test Set",
                             fontsize=11, color=TEXT, pad=10,
                             fontfamily="monospace")

                # Format floats
                for col in ["accuracy", "f1_macro", "roc_auc"]:
                    if col in df_pt.columns:
                        df_pt[col] = df_pt[col].apply(
                            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                        )

                tbl = ax.table(
                    cellText  = df_pt.values.tolist(),
                    colLabels = list(df_pt.columns),
                    loc       = "center",
                    cellLoc   = "center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1.2, 1.8)

                for (row, col), cell in tbl.get_celld().items():
                    cell.set_facecolor(PANEL if row > 0 else BG)
                    cell.set_edgecolor(BORDER)
                    cell.set_text_props(
                        color=TEXT if row > 0 else ACCENT,
                        fontfamily="monospace",
                        fontweight="bold" if row == 0 else "normal",
                    )

                pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
                plt.close(fig)

        log.info(f"  PDF report saved → {pdf_path}")