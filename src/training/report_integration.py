"""
reporter_integration.py
─────────────────────────────────────────────────────────────────────────────
Two patches to integrate the reporter into the existing pipeline.

PATCH 1 → goes into trainer.py   (end of fit() method)
PATCH 2 → goes into run_experiment.py  (end of run_single() and run_compare_modes())
"""

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 1 — Add to the END of trainer.fit(), replacing the final return block
# ═════════════════════════════════════════════════════════════════════════════

"""
Replace the last section of fit() from the comment
"── Final test evaluation" onwards with this:
"""

TRAINER_FIT_TAIL = '''
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
'''


# ═════════════════════════════════════════════════════════════════════════════
# PATCH 2 — Add reporter call in run_experiment.py
# ═════════════════════════════════════════════════════════════════════════════

"""
In run_compare_modes(), replace the final log.info block and json.dump
with this expanded version that also calls the reporter:
"""

RUN_COMPARE_MODES_TAIL = '''
    # ── Side-by-side comparison table ─────────────────────────────────────
    _log.info(f"\\n{'═'*76}")
    _log.info(f"  SCRATCH vs FINE-TUNE COMPARISON — Experiment {exp_key}")
    _log.info(f"{'═'*76}")
    _log.info(
        f"  {'METRIC':<30} {'SCRATCH':>12}  {'FINETUNE':>12}  {'DELTA':>10}"
    )
    _log.info(f"  {'─'*72}")

    metrics_to_compare = [
        ("Train Loss (last epoch)", "train/loss"),
        ("Train Accuracy",          "train/accuracy"),
        ("Train F1 (macro)",        "train/f1_macro"),
        ("Val Loss",                "val/loss"),
        ("Val Accuracy",            "val/accuracy"),
        ("Val F1 (macro)",          "val/f1_macro"),
        ("Val ROC-AUC",             "val/roc_auc"),
        ("Test Accuracy",           "test/accuracy"),
        ("Test F1 (macro)",         "test/f1_macro"),
        ("Test ROC-AUC",            "test/roc_auc"),
    ]

    for label, key in metrics_to_compare:
        scratch_val  = _get_metric(results.get("scratch",  {}), key)
        finetune_val = _get_metric(results.get("finetune", {}), key)

        s_str     = f"{scratch_val:.4f}"  if scratch_val  is not None else "   N/A"
        f_str     = f"{finetune_val:.4f}" if finetune_val is not None else "   N/A"
        delta_str = (f"{finetune_val - scratch_val:+.4f}"
                     if scratch_val is not None and finetune_val is not None
                     else "   N/A")

        _log.info(
            f"  {label:<30} {s_str:>12}  {f_str:>12}  {delta_str:>10}"
        )

    _log.info(f"{'═'*76}")

    # ── Save comparison JSON ───────────────────────────────────────────────
    out_path = _Path(args.output_dir) / f"exp{exp_key}_scratch_vs_finetune.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    _log.info(f"\\n  JSON saved → {out_path}")

    # ── Generate comparison plot and PDF ──────────────────────────────────
    try:
        from src.training.reporter import ExperimentReporter

        # Determine experiment name from map
        exp_names = {
            "1": "exp1_crs_vs_control",
            "2": "exp2_pre_vs_post",
            "3": "exp3_trajectory",
            "4": "exp4_paired_change",
            "5": "exp5_generalisation",
        }
        exp_name = exp_names.get(exp_key, f"exp{exp_key}")

        comparison_dir = _Path(args.output_dir) / f"{exp_name}_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        reporter = ExperimentReporter(
            results         = results,
            output_dir      = str(comparison_dir),
            experiment_name = f"{exp_name} — Scratch vs Fine-tune",
            mode            = "comparison",
            num_classes     = 2,   # update to 3 for exp3
        )
        reporter.generate()

    except Exception as e:
        _log.warning(f"  Comparison reporter failed (non-fatal): {e}")
'''


# ═════════════════════════════════════════════════════════════════════════════
# PATCH 3 — Add reporter call in run_single() after experiment.report()
# ═════════════════════════════════════════════════════════════════════════════

"""
In run_single(), after the line:
    experiment.report()

Add this block:
"""

RUN_SINGLE_REPORTER = '''
    # ── Generate plots and PDF report ─────────────────────────────────────
    # Note: reporter is also called inside trainer.fit() for single-mode runs.
    # This call handles the comparison case where mode="comparison".
    try:
        from src.training.reporter import ExperimentReporter
        reporter = ExperimentReporter(
            results         = results,
            output_dir      = str(_Path(cfg.output_dir) / experiment.name),
            experiment_name = f"{experiment.name} [{run_label or 'mixed'}]",
            mode            = "single",
            num_classes     = experiment.num_classes,
        )
        reporter.generate()
    except Exception as e:
        _log.warning(f"  Reporter failed (non-fatal): {e}")
'''


# ═════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE — where each patch goes
# ═════════════════════════════════════════════════════════════════════════════

INTEGRATION_GUIDE = """
WHERE TO ADD EACH PATCH
═══════════════════════

PATCH 1 → src/training/trainer.py
  Replace everything after the line:
      self.writer.close()
  ...up to and including the final return statement,
  with the contents of TRAINER_FIT_TAIL above.s

PATCH 2 → scripts/run_experiment.py  (inside run_compare_modes)
  Replace everything after the line:
      _log.info(f"{'═'*76}")   ← the closing separator
  ...up to the end of the function,
  with the contents of RUN_COMPARE_MODES_TAIL above.

PATCH 3 → scripts/run_experiment.py  (inside run_single)
  Add the contents of RUN_SINGLE_REPORTER after the line:
      experiment.report()

RESULT FOLDER STRUCTURE AFTER PATCHES
═══════════════════════════════════════
results/
└── exp1_finetune/
    └── exp1_crs_vs_control/
        ├── plots/
        │   ├── 01_loss_curves.png
        │   ├── 02_confusion_matrix.png
        │   ├── 03_roc_auc_curve.png
        │   ├── 04_f1_bar_chart.png
        │   └── 05_scratch_vs_finetune.png  (comparison mode only)
        ├── tables/
        │   └── summary_table.csv
        ├── report.pdf                       ← all plots + table in one PDF
        ├── best_model.pt
        ├── latest_checkpoint.pt
        └── results_summary.json

In Colab, all of this saves directly to Google Drive via --output_dir.
"""

if __name__ == "__main__":
    print(INTEGRATION_GUIDE)