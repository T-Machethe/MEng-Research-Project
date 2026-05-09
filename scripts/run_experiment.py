"""
scripts/run_experiment.py
─────────────────────────────────────────────────────────────────────────────
Single entry point for all five sinusitis wav2vec 2.0 experiments.

CLI flags
─────────
    --exp              : 1 | 2 | 3 | 4 | 5 | all
    --mode             : scratch | finetune  (default: finetune)
    --audio_type       : vowels | sustained | speech | tdu | all (default: all)
    --compare_types    : run all audio types + mixed and compare
    --compare_modes    : run scratch + finetune and compare side by side
    --csv_path         : path to clinical_all_sessions.csv
    --segment_dir      : path to preprocessed .pt segment files
    --output_dir       : where to save results and checkpoints
    --pretrained       : HuggingFace model ID (default: facebook/wav2vec2-base-960h)
    --freeze_layers    : int (default: 6)
    --imbalance        : none | weights | oversample  (default: weights)
    --batch_size       : int (default: 16)
    --num_epochs       : int (default: 30)
    --learning_rate    : float (default: 1e-4)
    --warmup_steps     : int (default: 500)
    --seed             : int (default: 42)

Usage examples
──────────────
    # Single experiment, finetune mode
    python scripts/run_experiment.py --exp 1 --mode finetune --num_epochs 30

    # Compare scratch vs finetune
    python scripts/run_experiment.py --exp 1 --compare_modes --num_epochs 30

    # Audio type comparison
    python scripts/run_experiment.py --exp 1 --compare_types --num_epochs 30

    # All experiments
    python scripts/run_experiment.py --exp all --compare_modes --num_epochs 30

    # Colab with explicit paths
    python scripts/run_experiment.py \
        --exp 1 --compare_modes --num_epochs 30 \
        --segment_dir /content/clean_audio \
        --csv_path /content/drive/MyDrive/Data/data_final/Clinical/clinical_all_sessions.csv \
        --output_dir /content/drive/MyDrive/MSc_Sinusitis_results
"""

import argparse
import json
import sys
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging

def _setup_logging(log_dir: str = None, verbose_console: bool = False):
    """
    Two-handler logging:
      Console  — WARNING by default; only errors + one epoch summary line
                 per epoch (via the dedicated "epoch_summary" logger).
                 Pass verbose_console=True (--verbose flag) to restore
                 full INFO output to the cell.
      File     — DEBUG always; full step-level detail written to
                 OUTPUT_DIR/training.log on Drive.
    """
    fmt_console = logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
 
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)       # capture everything at root level
    root.handlers.clear()
 
    # ── Console handler ───────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)   # ← stdout, not stderr
    ch.setLevel(logging.INFO if verbose_console else logging.WARNING)
    ch.setFormatter(fmt_console)
    root.addHandler(ch)
 
    # ── epoch_summary logger — always prints one line per epoch ──────────
    # trainer.py logs the compact epoch line through this logger so it
    # always appears on the console regardless of the WARNING threshold.
    es = logging.getLogger("epoch_summary")
    es.setLevel(logging.DEBUG)
    es.propagate = False
    es_ch = logging.StreamHandler(sys.stdout)   # ← stdout, not stderr
    es_ch.setLevel(logging.INFO)
    es_ch.setFormatter(fmt_console)
    es.addHandler(es_ch)
 
    # ── File handler — full DEBUG detail on Drive ─────────────────────────
    if log_dir:
        _Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            _Path(log_dir) / "training.log", mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        root.addHandler(fh)
        es.addHandler(fh)              # epoch summaries also go to file
 
_log = logging.getLogger(__name__)

from src.experiments.base import ExperimentConfig
from src.experiments.all_experiments import (
    Exp1CRSvsControl, Exp2PreVsPost, Exp3Trajectory,
    Exp4PairedChange, Exp5Generalisation,
)

EXPERIMENT_MAP = {
    "1": Exp1CRSvsControl,
    "2": Exp2PreVsPost,
    "3": Exp3Trajectory,
    "4": Exp4PairedChange,
    "5": Exp5Generalisation,
}

AUDIO_TYPE_COL_MAP = {
    "vowels":    ["a", "e", "i", "o", "u"],
    "sustained": ["a1", "a2", "a3"],
    "speech":    ["speech"],
    "tdu":       ["agua", "brasero", "dia", "mesa"],
    "all":       None,   # None = use all columns
}

EXP_NAME_MAP = {
    "1": "exp1_crs_vs_control",
    "2": "exp2_pre_vs_post",
    "3": "exp3_trajectory",
    "4": "exp4_paired_change",
    "5": "exp5_generalisation",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_config(args) -> ExperimentConfig:
    return ExperimentConfig(
        project_root       = str(PROJECT_ROOT),
        segment_dir        = (args.segment_dir
                              if args.segment_dir is not None
                              else str(PROJECT_ROOT / "Data" / "data_final" /
                                       "clean_audio")),
        csv_path           = (args.csv_path
                              if args.csv_path is not None
                              else str(PROJECT_ROOT / "Data" / "data_final" /
                                       "Clinical" / "clinical_all_sessions.csv")),
        output_dir         = args.output_dir,
        mode               = args.mode,
        pretrained         = args.pretrained,
        freeze_layers      = args.freeze_layers,
        freeze_encoder     = True,
        batch_size         = args.batch_size,
        num_epochs         = args.num_epochs,
        learning_rate      = args.learning_rate,
        warmup_steps       = args.warmup_steps,
        imbalance_strategy = args.imbalance,
        seed               = args.seed,
        save_every         = args.save_every,
        keep_last_n        = args.keep_last_n,
        num_workers        = 2,   # set to 0 for CPU/Windows testing
        label_smoothing     = args.label_smoothing,
        layerwise_lr_decay  = args.layerwise_lr_decay,
        early_stop_metric   = args.early_stop_metric,
        head_warmup_epochs  = args.head_warmup_epochs,
        use_focal_loss      = args.use_focal_loss,
        focal_gamma         = args.focal_gamma,
        use_svm             = args.use_svm,
        svm_C              = args.svm_C,
        svm_kernel         = args.svm_kernel,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment run
# ─────────────────────────────────────────────────────────────────────────────

def run_single(exp_key: str, cfg: ExperimentConfig,
               audio_cols=None, run_label: str = "") -> dict:
    """
    Run one experiment with optional audio column restriction.

    audio_cols=None  → mixed (all types)
    audio_cols=[...] → specialist (subset of types only)
    """
    ExperimentClass = EXPERIMENT_MAP[exp_key]

    # Set output subdirectory for this run.
    # cfg is a fresh object each call so mutating it here is safe.
    if run_label and run_label != "all":
        cfg.output_dir = str(_Path(cfg.output_dir) / f"exp{exp_key}_{run_label}")
    else:
        cfg.output_dir = str(_Path(cfg.output_dir) / f"exp{exp_key}_mixed")

    experiment = ExperimentClass(cfg)

    # Inject audio_cols into prepare() without changing base class interface
    def patched_prepare():
        from src.pipeline.dataloader import build_experiment_loaders
        from src.training.imbalance import compute_class_weights

        train_df, val_df, test_df, label_fn = experiment.prepare_data()

        experiment.class_weights = compute_class_weights(
            train_df, label_fn, experiment.num_classes
        )

        (experiment.train_loader,
         experiment.val_loader,
         experiment.test_loader) = build_experiment_loaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            segment_dir=str(experiment.segment_dir),
            label_fn=label_fn,
            batch_size=cfg.batch_size,
            imbalance_strategy=cfg.imbalance_strategy,
            class_weights=experiment.class_weights,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            audio_cols=audio_cols,
        )

    experiment.prepare = patched_prepare

    _log.info(f"\n{'-'*60}")
    _log.info(f"  {experiment.name}  [{run_label or 'mixed'}]")
    _log.info(f"  Audio cols: {audio_cols or 'ALL'}")
    _log.info(f"{'-'*60}")

    experiment.prepare()
    results = experiment.run()
    experiment.report()

    # Generate plots and PDF report
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

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Audio type comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_compare_types(exp_key: str, args) -> None:
    """
    Run the experiment once per audio type group + once mixed,
    then print a side-by-side comparison table.
    """
    all_results      = {}
    original_out_dir = args.output_dir   # save before any mutation

    for type_name, cols in AUDIO_TYPE_COL_MAP.items():
        _log.info(f"\n{'─'*60}")
        _log.info(f"  Running: {type_name.upper()}")
        _log.info(f"{'─'*60}")

        args.output_dir = original_out_dir   # reset each iteration
        cfg = build_config(args)

        try:
            results = run_single(exp_key, cfg,
                                 audio_cols=cols,
                                 run_label=type_name)
            all_results[type_name] = results
        except Exception as e:
            _log.error(f"  [{type_name}] Failed: {e}", exc_info=True)
            all_results[type_name] = {}

    args.output_dir = original_out_dir   # restore after loop

    # Save comparison JSON
    out_path = (_Path(args.output_dir) /
                f"exp{exp_key}_audio_type_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    _log.info(f"\n{'═'*76}")
    _log.info(f"  AUDIO TYPE COMPARISON — Experiment {exp_key}")
    _log.info(f"{'═'*76}")
    _log.info(f"  {'AUDIO TYPE':<14} {'ACC':>7}  {'F1':>7}  {'AUC':>7}")
    _log.info(f"  {'─'*44}")

    for type_name, results in all_results.items():
        if not results:
            _log.info(f"  {type_name:<14}  FAILED")
            continue
        acc = results.get("test/accuracy", float("nan"))
        f1  = results.get("test/f1_macro", float("nan"))
        auc = results.get("test/roc_auc",
              results.get("test/roc_auc_macro", float("nan")))
        _log.info(
            f"  {type_name.upper():<14}  "
            f"{acc:>7.4f}  {f1:>7.4f}  {auc:>7.4f}"
        )

    _log.info(f"{'═'*76}")
    _log.info(f"\n  Full results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Scratch vs finetune comparison
# ─────────────────────────────────────────────────────────────────────────────

def _get_metric(results: dict, key: str):
    """
    Extract a scalar metric from a results dict.
    For train/* keys, reads the last epoch from training_history.
    """
    if not results:
        return None

    # Direct key
    if key in results:
        val = results[key]
        if isinstance(val, (int, float)):
            return float(val)

    # Training history — last epoch
    if key.startswith("train/") and "training_history" in results:
        history = results["training_history"]
        if history:
            last = history[-1]
            if key in last:
                return float(last[key])

    return None


def run_compare_modes(exp_key: str, args) -> None:
    """
    Run the same experiment twice — scratch then finetune —
    and print a side-by-side comparison table.

    args.output_dir is saved before the loop and reset after each mode
    so the two runs save to separate subdirectories and do not
    overwrite each other's checkpoints.
    """
    results          = {}
    original_out_dir = args.output_dir   # save BEFORE the loop

    for mode in ["scratch", "finetune"]:
        _log.info(f"\n{'█'*60}")
        _log.info(f"  MODE: {mode.upper()}")
        _log.info(f"{'█'*60}")

        args.mode       = mode
        args.output_dir = original_out_dir   # reset EVERY iteration

        cfg = build_config(args)

        try:
            res = run_single(exp_key, cfg,
                             audio_cols=None,
                             run_label=mode)
            results[mode] = res
        except Exception as e:
            _log.error(f"  [{mode}] Failed: {e}", exc_info=True)
            results[mode] = {}

    args.output_dir = original_out_dir   # restore after loop

    # ── Side-by-side comparison table (outside the for loop) ─────────────
    _log.info(f"\n{'═'*76}")
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
                     if scratch_val  is not None and
                        finetune_val is not None
                     else "   N/A")

        _log.info(
            f"  {label:<30} {s_str:>12}  {f_str:>12}  {delta_str:>10}"
        )

    _log.info(f"{'═'*76}")

    # Save comparison JSON
    out_path = (_Path(args.output_dir) /
                f"exp{exp_key}_scratch_vs_finetune.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    _log.info(f"\n  JSON saved → {out_path}")

    # Generate comparison plots and PDF
    try:
        from src.training.reporter import ExperimentReporter

        exp_name       = EXP_NAME_MAP.get(exp_key, f"exp{exp_key}")
        comparison_dir = (_Path(args.output_dir) /
                          f"{exp_name}_comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # num_classes: 3 for exp3 (trajectory), 2 for all others
        num_classes = 3 if exp_key == "3" else 2

        reporter = ExperimentReporter(
            results         = results,
            output_dir      = str(comparison_dir),
            experiment_name = f"{exp_name} — Scratch vs Fine-tune",
            mode            = "comparison",
            num_classes     = num_classes,
        )
        reporter.generate()

    except Exception as e:
        _log.warning(f"  Comparison reporter failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run sinusitis wav2vec 2.0 experiments."
    )
    parser.add_argument("--exp", default="1",
                        choices=["1", "2", "3", "4", "5", "all"])
    parser.add_argument("--mode", default="finetune",
                        choices=["scratch", "finetune"])
    parser.add_argument("--audio_type", default="all",
                        choices=list(AUDIO_TYPE_COL_MAP.keys()),
                        help="Audio type for specialist training. "
                             "'all' = mixed (default).")
    parser.add_argument("--compare_types", action="store_true",
                        help="Run all audio types and compare results.")
    parser.add_argument("--compare_modes", action="store_true",
                        help="Run scratch + finetune and compare side by side.")
    parser.add_argument("--pretrained",
                        default="facebook/wav2vec2-base-960h")
    parser.add_argument("--freeze_layers", type=int, default=6)
    parser.add_argument("--imbalance", default="weights",
                        choices=["none", "weights", "oversample"])
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--num_epochs",    type=int,   default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps",  type=int,   default=500)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--save_every",   type=int,   default=5,
                        help="Save a numbered checkpoint every N epochs (default 5).")
    parser.add_argument("--keep_last_n",  type=int,   default=2,
                        help="Keep only the N most recent epoch checkpoints on disk (default 2).")
    parser.add_argument("--label_smoothing",    type=float, default=0.1,
                        help="Label smoothing for CrossEntropyLoss (default 0.1).")
    parser.add_argument("--layerwise_lr_decay", type=float, default=0.8,
                        help="Layer-wise LR decay for finetune (default 0.8).")
    parser.add_argument("--early_stop_metric", type=str, default="val_f1",
                        choices=["val_f1", "val_loss"],
                        help="Metric for early stopping (default val_f1).")
    parser.add_argument("--head_warmup_epochs", type=int, default=3,
                        help="Finetune: train head only for N epochs first (default 3).")
    parser.add_argument("--use_focal_loss",  action="store_true", default=True,
                        help="Use Focal Loss instead of CrossEntropy (default True).")
    parser.add_argument("--focal_gamma",     type=float, default=2.0,
                        help="Focal Loss gamma parameter (default 2.0).")
    parser.add_argument("--verbose",       action="store_true", default=False,
                        help="Print full INFO logs to console (default: only epoch summaries).")
    parser.add_argument("--use_svm",    action="store_true", default=False,
                        help="Run SVM on frozen finetune backbone embeddings.")
    parser.add_argument("--svm_C",      type=float, default=1.0,
                        help="SVM regularisation C (default 1.0).")
    parser.add_argument("--svm_kernel", type=str,   default="rbf",
                        choices=["rbf", "linear"],
                        help="SVM kernel (default rbf).")
    parser.add_argument("--segment_dir",   default=None,
                        help="Path to preprocessed .pt segment files.")
    parser.add_argument("--csv_path",      default=None,
                        help="Path to clinical_all_sessions.csv.")
    parser.add_argument("--output_dir",
                        default=str(PROJECT_ROOT / "results"),
                        help="Root directory for results and checkpoints.")
    args = parser.parse_args()

    _Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set up logging — quiet console by default, full detail in training.log
    _setup_logging(
        log_dir=args.output_dir,
        verbose_console=args.verbose,
    )

    exp_keys = (["1", "2", "3", "4", "5"] if args.exp == "all"
                else [args.exp])

    for exp_key in exp_keys:
        if args.compare_modes:
            run_compare_modes(exp_key, args)
        elif args.compare_types:
            run_compare_types(exp_key, args)
        else:
            cols = AUDIO_TYPE_COL_MAP[args.audio_type]
            cfg  = build_config(args)
            run_single(exp_key, cfg,
                       audio_cols=cols,
                       run_label=args.audio_type)


if __name__ == "__main__":
    main()