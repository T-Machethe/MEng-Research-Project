"""
New CLI flags:
 
    --audio_type  vowels|sustained|speech|tdu|all
                  Controls which audio columns are included in training.
                  'all' = mixed training (default).
                  Any other value = specialist training on that type only.
 
    --compare_types
                  Run the experiment once per audio type AND once mixed,
                  then print a side-by-side comparison table.
                  Ignores --audio_type when set.
 
Usage examples:
 
    # Mixed training (default)
    python scripts/run_experiment.py --exp 1
 
    # Train only on speech
    python scripts/run_experiment.py --exp 1 --audio_type speech
 
    # Run all audio types and compare
    python scripts/run_experiment.py --exp 1 --compare_types
"""
 
import argparse
import json
import sys
from pathlib import Path as _Path
 
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
 
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
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
 
 
 
def resolve_segment_dir(args):
    if args.segment_dir is not None:
        return args.segment_dir

    # fallback (prevents None crash)
    return str(PROJECT_ROOT / "Data" / "data_final" / "clean_audio")


def resolve_csv_path(args):
    if args.csv_path is not None:
        return args.csv_path

    return str(PROJECT_ROOT / "Data" / "data_final" / "Clinical" / "clinical_all_sessions.csv")

def build_config(args, audio_cols=None) -> ExperimentConfig:
    return ExperimentConfig(
        project_root       = str(PROJECT_ROOT),
        segment_dir        = args.segment_dir,
        csv_path = (args.csv_path
                    if args.csv_path is not None
                    else str(PROJECT_ROOT / "Data" / "data_final" / "Clinical" / "clinical_all_sessions.csv")
                ),
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
        num_workers        = 2, #change to 0 for test on CPU
        # audio_cols is passed separately to the dataset, not stored in cfg
    )
 
 
def run_single(exp_key: str, cfg: ExperimentConfig,
               audio_cols=None, run_label: str = "") -> dict:
    """
    Run one experiment with optional audio column restriction.
 
    audio_cols=None  → mixed (all types)
    audio_cols=[...] → specialist (subset of types)
    """
    ExperimentClass = EXPERIMENT_MAP[exp_key]
 
    # Patch output_dir to include audio type label so runs don't overwrite
    if run_label and run_label != "all":
        cfg.output_dir = str(_Path(cfg.output_dir) / f"exp{exp_key}_{run_label}")
    else:
        cfg.output_dir = str(_Path(cfg.output_dir) / f"exp{exp_key}_mixed")

    experiment = ExperimentClass(cfg)
 
    # Inject audio_cols into prepare() via monkey-patch of the dataset builder
    # This avoids changing the base class interface
    original_prepare = experiment.prepare
 
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
            audio_cols=audio_cols,   # ← injected here
        )
 
    experiment.prepare = patched_prepare
 
    _log.info(f"\n{'-'*60}")
    _log.info(f"  {experiment.name}  [{run_label or 'mixed'}]")
    _log.info(f"  Audio cols: {audio_cols or 'ALL'}")
    _log.info(f"{'-'*60}")
 
    experiment.prepare()
    results = experiment.run()
    experiment.report()
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
        
    return results
 
 
def run_compare_types(exp_key: str, args) -> None:
    """
    Run the experiment once per audio type group + once mixed,
    then print a side-by-side comparison table.
    """
    all_results = {}
 
    for type_name, cols in AUDIO_TYPE_COL_MAP.items():
        _log.info(f"\n{'─'*60}")
        _log.info(f"  Running: {type_name.upper()}")
        _log.info(f"{'─'*60}")
        cfg = build_config(args)
        try:
            results = run_single(exp_key, cfg,
                                  audio_cols=cols,
                                  run_label=type_name)
            all_results[type_name] = results
        except Exception as e:
            _log.error(f"  [{type_name}] Failed: {e}", exc_info=True)
            all_results[type_name] = {}
 
    # ── Save comparison table ─────────────────────────────────────────────
    out_path = (
        _Path(args.output_dir) /
        f"exp{exp_key}_audio_type_comparison.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
 
    # ── Print formatted comparison ────────────────────────────────────────
    _log.info(f"\n{'═'*76}")
    _log.info(f"  AUDIO TYPE COMPARISON — Experiment {exp_key}")
    _log.info(f"{'═'*76}")
    _log.info(
        f"  {'AUDIO TYPE':<14} {'N segs':>8}  "
        f"{'ACC':>7}  {'F1':>7}  {'AUC':>7}"
    )
    _log.info(f"  {'─'*66}")
 
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
    _log.info(f"\n  Full results saved -> {out_path}")
 
def _get_metric(results: dict, key: str):
    """
    Extract a scalar metric from results dict.
    For train metrics, gets the last epoch value from training_history.
    """
    if not results:
        return None

    # Direct key lookup
    if key in results:
        val = results[key]
        if isinstance(val, (int, float)):
            return float(val)

    # Training history — get last epoch value
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
    """
    results = {}

    for mode in ["scratch", "finetune"]:
        _log.info(f"\n{'█'*60}")
        _log.info(f"  MODE: {mode.upper()}")
        _log.info(f"{'█'*60}")

        args.mode = mode
        cfg = build_config(args)

        try:
            res = run_single(exp_key, cfg,
                             audio_cols=None,
                             run_label=mode)
            results[mode] = res
        except Exception as e:
            _log.error(f"  [{mode}] Failed: {e}", exc_info=True)
            results[mode] = {}

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

 
def main():
    parser = argparse.ArgumentParser(
        description="Run sinusitis wav2vec 2.0 experiments with "
                    "per-audio-type support."
    )
    parser.add_argument("--exp", default="1",
                        choices=["1","2","3","4","5","all"])
    parser.add_argument("--mode", default="finetune",
                        choices=["scratch", "finetune"])
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--audio_type", default="all",
                        choices=list(AUDIO_TYPE_COL_MAP.keys()),
                        help="Which audio type to train on. "
                             "'all' = mixed (default).")
    parser.add_argument("--compare_types", action="store_true",
                        help="Run all audio types + mixed and compare results.")
    parser.add_argument("--compare_modes", action="store_true",
                    help="Run scratch + finetune and compare side by side.")
    parser.add_argument("--pretrained",
                        default="facebook/wav2vec2-base-960h")
    parser.add_argument("--freeze_layers", type=int, default=6)
    parser.add_argument("--imbalance", default="weights",
                        choices=["none", "weights", "oversample"])
    parser.add_argument("--batch_size",    type=int,   default=16) # change to 8 for test in CPU
    parser.add_argument("--num_epochs",    type=int,   default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps",  type=int,   default=500)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--segment_dir",
            default=None,
            help="Path to clean audio segments"
        )
    parser.add_argument(
    "--csv_path",
    default=None,
    help="Path to clinical_all_sessions.csv"
)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "results"),
    )
    args = parser.parse_args()
 
    _Path(args.output_dir).mkdir(parents=True, exist_ok=True)
 
    exp_keys = (["1","2","3","4","5"] if args.exp == "all"
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