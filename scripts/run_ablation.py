"""
run_ablation.py
─────────────────────────────────────────────────────────────────────────────
Two-phase ablation study on Experiment 1 (binary CRS detection) using
XLS-R-finetune — the best-performing finetune model on Exp1 (F1=0.696,
AUC=0.791).

Three factors varied one at a time (all others at current defaults):
  A. freeze — frozen Transformer layers: 0, 2, 4 (current), 6, 8, all-frozen
  B. loss   — CrossEntropy vs FocalLoss γ=1/2(current)/3
  C. decay  — layerwise LR decay λ=1.0/0.9(current)/0.8/0.7

PHASE 1 (--run):    train all variants, save results_summary.json per variant
PHASE 2 (--report): read ablation_results.json, print tables, write LaTeX

Usage
─────
    # Phase 1 — run one factor per Colab session (~40 min per factor on T4)
    python scripts/run_ablation.py --run \\
        --factor freeze \\
        --segment_dir /content/clean_audio_3s \\
        --csv_path    /path/to/clinical_all_sessions.csv \\
        --output_dir  /content/drive/MyDrive/MSc_Sinusitis_results/ablation \\
        --num_epochs  8

    # Phase 2 — generate LaTeX tables after all factors complete
    python scripts/run_ablation.py --report \\
        --output_dir  /content/drive/MyDrive/MSc_Sinusitis_results/ablation \\
        --output_tex  /content/drive/MyDrive/thesis_outputs/ablation_tables.tex

    # Dry run — print commands without using GPU
    python scripts/run_ablation.py --run --factor all --dry_run ...
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Defaults (current XLS-R finetune config) ──────────────────────────────────
DEFAULTS = dict(
    backbone           = "xlsr",
    mode               = "finetune",
    exp                = "1",
    num_epochs         = 8,
    batch_size         = 8,           # XLS-R memory constraint
    learning_rate      = 1e-5,
    freeze_layers      = 4,           # current
    focal_gamma        = 2.0,         # current
    layerwise_lr_decay = 0.9,         # current (auto-adjusted for 24 layers)
    label_smoothing    = 0.1,
    head_warmup_epochs = 1,
    early_stop_patience= 3,
    early_stop_metric  = "val_f1",
    imbalance          = "weights",
    warmup_steps       = 100,
    use_svm            = True,
)

# ── Ablation variants ──────────────────────────────────────────────────────────
ABLATION_GROUPS = {
    "freeze": [
        ("freeze=0  (all layers train)", {"freeze_layers": 0}),
        ("freeze=2",                     {"freeze_layers": 2}),
        ("freeze=4  [CURRENT]",          {"freeze_layers": 4}),
        ("freeze=6",                     {"freeze_layers": 6}),
        ("freeze=8",                     {"freeze_layers": 8}),
        ("freeze=all  (SVM probe only)", {"freeze_layers": 24, "use_svm": True}),
    ],
    "loss": [
        ("CrossEntropy",                 {"focal_gamma": 0.0, "label_smoothing": 0.0}),
        ("FocalLoss γ=1",                {"focal_gamma": 1.0}),
        ("FocalLoss γ=2  [CURRENT]",     {"focal_gamma": 2.0}),
        ("FocalLoss γ=3",                {"focal_gamma": 3.0}),
    ],
    "decay": [
        ("λ=1.0  (uniform LR)",          {"layerwise_lr_decay": 1.0}),
        ("λ=0.9  [CURRENT]",             {"layerwise_lr_decay": 0.9}),
        ("λ=0.8",                        {"layerwise_lr_decay": 0.8}),
        ("λ=0.7",                        {"layerwise_lr_decay": 0.7}),
    ],
}

CURRENT_MARKER = "[CURRENT]"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Training
# ══════════════════════════════════════════════════════════════════════════════

def build_cmd(segment_dir, csv_path, run_out, overrides):
    cfg = {**DEFAULTS, **overrides}
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
        "--exp",                str(cfg["exp"]),
        "--backbone",           cfg["backbone"],
        "--mode",               cfg["mode"],
        "--num_epochs",         str(cfg["num_epochs"]),
        "--batch_size",         str(cfg["batch_size"]),
        "--learning_rate",      str(cfg["learning_rate"]),
        "--freeze_layers",      str(cfg["freeze_layers"]),
        "--focal_gamma",        str(cfg["focal_gamma"]),
        "--layerwise_lr_decay", str(cfg["layerwise_lr_decay"]),
        "--label_smoothing",    str(cfg["label_smoothing"]),
        "--head_warmup_epochs", str(cfg["head_warmup_epochs"]),
        "--early_stop_patience",str(cfg["early_stop_patience"]),
        "--early_stop_metric",  cfg["early_stop_metric"],
        "--imbalance",          cfg["imbalance"],
        "--warmup_steps",       str(cfg["warmup_steps"]),
        "--segment_dir",        segment_dir,
        "--csv_path",           csv_path,
        "--output_dir",         str(run_out),
    ]
    if cfg.get("use_svm"):
        cmd.append("--use_svm")
    return cmd


def _safe_label(label: str) -> str:
    return re.sub(r"_+", "_",
        label.replace(" ","_").replace("=","_")
             .replace("[","").replace("]","")
             .replace("(","").replace(")","")
             .replace("/","_").replace("γ","g")
             .replace("λ","L")).strip("_")


def _load_result(label, overrides, run_out, status="complete"):
    res_path = run_out / "results_summary.json"
    with open(res_path) as f:
        res = json.load(f)
    return {
        "label":     label,
        "overrides": overrides,
        "status":    status,
        "val_f1":    res.get("val/f1_macro"),
        "test_f1":   res.get("test/f1_macro"),
        "test_auc":  res.get("test/roc_auc"),
        "test_acc":  res.get("test/accuracy"),
        "svm_f1":    res.get("svm", {}).get("test/f1_macro"),
        "svm_auc":   res.get("svm", {}).get("test/roc_auc"),
        "epochs":    res.get("epochs_trained"),
    }


def run_variant(label, overrides, segment_dir, csv_path, factor_dir, dry_run):
    # factor_dir = ablation/freeze  →  run_out = ablation/freeze/freeze_4/
    folder  = VARIANT_FOLDERS.get(label, _safe_label(label))
    run_out = factor_dir / folder
    res_path = run_out / "results_summary.json"

    # ── Fast skip: results already on disk ───────────────────────────────
    if res_path.exists():
        print(f"\n  ↩  SKIP (already complete): {label}")
        print(f"     {run_out.name}")
        return _load_result(label, overrides, run_out, status="skipped")

    print(f"\n  {'─'*58}")
    print(f"  {label}")
    print(f"  Overrides : {overrides}")
    print(f"  Output    : {run_out}")

    if dry_run:
        print("  [DRY RUN — not executed]")
        return {"label": label, "overrides": overrides, "status": "dry_run"}

    cmd = build_cmd(segment_dir, csv_path, run_out, overrides)

    # Capture stderr so failures print the actual error rather than silent code 1
    result = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"\n  ✗  Variant failed (exit {result.returncode}): {label}")
        if result.stderr and result.stderr.strip():
            # Print last 40 lines of stderr — enough to see the traceback
            lines = result.stderr.strip().splitlines()
            print("\n  --- stderr (last 40 lines) ---")
            for ln in lines[-40:]:
                print(f"  {ln}")
            print("  --- end stderr ---\n")
        return {"label": label, "overrides": overrides, "status": "failed"}

    if res_path.exists():
        return _load_result(label, overrides, run_out)
    return {"label": label, "overrides": overrides, "status": "failed"}


def phase_run(args):
    factors = list(ABLATION_GROUPS.keys()) if args.factor == "all" else [args.factor]
    DEFAULTS["num_epochs"] = args.num_epochs
    out = Path(args.output_dir)
    all_results = {}

    # Load existing results so partial runs can resume
    agg_path = out / "ablation_results.json"
    if agg_path.exists():
        with open(agg_path) as f:
            all_results = json.load(f)

    for factor in factors:
        print(f"\n{'═'*62}")
        print(f"  ABLATION: {factor.upper()}  |  backbone=XLS-R  |  exp=1")
        print("═"*62)
        factor_dir = out / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for label, overrides in ABLATION_GROUPS[factor]:
            res = run_variant(label, overrides, args.segment_dir,
                              args.csv_path, factor_dir, args.dry_run)
            rows.append(res)
            print(f"  ✓ test_f1={res.get('test_f1','?')}  "
                  f"test_auc={res.get('test_auc','?')}  "
                  f"svm_f1={res.get('svm_f1','?')}")
        all_results[factor] = rows
        with open(agg_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved → {agg_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Report
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v, d=3):
    return f"{float(v):.{d}f}" if v is not None else "—"


def print_console(all_results):
    for factor, rows in all_results.items():
        print(f"\n{'═'*68}")
        print(f"  FACTOR: {factor.upper()}")
        print("═"*68)
        print(f"  {'Configuration':<35} {'Val F1':>7} {'Test F1':>8} "
              f"{'AUC':>7} {'SVM F1':>8} {'Ep':>4}")
        print("─"*68)
        for r in rows:
            lbl     = r.get("label","—")
            is_curr = CURRENT_MARKER in lbl
            print(
                f"  {'►' if is_curr else ' '} "
                f"{lbl.replace(CURRENT_MARKER,'*').strip():<33} "
                f"{fmt(r.get('val_f1')):>7} "
                f"{fmt(r.get('test_f1')):>8} "
                f"{fmt(r.get('test_auc')):>7} "
                f"{fmt(r.get('svm_f1')):>8} "
                f"{str(r.get('epochs','?')):>4}"
            )


def make_latex(factor_name, rows):
    captions = {
        "freeze": (
            "Ablation: number of frozen Transformer layers "
            "(Experiment~1, XLS-R finetune, all other settings fixed at defaults). "
            "\\rowcolor{gray!15} = current pipeline configuration."
        ),
        "loss": (
            "Ablation: loss function and Focal Loss concentration $\\gamma$ "
            "(Experiment~1, XLS-R finetune)."
        ),
        "decay": (
            "Ablation: layerwise learning-rate decay $\\lambda$ "
            "(Experiment~1, XLS-R finetune). "
            "$\\lambda=1.0$ = uniform learning rate across all layers."
        ),
    }
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{captions.get(factor_name, factor_name)}}}",
        rf"\label{{tab:ablation_{factor_name}}}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Configuration & Val F1 & Test F1 & Test AUC & Test Acc & SVM F1 & SVM AUC \\",
        r"\midrule",
    ]
    for r in rows:
        label   = r.get("label","—").replace(CURRENT_MARKER,"").strip()
        is_curr = CURRENT_MARKER in r.get("label","")
        row_str = (
            f"  {label} & {fmt(r.get('val_f1'))} & {fmt(r.get('test_f1'))} & "
            f"{fmt(r.get('test_auc'))} & {fmt(r.get('test_acc'))} & "
            f"{fmt(r.get('svm_f1'))} & {fmt(r.get('svm_auc'))} \\\\"
        )
        if is_curr:
            lines.append(r"\rowcolor{gray!15}")
        lines.append(row_str)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


DISCUSSION_TEMPLATE = """
% ─────────────────────────────────────────────────────────────────────────────
% ABLATION DISCUSSION TEMPLATE — fill FILL: placeholders from your results
% ─────────────────────────────────────────────────────────────────────────────
\\subsection*{Ablation Study: Validation of Key Design Choices}

The three most consequential hyperparameters for the finetune paradigm were
validated against alternatives on Experiment~1 (binary CRS detection,
XLS-R finetune, F1=0.696 at default settings) using a one-factor-at-a-time
protocol. All other training settings were held at the defaults reported in
Section~\\ref{sec:training_config}.

\\paragraph{Frozen layers.}
Freezing \\textbf{FILL: N} Transformer layers produced the highest
test F1 (FILL) and test AUC (FILL).
\\textbf{FILL: Describe pattern — does performance peak at 4, or does it
monotonically improve/degrade?}
The SVM probe on the all-frozen backbone achieved F1 FILL (AUC FILL),
confirming that XLS-R's pretrained representations carry useful
clinical acoustic structure even without any backbone adaptation.

\\paragraph{Loss function.}
\\textbf{FILL: FocalLoss γ=N} achieved the highest test F1 (FILL) versus
standard cross-entropy (FILL) and alternative $\\gamma$ values.
\\textbf{FILL: Does focal loss consistently outperform cross-entropy?
Does γ=2 outperform γ=1 and γ=3?}
The class imbalance in training (FESS vs Control) \\textbf{FILL: justifies
/ does not justify} the use of Focal Loss, as minority-class performance
\\textbf{FILL: improved / did not improve} over cross-entropy.

\\paragraph{Layerwise LR decay.}
$\\lambda=$\\textbf{FILL: N} produced the highest validation F1 (FILL).
Uniform learning rate ($\\lambda=1.0$) \\textbf{FILL: overfitted / was
competitive / underperformed}, while stronger decay ($\\lambda=0.7$)
\\textbf{FILL: slowed convergence / was competitive}.
The current choice of $\\lambda=0.9$ \\textbf{FILL: is / is not}
supported by the ablation evidence.

Collectively, the ablation confirms that the three design choices are
\\textbf{FILL: well-justified / partially justified / not individually
decisive} for this task at the available training data scale.
The most sensitive choice was \\textbf{FILL: frozen layers / loss / decay},
producing a test F1 range of FILL across variants.
"""


def phase_report(args):
    agg_path = Path(args.output_dir) / "ablation_results.json"
    if not agg_path.exists():
        print(f"  No results file found at {agg_path}.")
        print("  Run Phase 1 first: python scripts/run_ablation.py --run ...")
        sys.exit(1)

    with open(agg_path) as f:
        all_results = json.load(f)

    print_console(all_results)

    tables    = [make_latex(f, rows) for f, rows in all_results.items()]
    latex_out = (
        "% Requires \\usepackage{booktabs} and \\usepackage{colortbl}\n\n"
        + "\n\n".join(tables)
        + DISCUSSION_TEMPLATE
    )

    print("\n\n% ════════════════════════════════════════════\n"
          "  LaTeX Output\n"
          "% ════════════════════════════════════════════\n")
    print(latex_out)

    if args.output_tex:
        Path(args.output_tex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_tex).write_text(latex_out)
        print(f"\n  LaTeX written → {args.output_tex}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study runner and reporter for Exp1 / XLS-R finetune."
    )
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--run",    action="store_true", help="Phase 1: train variants")
    mode_grp.add_argument("--report", action="store_true", help="Phase 2: generate report")

    parser.add_argument("--output_dir",  required=True,
                        help="Root directory for ablation outputs")
    parser.add_argument("--factor",
                        choices=["all","freeze","loss","decay"], default="all")
    # Phase 1 only
    parser.add_argument("--segment_dir", default=None)
    parser.add_argument("--csv_path",    default=None)
    parser.add_argument("--num_epochs",  type=int, default=8)
    parser.add_argument("--dry_run",     action="store_true")
    # Phase 2 only
    parser.add_argument("--output_tex",  default=None,
                        help="Path to write LaTeX output file")

    args = parser.parse_args()

    if args.run:
        if not args.segment_dir or not args.csv_path:
            parser.error("--run requires --segment_dir and --csv_path")
        phase_run(args)
    else:
        phase_report(args)


if __name__ == "__main__":
    main()