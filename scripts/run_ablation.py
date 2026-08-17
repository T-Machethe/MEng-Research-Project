"""
run_ablation.py
─────────────────────────────────────────────────────────────────────────────
Two-phase ablation study on Experiment 1 (binary CRS detection), for
whichever backbone/mode you point it at via --backbone/--mode.

Three factors varied one at a time (all others at current defaults):
  A. freeze — frozen Transformer layers: 0, 2, 4 (current), 6, 8, all-frozen
  B. loss   — CrossEntropy vs FocalLoss γ=1/2(current)/3
  C. decay  — layerwise LR decay λ=1.0/0.9(current)/0.8/0.7

PHASE 1 (--run):    train all variants, save results_summary.json per variant
PHASE 2 (--report): read ablation_results.json, print tables, write LaTeX

Backbone/mode scoping
────────────────────────
Originally hardcoded to XLS-R finetune only. Now takes --backbone/--mode,
and results are namespaced under <output_dir>/<backbone>_<mode>/<factor>/
so ablating multiple architectures (e.g. your nested-CV shortlist) into
the SAME --output_dir doesn't collide — each backbone/mode gets its own
subtree and its own ablation_results.json. --report also takes
--backbone/--mode to pick which subtree to read.

The "freeze=all" variant is backbone-aware: it freezes every Transformer
layer the backbone actually has (12 for wav2vec2-base/WavLM-base, 24 for
XLS-R-300m — see NUM_LAYERS below), not a value hardcoded for XLS-R. This
was a real bug for any non-XLS-R backbone: passing --freeze_layers 24 to
a 12-layer backbone doesn't error, it just silently clamps/over-freezes,
which would have made every non-XLS-R "freeze=all" ablation variant
meaningless without any visible failure.

Note: the freeze levels in between (0, 2, 4, 6, 8) are kept as ABSOLUTE
layer counts across every backbone for direct comparability, rather than
rescaled proportionally to each backbone's depth (e.g. 8/24 vs 8/12 are
not the same fraction of the network). State this explicitly if
comparing freeze ablations across backbones of different depths in the
thesis — "freeze=8" means something structurally different for a
12-layer vs 24-layer backbone.

Usage
─────
    # Phase 1 — run one factor per Colab session (~40 min per factor on T4)
    python scripts/run_ablation.py --run \\
        --backbone wav2vec2 --mode finetune \\
        --factor freeze \\
        --segment_dir /content/clean_audio_3s \\
        --csv_path    /path/to/clinical_all_sessions.csv \\
        --output_dir  /content/drive/MyDrive/MSc_Sinusitis_results/ablation \\
        --num_epochs  8

    # Phase 2 — generate LaTeX tables after all factors complete
    python scripts/run_ablation.py --report \\
        --backbone wav2vec2 --mode finetune \\
        --output_dir  /content/drive/MyDrive/MSc_Sinusitis_results/ablation \\
        --output_tex  /content/drive/MyDrive/thesis_outputs/ablation_wav2vec2_finetune.tex

    # Dry run — print commands without using GPU
    python scripts/run_ablation.py --run --backbone xlsr --mode finetune --factor all --dry_run ...
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Actual Transformer depth per backbone family — see src/training/trainer.py's
# _build_model() scratch-mode Config construction, and
# scripts/run_cross_cohort_specificity.py's build_inference_model(), both of
# which use these same depths (12 for wav2vec2/WavLM-base, 24 for XLS-R-300m).
NUM_LAYERS = {"wav2vec2": 12, "wavlm": 12, "xlsr": 24}


def build_defaults(backbone: str, mode: str) -> dict:
    """Was a fixed module-level dict (XLS-R finetune only) — now built per
    backbone/mode so the "current" config those defaults represent still
    makes sense (e.g. batch_size=8 was an XLS-R-specific memory
    constraint; kept for now since it's a safe default across backbones,
    but override with --batch_size if a specific backbone can handle more)."""
    return dict(
        backbone           = backbone,
        mode               = mode,
        exp                = "1",
        num_epochs         = 8,
        batch_size         = 8,
        learning_rate      = 1e-5,
        freeze_layers      = 4,           # current
        focal_gamma        = 2.0,         # current
        layerwise_lr_decay = 0.9,         # current
        label_smoothing    = 0.1,
        head_warmup_epochs = 1,
        early_stop_patience= 3,
        early_stop_metric  = "val_f1",
        imbalance          = "weights",
        warmup_steps       = 100,
        use_svm            = True,
    )


def build_ablation_groups(backbone: str) -> dict:
    """Was a fixed module-level dict with "freeze=all" hardcoded to 24
    (XLS-R's layer count) — now computes the all-frozen variant from the
    ACTUAL backbone's depth via NUM_LAYERS, so this is correct for
    wav2vec2/WavLM (12 layers) too, not just XLS-R."""
    num_layers = NUM_LAYERS[backbone]
    return {
        "freeze": [
            ("freeze=0  (all layers train)", {"freeze_layers": 0}),
            ("freeze=2",                     {"freeze_layers": 2}),
            ("freeze=4  [CURRENT]",          {"freeze_layers": 4}),
            ("freeze=6",                     {"freeze_layers": 6}),
            ("freeze=8",                     {"freeze_layers": 8}),
            (f"freeze=all ({num_layers}, SVM probe only)",
             {"freeze_layers": num_layers, "use_svm": True}),
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

# Short folder names — one word per variant, no special characters
# Structure: ablation/{factor}/{folder_name}/results_summary.json
VARIANT_FOLDERS = {
    # freeze
    "freeze=0  (all layers train)":   "freeze_0",
    "freeze=2":                        "freeze_2",
    "freeze=4  [CURRENT]":             "freeze_4",
    "freeze=6":                        "freeze_6",
    "freeze=8":                        "freeze_8",
    "freeze=all  (SVM probe only)":    "freeze_all",
    # loss
    "CrossEntropy":                    "ce",
    "FocalLoss γ=1":                   "focal_g1",
    "FocalLoss γ=2  [CURRENT]":        "focal_g2",
    "FocalLoss γ=3":                   "focal_g3",
    # decay
    "λ=1.0  (uniform LR)":            "L1p0",
    "λ=0.9  [CURRENT]":               "L0p9",
    "λ=0.8":                           "L0p8",
    "λ=0.7":                           "L0p7",
}



# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Training
# ══════════════════════════════════════════════════════════════════════════════

def build_cmd(segment_dir, csv_path, run_out, overrides, defaults):
    cfg = {**defaults, **overrides}
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


def _load_result(label, overrides, res_path, status="complete"):
    """Load metrics from a results_summary.json path."""
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


def run_variant(label, overrides, segment_dir, csv_path, factor_dir, dry_run, defaults):
    # factor_dir = ablation/<backbone>_<mode>/freeze  →  run_out = .../freeze/freeze_4/
    # "freeze=all" now embeds the backbone-specific layer count (12 vs 24)
    # in the label, so it won't exact-match VARIANT_FOLDERS's static key —
    # normalize that one case explicitly rather than needing a static
    # entry per possible layer count.
    if label.startswith("freeze=all"):
        folder = "freeze_all"
    else:
        folder = VARIANT_FOLDERS.get(label, _safe_label(label))
    run_out = factor_dir / folder

    # run_experiment.py appends the mode ("finetune") as a subfolder via run_label,
    # so results_summary.json lands at run_out/finetune/results_summary.json
    # run_experiment.py uses exp{exp_key}_mixed when no run_label is passed
    res_path = run_out / f"exp{defaults['exp']}_mixed" / "results_summary.json"

    # ── Fast skip: check before spawning any subprocess ───────────────────
    if res_path.exists():
        print(f"\n  ↩  SKIP (already complete): {label}")
        print(f"     {res_path}")
        return _load_result(label, overrides, res_path)

    print(f"\n  {'─'*58}")
    print(f"  {label}")
    print(f"  Overrides : {overrides}")
    print(f"  Output    : {run_out}")

    if dry_run:
        print("  [DRY RUN — not executed]")
        return {"label": label, "overrides": overrides, "status": "dry_run"}

    cmd = build_cmd(segment_dir, csv_path, run_out, overrides, defaults)

    # Capture stderr so failures show the actual traceback
    result = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"\n  ✗  Variant failed (exit {result.returncode}): {label}")
        if result.stderr and result.stderr.strip():
            lines = result.stderr.strip().splitlines()
            print("\n  --- stderr (last 40 lines) ---")
            for ln in lines[-40:]:
                print(f"  {ln}")
            print("  --- end stderr ---\n")
        return {"label": label, "overrides": overrides, "status": "failed"}

    if res_path.exists():
        return _load_result(label, overrides, res_path)
    return {"label": label, "overrides": overrides, "status": "failed"}


def phase_run(args):
    defaults = build_defaults(args.backbone, args.mode)
    ablation_groups = build_ablation_groups(args.backbone)
    defaults["num_epochs"] = args.num_epochs

    factors = list(ablation_groups.keys()) if args.factor == "all" else [args.factor]

    # Namespaced by backbone/mode so multiple architectures can be ablated
    # into the SAME --output_dir without overwriting each other — this was
    # a real collision risk before: two backbones both writing to
    # <output_dir>/freeze/freeze_4/ would silently clobber one another.
    backbone_mode = f"{args.backbone}_{args.mode}"
    out = Path(args.output_dir) / backbone_mode
    all_results = {}

    # Load existing results so partial runs can resume
    agg_path = out / "ablation_results.json"
    if agg_path.exists():
        with open(agg_path) as f:
            all_results = json.load(f)

    for factor in factors:
        print(f"\n{'═'*62}")
        print(f"  ABLATION: {factor.upper()}  |  backbone={args.backbone}  |  "
              f"mode={args.mode}  |  exp=1")
        print("═"*62)
        factor_dir = out / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for label, overrides in ablation_groups[factor]:
            res = run_variant(label, overrides, args.segment_dir,
                              args.csv_path, factor_dir, args.dry_run, defaults)
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


def make_latex(factor_name, rows, backbone, mode):
    backbone_label = {"wav2vec2": "wav2vec2", "wavlm": "WavLM", "xlsr": "XLS-R"}.get(backbone, backbone)
    arch_str = f"{backbone_label} {mode}"
    captions = {
        "freeze": (
            f"Ablation: number of frozen Transformer layers "
            f"(Experiment~1, {arch_str}, all other settings fixed at defaults). "
            "\\rowcolor{gray!15} = current pipeline configuration."
        ),
        "loss": (
            f"Ablation: loss function and Focal Loss concentration $\\gamma$ "
            f"(Experiment~1, {arch_str})."
        ),
        "decay": (
            f"Ablation: layerwise learning-rate decay $\\lambda$ "
            f"(Experiment~1, {arch_str}). "
            "$\\lambda=1.0$ = uniform learning rate across all layers."
        ),
    }
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{captions.get(factor_name, factor_name)}}}",
        rf"\label{{tab:ablation_{backbone}_{mode}_{factor_name}}}",
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
XLS-R finetune, FILL: F1 at default settings) using a one-factor-at-a-time
protocol. All other training settings were held at the defaults reported in
Section~\\ref{sec:training_config}.

\\paragraph{Frozen layers.}
Freezing \\textbf{FILL: N} Transformer layers produced the highest
test F1 (FILL) and test AUC (FILL).
\\textbf{FILL: Describe pattern — does performance peak at 4, or does it
monotonically improve/degrade?}
The SVM probe on the all-frozen backbone achieved F1 FILL (AUC FILL),
confirming that the backbone's pretrained representations carry useful
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
    backbone_mode = f"{args.backbone}_{args.mode}"
    agg_path = Path(args.output_dir) / backbone_mode / "ablation_results.json"
    if not agg_path.exists():
        print(f"  No results file found at {agg_path}.")
        print(f"  Run Phase 1 first: python scripts/run_ablation.py --run "
              f"--backbone {args.backbone} --mode {args.mode} ...")
        sys.exit(1)

    with open(agg_path) as f:
        all_results = json.load(f)

    print_console(all_results)

    tables    = [make_latex(f, rows, args.backbone, args.mode) for f, rows in all_results.items()]
    latex_out = (
        "% Requires \\usepackage{booktabs} and \\usepackage{colortbl}\n\n"
        + "\n\n".join(tables)
        + DISCUSSION_TEMPLATE.replace("XLS-R finetune", f"{args.backbone} {args.mode}")
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
        description="Ablation study runner and reporter for Exp1, any backbone/mode."
    )
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--run",    action="store_true", help="Phase 1: train variants")
    mode_grp.add_argument("--report", action="store_true", help="Phase 2: generate report")

    parser.add_argument("--backbone", choices=list(NUM_LAYERS.keys()), default="xlsr",
                        help="Which backbone to ablate. Determines the 'freeze=all' "
                             "layer count (12 for wav2vec2/wavlm, 24 for xlsr) and the "
                             "output namespace, so multiple backbones can share one "
                             "--output_dir without colliding.")
    parser.add_argument("--mode", choices=["scratch", "finetune"], default="finetune")
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