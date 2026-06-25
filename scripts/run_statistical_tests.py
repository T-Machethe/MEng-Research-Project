"""
run_statistical_tests.py
─────────────────────────────────────────────────────────────────────────────
Segment-level statistical analyses for Experiment 1. Operates entirely on
saved prediction arrays — no GPU or model reloading required.

Analyses
────────
  1. Permutation test (10,000 permutations, one-sided) for all 9 model
     configurations. Reports observed F1, null distribution mean ± SD,
     and p-value.

  2. Bootstrap 95% confidence intervals for F1 and AUC (10,000 resamples)
     for all 9 configurations.

  3. Continuity-corrected McNemar's test for 4 pairwise comparisons:
       (a) wav2vec2-scratch vs XLS-R-finetune
       (b) wav2vec2-scratch vs WavLM-scratch
       (c) XLS-R-finetune  vs WavLM-finetune
       (d) XLS-R-SVM       vs wav2vec2-scratch

Input
─────
  A single JSON file containing integer prediction arrays and float
  probability arrays keyed as:
    y_true
    y_pred_w2v2_scratch,  y_pred_wavlm_scratch,  y_pred_xlsr_scratch
    y_pred_w2v2_ft,       y_pred_wavlm_ft,       y_pred_xlsr_ft
    y_pred_w2v2_svm,      y_pred_wavlm_svm,      y_pred_xlsr_svm
    y_prob_w2v2_scratch,  y_prob_wavlm_scratch,  y_prob_xlsr_scratch
    y_prob_w2v2_ft,       y_prob_wavlm_ft,       y_prob_xlsr_ft
    y_prob_w2v2_svm,      y_prob_wavlm_svm,      y_prob_xlsr_svm

  Missing keys (e.g. xlsr_scratch before training completes) are handled
  gracefully — those rows appear as "pending" in the output tables.

Output — four files written to <output_dir>:
    stat_permutation_test.tex   LaTeX table + summary sentence
    stat_bootstrap_ci.tex       LaTeX table
    stat_mcnemar.tex            LaTeX table
    stat_mcnemar_prose.tex      Prose paragraph ready for thesis insertion

Usage
─────
    python scripts/run_statistical_tests.py \
        --predictions /path/to/exp1_predictions.json \
        --output_dir  /content/drive/MyDrive/MSc_Sinusitis_results/thesis_outputs

    # Default output_dir matches the thesis_outputs convention used throughout:
    python scripts/run_statistical_tests.py --predictions exp1_predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

SEED        = 42
N_PERM      = 10_000
N_BOOT      = 10_000
DEFAULT_OUT = "/content/drive/MyDrive/MSc_Sinusitis_results/thesis_outputs"


# ── Row definitions — ordering matches the thesis results tables ─────────────

MODELS = [
    # (display_name,        pred_key,               prob_key,               is_svm)
    ("wav2vec2-scratch",    "y_pred_w2v2_scratch",  "y_prob_w2v2_scratch",  False),
    ("WavLM-scratch",       "y_pred_wavlm_scratch", "y_prob_wavlm_scratch", False),
    ("XLS-R-scratch",       "y_pred_xlsr_scratch",  "y_prob_xlsr_scratch",  False),
    ("wav2vec2-FT (MLP)",   "y_pred_w2v2_ft",       "y_prob_w2v2_ft",       False),
    ("WavLM-FT (MLP)",      "y_pred_wavlm_ft",      "y_prob_wavlm_ft",      False),
    ("XLS-R-FT (MLP)",      "y_pred_xlsr_ft",       "y_prob_xlsr_ft",       False),
    ("wav2vec2-FT (SVM)",   "y_pred_w2v2_svm",      "y_prob_w2v2_svm",      True),
    ("WavLM-FT (SVM)",      "y_pred_wavlm_svm",     "y_prob_wavlm_svm",     True),
    ("XLS-R-FT (SVM)",      "y_pred_xlsr_svm",      "y_prob_xlsr_svm",      True),
]

MCNEMAR_PAIRS = [
    # (label_A,           pred_key_A,            label_B,           pred_key_B)
    ("wav2vec2-scratch",  "y_pred_w2v2_scratch", "XLS-R-FT (MLP)",  "y_pred_xlsr_ft"),
    ("wav2vec2-scratch",  "y_pred_w2v2_scratch", "WavLM-scratch",   "y_pred_wavlm_scratch"),
    ("XLS-R-FT (MLP)",   "y_pred_xlsr_ft",      "WavLM-FT (MLP)",  "y_pred_wavlm_ft"),
    ("XLS-R-FT (SVM)",   "y_pred_xlsr_svm",     "wav2vec2-scratch", "y_pred_w2v2_scratch"),
]

LATEX_MODEL_NAMES = {
    "wav2vec2-scratch":  r"\textit{wav2vec2}-scratch",
    "WavLM-scratch":     r"WavLM-scratch",
    "XLS-R-scratch":     r"XLS-R-scratch",
    "wav2vec2-FT (MLP)": r"\textit{wav2vec2}-FT (MLP)",
    "WavLM-FT (MLP)":    r"WavLM-FT (MLP)",
    "XLS-R-FT (MLP)":    r"XLS-R-FT (MLP)",
    "wav2vec2-FT (SVM)": r"\textit{wav2vec2}-FT (SVM)",
    "WavLM-FT (SVM)":    r"WavLM-FT (SVM)",
    "XLS-R-FT (SVM)":    r"XLS-R-FT (SVM)",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data(json_path: Path) -> dict:
    with open(json_path) as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}


def fmt_pval(p: float) -> str:
    if p < 0.001:
        return r"$p < 0.001$"
    return f"{p:.3f}"


def fmt_pval_prose(p: float) -> str:
    if p < 0.001:
        return "$p < 0.001$"
    return f"$p = {p:.3f}$"


def lname(key: str) -> str:
    return LATEX_MODEL_NAMES.get(key, key)


# ── Analysis 1 — Permutation test ────────────────────────────────────────────

def run_permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> tuple:
    """
    One-sided permutation test: permute y_pred and compute null F1.
    Returns (observed_f1, null_mean, null_std, p_value).
    """
    rng      = np.random.RandomState(seed)
    observed = f1_score(y_true, y_pred, average="macro")
    null_f1  = np.empty(n_perm)
    for i in range(n_perm):
        null_f1[i] = f1_score(y_true, rng.permutation(y_pred), average="macro")
    p_val = float((null_f1 >= observed).mean())
    return float(observed), float(null_f1.mean()), float(null_f1.std()), p_val


def analysis1(data: dict, out: Path, exp_tag: str = ""):
    print("\n── Analysis 1: Permutation tests ───────────────────────────────")

    rows  = []
    sig   = []
    insig = []

    for name, pred_key, _, _ in MODELS:
        if pred_key not in data:
            print(f"  {name}: PENDING (key '{pred_key}' not found)")
            rows.append(None)
            continue
        obs, nmean, nstd, pval = run_permutation_test(
            data["y_true"], data[pred_key])
        rows.append((name, obs, nmean, nstd, pval))
        flag = "✓" if pval < 0.05 else "✗"
        print(f"  {name:<24} obs={obs:.3f}  null={nmean:.3f}±{nstd:.3f}  "
              f"p={pval:.4f}  {flag}")
        (sig if pval < 0.05 else insig).append(name)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Segment-level permutation test results for Experiment~1 "
        r"(10{,}000 random permutations of predicted labels, one-sided). "
        r"The null distribution is the macro-averaged F1 expected under "
        r"random label assignment. All five test patients contribute "
        r"1{,}875 segments; the test is conducted at the segment level.}",
        r"\label{tab:permutation_test_exp1}",
        r"\begin{tabular}{lrll}",
        r"\toprule",
        r"Model & Observed F1 & Null mean $\pm$ SD & $p$-value \\",
        r"\midrule",
    ]
    for row in rows:
        if row is None:
            lines.append(r"\textit{(pending)} & -- & -- & -- \\")
            continue
        name, obs, nm, nsd, pval = row
        lines.append(
            f"  {lname(name)} & {obs:.3f} & "
            f"${nm:.3f} \\pm {nsd:.3f}$ & "
            f"{fmt_pval(pval)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    if insig:
        exception_clause = (
            "with the exception of "
            + (", ".join(insig[:-1]) + " and " + insig[-1]
               if len(insig) > 1 else insig[0])
            + " which did not reach significance"
        )
    else:
        exception_clause = "with no exceptions"

    sentence = (
        "Segment-level permutation tests (10,000 permutations) "
        "confirm that all configurations achieving F1 above binary chance "
        "(0.500) produce results that are not attributable to chance at the "
        "segment level (all $p < 0.05$), "
        f"{exception_clause}; "
        "all results should be interpreted at the segment level rather than "
        "the patient level given the five-patient test cohort."
    )
    lines += [r"% Summary sentence for thesis body:", f"% {sentence}", ""]

    path = out / f"stat_permutation_test{exp_tag}.tex"
    path.write_text("\n".join(lines))
    print(f"\n  Saved → {path}")
    print(f"\n  Summary sentence:\n  {sentence}")


# ── Analysis 2 — Bootstrap confidence intervals ───────────────────────────────

def run_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple:
    """
    Bootstrap 95% CI for macro-F1 and AUC.
    Returns (f1_lo, f1_hi, auc_lo, auc_hi).
    """
    rng  = np.random.RandomState(seed)
    n    = len(y_true)
    f1s  = np.empty(n_boot)
    aucs = np.full(n_boot, np.nan)

    for i in range(n_boot):
        idx    = rng.choice(n, n, replace=True)
        yt_b   = y_true[idx]
        yp_b   = y_pred[idx]
        f1s[i] = f1_score(yt_b, yp_b, average="macro")
        if y_prob is not None:
            try:
                if y_prob.ndim == 2:   # multi-class: full prob matrix
                    aucs[i] = roc_auc_score(
                        yt_b, y_prob[idx],
                        multi_class="ovr", average="macro")
                else:                  # binary: class-1 probability column
                    aucs[i] = roc_auc_score(yt_b, y_prob[idx])
            except ValueError:
                pass

    f1_lo,  f1_hi  = np.percentile(f1s, [2.5, 97.5])
    auc_lo, auc_hi = np.nanpercentile(aucs, [2.5, 97.5])
    return float(f1_lo), float(f1_hi), float(auc_lo), float(auc_hi)


def analysis2(data: dict, out: Path, exp_tag: str = ""):
    print("\n── Analysis 2: Bootstrap confidence intervals ───────────────────")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Bootstrap 95\% confidence intervals for Experiment~1 "
        r"macro-F1 and AUC (10{,}000 bootstrap resamples with replacement, "
        r"seed\,=\,42). Intervals are computed at the segment level across "
        r"1{,}875 test segments from five test patients.}",
        r"\label{tab:bootstrap_ci_exp1}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Model & F1 [95\% CI] & AUC [95\% CI] \\",
        r"\midrule",
    ]

    for name, pred_key, prob_key, _ in MODELS:
        if pred_key not in data:
            lines.append(
                f"  {lname(name)} & \\textit{{pending}} & \\textit{{pending}} \\\\"
            )
            continue
        y_prob = data.get(prob_key)
        f1l, f1h, al, ah = run_bootstrap_ci(
            data["y_true"], data[pred_key], y_prob)
        f1_ci  = f"[{f1l:.3f},\\;{f1h:.3f}]"
        auc_ci = f"[{al:.3f},\\;{ah:.3f}]" if not np.isnan(al) else "N/A"
        obs_f1 = f1_score(data["y_true"], data[pred_key], average="macro")
        print(f"  {name:<24} F1={obs_f1:.3f} [{f1l:.3f},{f1h:.3f}]  "
              f"AUC [{al:.3f},{ah:.3f}]")
        lines.append(
            f"  {lname(name)} & ${f1_ci}$ & ${auc_ci}$ \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = out / f"stat_bootstrap_ci{exp_tag}.tex"
    path.write_text("\n".join(lines))
    print(f"\n  Saved → {path}")


# ── Analysis 3 — McNemar's test ───────────────────────────────────────────────

def run_mcnemar(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple:
    """
    Continuity-corrected McNemar's test.
    b = A correct and B wrong.  c = A wrong and B correct.
    Returns (b, c, chi2, p_value).
    """
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = int(np.sum( correct_a & ~correct_b))
    c = int(np.sum(~correct_a &  correct_b))
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    pval = float(1 - stats.chi2.cdf(chi2, df=1))
    return b, c, float(chi2), pval


def analysis3(data: dict, out: Path, exp_tag: str = ""):
    print("\n── Analysis 3: McNemar's tests ─────────────────────────────────")

    results = []
    for la, ka, lb, kb in MCNEMAR_PAIRS:
        if ka not in data or kb not in data:
            print(f"  {la} vs {lb}: PENDING (missing predictions)")
            results.append(None)
            continue
        b, c, chi2, pval = run_mcnemar(data["y_true"], data[ka], data[kb])
        flag = "significant" if pval < 0.05 else "NOT significant"
        print(f"  {la} vs {lb}: b={b}, c={c}, "
              f"χ²={chi2:.3f}, p={pval:.4f}  [{flag}]")
        results.append((la, lb, b, c, chi2, pval))

    # LaTeX table
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Continuity-corrected McNemar's test on segment-level "
        r"binary predictions for Experiment~1 (1{,}875 test segments). "
        r"$b$ = segments where Model~A is correct and Model~B is wrong; "
        r"$c$ = segments where Model~B is correct and Model~A is wrong. "
        r"$\chi^2$ is computed on discordant pairs only.}",
        r"\label{tab:mcnemar_exp1}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparison & $b$ & $c$ & $\chi^2$ & $p$-value \\",
        r"\midrule",
    ]
    for i, (la, _, lb, _) in enumerate(MCNEMAR_PAIRS):
        comp = f"{lname(la)} vs {lname(lb)}"
        if results[i] is None:
            lines.append(f"  {comp} & -- & -- & -- & \\textit{{pending}} \\\\")
            continue
        _, _, b, c, chi2, pval = results[i]
        lines.append(f"  {comp} & {b} & {c} & {chi2:.3f} & {fmt_pval(pval)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Prose paragraph
    complete    = [r for r in results if r is not None]
    sig_pairs   = [r for r in complete if r[5] < 0.05]
    insig_pairs = [r for r in complete if r[5] >= 0.05]

    sentences = []
    for la, lb, b, c, chi2, pval in sig_pairs:
        direction = "outperforms" if b > c else "is outperformed by"
        sentences.append(
            f"The comparison between {lname(la)} and {lname(lb)} is "
            f"statistically significant ({fmt_pval_prose(pval)}, "
            f"$\\chi^2 = {chi2:.3f}$, $b = {b}$, $c = {c}$): "
            f"{lname(la)} {direction} {lname(lb)} on a significantly "
            f"greater number of discordant segments."
        )
    for la, lb, b, c, chi2, pval in insig_pairs:
        sentences.append(
            f"The difference between {lname(la)} and {lname(lb)} does not "
            f"reach statistical significance ({fmt_pval_prose(pval)}, "
            f"$\\chi^2 = {chi2:.3f}$, $b = {b}$, $c = {c}$), indicating "
            f"that the segment-level performance of these two configurations "
            f"cannot be reliably distinguished on the five-patient test cohort."
        )
    if insig_pairs:
        sentences.append(
            "These null results are reported honestly and reflect the limited "
            "statistical power of a five-patient test set; they do not imply "
            "equivalence of the compared systems."
        )

    prose = (
        r"\paragraph{Pairwise significance (Experiment~1)}" + "\n\n"
        + " ".join(sentences)
    )

    lines += ["", r"% ── Prose paragraph for thesis Results section ──",
              prose, ""]
    path = out / f"stat_mcnemar{exp_tag}.tex"
    path.write_text("\n".join(lines))
    print(f"\n  Saved → {path}")

    prose_path = out / f"stat_mcnemar_prose{exp_tag}.tex"
    prose_path.write_text(prose + "\n")
    print(f"  Saved → {prose_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Statistical tests for Experiment 1 — permutation, bootstrap CI, McNemar"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--exp",   default="1",
                        help="Experiment number — appended to output filenames")
    parser.add_argument("--n_perm", type=int, default=N_PERM)
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    exp_tag = f"_exp{args.exp}"

    pred_path = Path(args.predictions)
    out_dir   = Path(args.output_dir)

    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from: {pred_path}")
    data = load_data(pred_path)
    keys = [k for k in data if k.startswith("y_pred")]
    print(f"Found {len(keys)} prediction arrays: {keys}")
    print(f"y_true length: {len(data.get('y_true', []))}")

    analysis1(data, out_dir, exp_tag=exp_tag)
    analysis2(data, out_dir, exp_tag=exp_tag)
    analysis3(data, out_dir, exp_tag=exp_tag)
    print(f"\n  All tables → {out_dir}")

    print(f"\n{'═'*62}")
    print(f"  All tables written to: {out_dir}")
    print(f"    stat_permutation_test.tex")
    print(f"    stat_bootstrap_ci.tex")
    print(f"    stat_mcnemar.tex")
    print(f"    stat_mcnemar_prose.tex")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()