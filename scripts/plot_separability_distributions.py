"""
plot_separability_distributions.py
─────────────────────────────────────────────────────────────────────────────
Produces one figure per audio task (13 total), each a 2x2 grid of F0,
Jitter, Shimmer, and HNR distributions comparing FESS vs Control on the
Exp1 training partition.

Reads the raw per-patient values saved by
run_group_separability_analysis.py (group_separability_raw_patient_values.csv)
and the corresponding summary stats (group_separability_summary.csv) for
p-value / effect-size annotations.

Only the "full utterance" condition is plotted by default (--condition to
change), since full vs segment effect sizes were near-identical in the
summary analysis -- plotting both would double the figure count for
negligible additional information. Use --condition both to plot both.

Usage
─────
    python scripts/plot_separability_distributions.py
    python scripts/plot_separability_distributions.py --condition both
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "Plots and visuals" / "diagnostics"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "separability_figures"

# Full cohort names -- C3-17: no abbreviated labels in Results-facing figures
GROUP_NAMES = {"FESS": "Functional Endoscopic\nSinus Surgery", "Contr": "Control"}
GROUP_COLOURS = {"FESS": "#C62828", "Contr": "#2E7D32"}

TASK_LABELS = {
    "a": "Vowel /a/", "e": "Vowel /e/", "i": "Vowel /i/",
    "o": "Vowel /o/", "u": "Vowel /u/",
    "a1": "Sustained /a/ (rep. 1)", "a2": "Sustained /a/ (rep. 2)",
    "a3": "Sustained /a/ (rep. 3)",
    "agua": "TDU word: \u201cagua\u201d", "brasero": "TDU word: \u201cbrasero\u201d",
    "dia": "TDU word: \u201cd\u00eda\u201d", "mesa": "TDU word: \u201cmesa\u201d",
    "speech": "Free-speech monologue",
}

FEATURES = [
    ("f0", "F0 (Hz)"),
    ("jitter", "Jitter (%)"),
    ("shimmer", "Shimmer (%)"),
    ("hnr", "HNR (dB)"),
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F7F9FC",
    "axes.edgecolor": "#BDC3C7",
    "font.family": "sans-serif",
    "font.size": 14,          # C3-16: match main document font size
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
})


def plot_task(task: str, raw_df: pd.DataFrame, summary_df: pd.DataFrame,
              condition: str, output_dir: Path):
    task_df = raw_df[raw_df["task"] == task]
    if task_df.empty:
        print(f"  [skip] no data for task '{task}'")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f"{TASK_LABELS.get(task, task)} \u2014 Acoustic Feature Distributions\n"
        f"FESS vs Control (training partition, {condition} recordings)",
        fontsize=16, y=1.00,
    )

    for ax, (feat_key, feat_label) in zip(axes.flat, FEATURES):
        col = f"{condition}_{feat_key}"
        if col not in task_df.columns:
            ax.axis("off")
            continue

        fess_vals = task_df.loc[task_df["_group"] == "FESS", col].dropna()
        contr_vals = task_df.loc[task_df["_group"] == "Contr", col].dropna()

        data = [fess_vals, contr_vals]
        labels = [GROUP_NAMES["FESS"], GROUP_NAMES["Contr"]]
        colours = [GROUP_COLOURS["FESS"], GROUP_COLOURS["Contr"]]

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.35)
        for median in bp["medians"]:
            median.set_color("#1A237E")
            median.set_linewidth(2)

        # Jittered strip plot on top for individual patient visibility
        rng = np.random.default_rng(42)
        for i, (vals, colour) in enumerate(zip(data, colours), start=1):
            x = rng.normal(i, 0.05, size=len(vals))
            ax.scatter(x, vals, color=colour, alpha=0.7, s=22, zorder=3,
                       edgecolors="white", linewidths=0.5)

        # Annotate with p-value / effect size from the summary table
        row = summary_df[
            (summary_df["task"] == task)
            & (summary_df["feature"].str.lower() == feat_key)
            & (summary_df["condition"] == condition)
        ]
        if not row.empty:
            p = row["p_value"].iloc[0]
            d = row["cohens_d"].iloc[0]
            sig_marker = "*" if pd.notna(p) and p < 0.05 else "n.s."
            ax.set_title(f"{feat_label}   (p={p:.3f} {sig_marker}, d={d:.2f})",
                         fontsize=13)
        else:
            ax.set_title(feat_label, fontsize=13)

        ax.set_ylabel(feat_label)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"separability_{task}_{condition}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default=None)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--condition", choices=["full", "segment", "both"],
                         default="full")
    args = parser.parse_args()

    raw_csv = Path(args.raw_csv) if args.raw_csv else (
        DEFAULT_INPUT_DIR / "group_separability_raw_patient_values.csv"
    )
    summary_csv = Path(args.summary_csv) if args.summary_csv else (
        DEFAULT_INPUT_DIR / "group_separability_summary.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv.exists():
        print(f"Raw values file not found: {raw_csv}")
        print("Re-run run_group_separability_analysis.py with the raw-CSV "
              "export added first.")
        return

    raw_df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv)

    conditions = ["full", "segment"] if args.condition == "both" else [args.condition]

    tasks = [t for t in TASK_LABELS if t in raw_df["task"].unique()]
    print(f"Plotting {len(tasks)} tasks x {len(conditions)} condition(s)...")
    for condition in conditions:
        for task in tasks:
            plot_task(task, raw_df, summary_df, condition, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
