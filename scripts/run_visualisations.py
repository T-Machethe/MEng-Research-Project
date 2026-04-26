"""
visualise.py
─────────────────────────────────────────────────────────────────────────────
Preprocessing visualisations for the sinusitis wav2vec 2.0 pipeline.

Produces per-file plots showing:
  1. Raw waveform vs cleaned/VAD-filtered waveform (before / after)
  2. Amplitude normalisation before / after
  3. Average duration, file count and duration distribution per group

Saves all figures to:
    Project Folder/Plots and visuals/preprocessing/

Also displays each figure interactively.

Usage
─────
    python scripts/visualise.py                        # uses CSV to pick samples
    python scripts/visualise.py --n_samples 3          # plot 3 files per group
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")

# ── Locate project root (scripts/ is one level below Project Folder) ─────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Import pipeline components ────────────────────────────────────────────────
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.cleaning import (
    remove_leading_silence,
    highpass_filter,
    voice_activity_detection,
)
from src.audio.segmentation import global_amplitude_normalize
from src.config import TARGET_SR, LEADING_TRIM_S, VAD_THRESHOLD
from src.utils.paths import resolve_path

# ── Plot style ────────────────────────────────────────────────────────────────
PALETTE = {
    "Sept":   "#2196F3",   # blue
    "Fess":   "#E91E63",   # pink/red
    "Contr":  "#4CAF50",   # green
    "Tonsill":"#FF9800",   # orange
}
BG        = "#0F1117"
PANEL_BG  = "#1A1D27"
TEXT      = "#E8EAF0"
GRID      = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID,
})

OUTPUT_DIR = PROJECT_ROOT / "Plots and visuals" / "preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load and step-by-step clean a single file, returning each stage
# ─────────────────────────────────────────────────────────────────────────────

def load_stages(filepath: str):
    """
    Load a WAV file and return the waveform at each preprocessing stage.

    Returns
    -------
    stages : dict with keys
        'raw'        – original waveform (numpy, 1-D)
        'trimmed'    – after leading silence removal
        'filtered'   – after high-pass filter
        'vad'        – after VAD (voiced frames only)
        'normalised' – after global amplitude normalisation
        'sr'         – sample rate (int)
    """
    waveform, sr = torchaudio.load(filepath)
    waveform = waveform.mean(dim=0, keepdim=True)   # mono [1, T]

    raw = waveform.squeeze().numpy().copy()

    # Stage 1 – trim leading silence
    trimmed = remove_leading_silence(waveform, sr)

    # Stage 2 – resample
    if sr != TARGET_SR:
        trimmed = T.Resample(sr, TARGET_SR)(trimmed)
        sr = TARGET_SR

    # Stage 3 – high-pass filter
    filtered = highpass_filter(trimmed, sr)

    # Stage 4 – VAD
    vad = voice_activity_detection(filtered, sr)

    # Stage 5 – amplitude normalisation
    peak = vad.abs().max()
    normalised = vad / (peak + 1e-8)

    return {
        "raw":        raw,
        "trimmed":    trimmed.squeeze().numpy(),
        "filtered":   filtered.squeeze().numpy(),
        "vad":        vad.squeeze().numpy(),
        "normalised": normalised.squeeze().numpy(),
        "sr":         sr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Raw vs Cleaned waveform
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_vs_cleaned(stages: dict, title: str, save_path: Path):
    """
    Two-panel figure: raw waveform (top) vs VAD-filtered waveform (bottom).
    Shows visually how much non-voiced content is removed.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), facecolor=BG)
    fig.suptitle(f"Raw vs Cleaned Waveform\n{title}",
                 fontsize=11, color=TEXT, y=1.01)

    sr   = stages["sr"]
    raw  = stages["raw"]
    vad  = stages["vad"]

    t_raw = np.linspace(0, len(raw) / sr, len(raw))
    t_vad = np.linspace(0, len(vad) / sr, len(vad))

    axes[0].plot(t_raw, raw, color="#5C9BD6", linewidth=0.5, alpha=0.9)
    axes[0].set_title("Raw Waveform", fontsize=9, color=TEXT, pad=4)
    axes[0].set_ylabel("Amplitude", fontsize=8)
    axes[0].set_xlabel(f"Time (s)  —  duration: {len(raw)/sr:.2f}s", fontsize=8)
    axes[0].grid(True, linewidth=0.4)
    axes[0].axhline(0, color=GRID, linewidth=0.8)

    axes[1].plot(t_vad, vad, color="#E91E63", linewidth=0.5, alpha=0.9)
    axes[1].set_title(
        f"After Trim + High-pass + VAD  "
        f"(kept {len(vad)/sr:.2f}s / {len(raw)/sr:.2f}s = "
        f"{100*len(vad)/max(len(raw),1):.1f}%)",
        fontsize=9, color=TEXT, pad=4
    )
    axes[1].set_ylabel("Amplitude", fontsize=8)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].grid(True, linewidth=0.4)
    axes[1].axhline(0, color=GRID, linewidth=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Amplitude normalisation before / after
# ─────────────────────────────────────────────────────────────────────────────

def plot_normalisation(stages: dict, title: str, save_path: Path):
    """
    Two-panel figure: VAD waveform before normalisation (top) vs
    after global peak normalisation (bottom).
    Annotates peak amplitude values to show the effect clearly.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), facecolor=BG)
    fig.suptitle(f"Amplitude Normalisation\n{title}",
                 fontsize=11, color=TEXT, y=1.01)

    sr   = stages["sr"]
    pre  = stages["vad"]
    post = stages["normalised"]

    t = np.linspace(0, len(pre) / sr, len(pre))

    peak_pre  = np.abs(pre).max()
    peak_post = np.abs(post).max()

    axes[0].plot(t, pre, color="#FF9800", linewidth=0.5, alpha=0.9)
    axes[0].set_title(f"Before Normalisation  (peak = {peak_pre:.4f})",
                      fontsize=9, color=TEXT, pad=4)
    axes[0].set_ylabel("Amplitude", fontsize=8)
    axes[0].set_xlabel("Time (s)", fontsize=8)
    axes[0].axhline(peak_pre,  color="#FF5252", linewidth=0.8,
                    linestyle="--", label=f"peak +{peak_pre:.3f}")
    axes[0].axhline(-peak_pre, color="#FF5252", linewidth=0.8, linestyle="--")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, linewidth=0.4)

    axes[1].plot(t, post, color="#4CAF50", linewidth=0.5, alpha=0.9)
    axes[1].set_title(f"After Global Peak Normalisation  (peak = {peak_post:.4f})",
                      fontsize=9, color=TEXT, pad=4)
    axes[1].set_ylabel("Amplitude", fontsize=8)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].axhline(1.0,  color="#69F0AE", linewidth=0.8,
                    linestyle="--", label="peak ±1.0")
    axes[1].axhline(-1.0, color="#69F0AE", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, linewidth=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Duration summary per group
# ─────────────────────────────────────────────────────────────────────────────

def plot_duration_summary(df: pd.DataFrame, audio_root: Path,
                          save_path: Path, n_sample: int = 60):
    """
    Three-panel figure per group:
      Left  – average duration bar chart (per audio type)
      Centre – file count bar chart (per audio type)
      Right  – duration distribution violin / box plot (per group)

    Samples up to n_sample files per group to keep runtime manageable.
    """
    from src.utils.paths import COLUMN_TO_SUBFOLDER

    audio_cols = list(COLUMN_TO_SUBFOLDER.keys())
    groups     = ["Sept", "Fess", "Contr", "Tonsill"]

    # ── Collect durations ─────────────────────────────────────────────────
    records = []
    for _, row in df.iterrows():
        group = str(row.get("GROUP", "")).strip()
        # Normalise group name to match folder names
        group_map = {"FESS": "Fess", "Contr": "Contr",
                     "Sept": "Sept", "Tonsill": "Tonsill"}
        group = group_map.get(group, group)

        for col in audio_cols:
            if col not in df.columns:
                continue
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                path = resolve_path(str(val), PROJECT_ROOT, col=col)
                if not path.exists():
                    continue
                info = torchaudio.info(str(path))
                duration = info.num_frames / info.sample_rate
                records.append({
                    "group":    group,
                    "col":      col,
                    "duration": duration,
                })
            except Exception:
                continue

    if not records:
        print("  [WARNING] No audio files found for duration summary.")
        return

    dur_df = pd.DataFrame(records)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle("Audio Duration Summary by Group & Task",
                 fontsize=13, color=TEXT, y=1.005)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    for ax, group in zip(axes, groups):
        gdf = dur_df[dur_df["group"] == group]
        if gdf.empty:
            ax.set_title(f"{group} — no data", color=TEXT, fontsize=9)
            continue

        col_order = [c for c in audio_cols if c in gdf["col"].unique()]
        means  = gdf.groupby("col")["duration"].mean().reindex(col_order)
        counts = gdf.groupby("col")["duration"].count().reindex(col_order)

        color = PALETTE.get(group, "#90CAF9")
        x     = np.arange(len(col_order))
        width = 0.38

        ax2 = ax.twinx()

        bars1 = ax.bar(x - width / 2, means.values, width,
                       color=color, alpha=0.85, label="Avg duration (s)")
        bars2 = ax2.bar(x + width / 2, counts.values, width,
                        color="#B0BEC5", alpha=0.6, label="File count")

        ax.set_title(f"{group}", fontsize=11, color=color, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(col_order, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Avg Duration (s)", fontsize=8, color=color)
        ax2.set_ylabel("File Count", fontsize=8, color="#B0BEC5")
        ax.yaxis.label.set_color(color)
        ax2.yaxis.label.set_color("#B0BEC5")
        ax.grid(True, axis="y", linewidth=0.4)
        ax.set_facecolor(PANEL_BG)
        ax2.set_facecolor(PANEL_BG)

        # Annotate avg duration on bars
        for bar, val in zip(bars1, means.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f"{val:.1f}s", ha="center", va="bottom",
                        fontsize=6, color=TEXT)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=6, loc="upper right")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {save_path.name}")

    # ── Distribution violin plot (all groups combined) ────────────────────
    fig2, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    fig2.suptitle("Duration Distribution by Group",
                  fontsize=12, color=TEXT)

    parts = ax.violinplot(
        [dur_df[dur_df["group"] == g]["duration"].values
         for g in groups if not dur_df[dur_df["group"] == g].empty],
        positions=range(len(groups)),
        showmedians=True,
        showextrema=True,
    )

    for i, (pc, g) in enumerate(zip(parts["bodies"], groups)):
        pc.set_facecolor(PALETTE.get(g, "#90CAF9"))
        pc.set_alpha(0.7)

    parts["cmedians"].set_color(TEXT)
    parts["cmins"].set_color(GRID)
    parts["cmaxes"].set_color(GRID)
    parts["cbars"].set_color(GRID)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Duration (s)", fontsize=9)
    ax.set_xlabel("Group", fontsize=9)
    ax.grid(True, axis="y", linewidth=0.4)

    # Annotate median and mean
    for i, g in enumerate(groups):
        gdata = dur_df[dur_df["group"] == g]["duration"]
        if gdata.empty:
            continue
        ax.text(i, gdata.max() + 0.3,
                f"n={len(gdata)}\nμ={gdata.mean():.1f}s",
                ha="center", va="bottom", fontsize=7, color=TEXT)

    dist_path = save_path.parent / "duration_distribution_violin.png"
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig2)
    print(f"  Saved → {dist_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(n_samples: int = 2):
    csv_path = (PROJECT_ROOT / "Data" / "data_final" /
                "Clinical" / "clinical_all_sessions.csv")

    df = pd.read_csv(csv_path)
    audio_cols = list(df.columns[df.columns.get_loc("TIME") + 1: -1])

    groups     = df["GROUP"].unique()
    group_map  = {"FESS": "Fess", "Contr": "Contr",
                  "Sept": "Sept", "Tonsill": "Tonsill"}

    print("\n── Preprocessing visualisations ────────────────────────────")

    plotted = {g: 0 for g in group_map}

    for _, row in df.iterrows():
        raw_group = str(row.get("GROUP", "")).strip()
        group     = group_map.get(raw_group, raw_group)

        if plotted.get(group, 0) >= n_samples:
            continue

        for col in audio_cols:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                path = resolve_path(str(val), PROJECT_ROOT, col=col)
                if not path.exists():
                    continue

                print(f"\n  [{group}] {col} → {path.name}")
                stages = load_stages(str(path))
                label  = f"{group} | {col} | ID {row['ID']} ses{row['session']}"
                safe   = f"{group}_{col}_ID{row['ID']}_ses{row['session']}"

                # Plot 1: raw vs cleaned
                plot_raw_vs_cleaned(
                    stages, label,
                    OUTPUT_DIR / f"waveform_raw_vs_clean_{safe}.png"
                )

                # Plot 2: normalisation
                plot_normalisation(
                    stages, label,
                    OUTPUT_DIR / f"normalisation_{safe}.png"
                )

                plotted[group] = plotted.get(group, 0) + 1
                break   # one col per row is enough; move to next speaker

            except Exception as e:
                print(f"    [SKIP] {e}")
                continue

        if all(v >= n_samples for v in plotted.values()):
            break

    # Plot 3: duration summary (uses full CSV)
    print("\n── Duration summary plots ──────────────────────────────────")
    plot_duration_summary(
        df,
        PROJECT_ROOT / "Data" / "data_final" / "Audios",
        OUTPUT_DIR / "duration_summary_by_group.png",
    )

    print(f"\nAll plots saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2,
                        help="Number of files to visualise per group")
    args = parser.parse_args()
    main(n_samples=args.n_samples)