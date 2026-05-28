"""
eda.py
─────────────────────────────────────────────────────────────────────────────
Exploratory Data Analysis for the sinusitis clinical audio dataset.

Produces publication-quality figures covering:
  1. Duration distribution per group and audio type (box + swarm)
  2. Missing file counts per group and column (heatmap)
  3. VAD keep-rate — how much audio survives VAD per group (bar + strip)

All figures saved to:
    Project Folder/Plots and visuals/eda/

Usage
─────
    python scripts/eda.py
    python scripts/eda.py --max_files 300    # limit files scanned (speed)
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT  # overridden by --data_root in Colab
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.cleaning import (
    remove_leading_silence,
    highpass_filter,
    voice_activity_detection,
)
from src.config import TARGET_SR
from src.utils.paths import resolve_path, COLUMN_TO_SUBFOLDER

# ── Style ─────────────────────────────────────────────────────────────────────
BG       = "#0B0E14"
PANEL_BG = "#13161F"
TEXT     = "#DDE1EE"
GRID     = "#252836"
ACCENT   = "#00BCD4"

PALETTE = {
    "Sept":    "#2196F3",
    "Fess":    "#E91E63",
    "Contr":   "#4CAF50",
    "Tonsill": "#FF9800",
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID,
    "boxplot.whiskerprops.color":  TEXT,
    "boxplot.capprops.color":      TEXT,
    "boxplot.medianprops.color":   ACCENT,
    "boxplot.flierprops.color":    TEXT,
    "boxplot.flierprops.markeredgecolor": TEXT,
})

OUTPUT_DIR = PROJECT_ROOT / "results" / "Plots and visuals" / "eda"  # overridden by --output_dir

GROUP_MAP = {
    "FESS":    "Fess",
    "Contr":   "Contr",
    "Sept":    "Sept",
    "Tonsill": "Tonsill",
}
GROUPS = ["Sept", "Fess", "Contr", "Tonsill"]


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_metadata(df: pd.DataFrame,
                     max_files: int = 9999) -> pd.DataFrame:
    """
    Scan every audio file referenced in the CSV and collect:
      - group, subject ID, session, column (task)
      - raw duration (seconds)
      - VAD keep duration (seconds)
      - whether the file exists on disk
    """
    audio_cols = [c for c in COLUMN_TO_SUBFOLDER.keys() if c in df.columns]
    records    = []
    scanned    = 0

    print(f"\nScanning up to {max_files} files for EDA metadata...")

    for _, row in df.iterrows():
        raw_group = str(row.get("GROUP", "")).strip()
        group     = GROUP_MAP.get(raw_group, raw_group)
        subj_id   = row.get("ID", "?")
        session   = row.get("session", "?")

        for col in audio_cols:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                records.append({
                    "group": group, "id": subj_id,
                    "session": session, "col": col,
                    "exists": False, "raw_dur": np.nan,
                    "vad_dur": np.nan, "vad_keep_pct": np.nan,
                })
                continue

            try:
                path = resolve_path(str(val), DATA_ROOT, col=col)
            except Exception:
                records.append({
                    "group": group, "id": subj_id,
                    "session": session, "col": col,
                    "exists": False, "raw_dur": np.nan,
                    "vad_dur": np.nan, "vad_keep_pct": np.nan,
                })
                continue

            if not path.exists():
                records.append({
                    "group": group, "id": subj_id,
                    "session": session, "col": col,
                    "exists": False, "raw_dur": np.nan,
                    "vad_dur": np.nan, "vad_keep_pct": np.nan,
                })
                continue

            # File exists — get duration and VAD stats
            try:
                info     = torchaudio.info(str(path))
                raw_dur  = info.num_frames / info.sample_rate

                # Run VAD to get keep duration
                wav, sr  = torchaudio.load(str(path))
                wav      = wav.mean(dim=0, keepdim=True)
                wav      = remove_leading_silence(wav, sr)
                if sr != TARGET_SR:
                    wav  = T.Resample(sr, TARGET_SR)(wav)
                    sr   = TARGET_SR
                wav      = highpass_filter(wav, sr)
                vad_wav  = voice_activity_detection(wav, sr)
                vad_dur  = vad_wav.shape[1] / sr
                keep_pct = 100.0 * vad_dur / max(raw_dur, 1e-6)

                records.append({
                    "group": group, "id": subj_id,
                    "session": session, "col": col,
                    "exists": True,
                    "raw_dur": raw_dur,
                    "vad_dur": vad_dur,
                    "vad_keep_pct": keep_pct,
                })
                scanned += 1
                if scanned % 50 == 0:
                    print(f"  ...{scanned} files scanned")
                if scanned >= max_files:
                    print(f"  Reached max_files={max_files}, stopping scan.")
                    return pd.DataFrame(records)

            except Exception as e:
                records.append({
                    "group": group, "id": subj_id,
                    "session": session, "col": col,
                    "exists": True, "raw_dur": np.nan,
                    "vad_dur": np.nan, "vad_keep_pct": np.nan,
                })

    print(f"  Done. {scanned} files scanned, {len(records)} records total.")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 1 — Duration distribution per group and audio type
# ─────────────────────────────────────────────────────────────────────────────

def plot_duration_distribution(meta: pd.DataFrame):
    """
    Grid of box plots: one subplot per group, x-axis = audio task (col),
    y-axis = raw duration (seconds).
    A second figure shows a combined strip + box across all groups.
    """
    exist = meta[meta["exists"] & meta["raw_dur"].notna()]
    if exist.empty:
        print("  [SKIP] No duration data available.")
        return

    audio_cols = sorted(exist["col"].unique())

    # ── Per-group subplots ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BG)
    fig.suptitle("Raw Duration Distribution by Group & Audio Task",
                 fontsize=13, color=TEXT, y=1.01)
    axes = axes.flatten()

    for ax, group in zip(axes, GROUPS):
        gdf = exist[exist["group"] == group]
        color = PALETTE.get(group, "#90CAF9")

        if gdf.empty:
            ax.set_title(f"{group} — no data", color=TEXT)
            continue

        cols_present = [c for c in audio_cols if c in gdf["col"].unique()]
        data_by_col  = [gdf[gdf["col"] == c]["raw_dur"].dropna().values
                        for c in cols_present]

        bp = ax.boxplot(
            data_by_col,
            patch_artist=True,
            notch=False,
            vert=True,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color(ACCENT)
            median.set_linewidth(1.8)

        # Overlay individual points (jittered)
        for i, (col_data, col_name) in enumerate(zip(data_by_col, cols_present), 1):
            jitter = np.random.uniform(-0.18, 0.18, len(col_data))
            ax.scatter(np.full_like(col_data, i) + jitter,
                       col_data, alpha=0.45, s=12,
                       color=color, zorder=3)

        ax.set_title(f"{group}  (n={len(gdf)} files)",
                     fontsize=10, color=color, pad=5)
        ax.set_xticks(range(1, len(cols_present) + 1))
        ax.set_xticklabels(cols_present, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Duration (s)", fontsize=8)
        ax.grid(True, axis="y", linewidth=0.4)
        ax.set_facecolor(PANEL_BG)

        # Annotate median per column
        for i, col_data in enumerate(data_by_col, 1):
            if len(col_data):
                med = np.median(col_data)
                ax.text(i, med + 0.1, f"{med:.1f}s",
                        ha="center", va="bottom", fontsize=6, color=ACCENT)

    plt.tight_layout()
    p = OUTPUT_DIR / "eda_duration_by_group_task.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {p.name}")

    # ── Cross-group comparison (one box per group) ────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor=BG)
    fig2.suptitle("Overall Duration Distribution Across Groups",
                  fontsize=12, color=TEXT)

    group_data = [exist[exist["group"] == g]["raw_dur"].dropna().values
                  for g in GROUPS]

    bp2 = ax2.boxplot(group_data, patch_artist=True, notch=False)
    for patch, g in zip(bp2["boxes"], GROUPS):
        patch.set_facecolor(PALETTE.get(g, "#90CAF9"))
        patch.set_alpha(0.6)
    for median in bp2["medians"]:
        median.set_color(ACCENT)
        median.set_linewidth(2)

    for i, (gdata, g) in enumerate(zip(group_data, GROUPS), 1):
        if len(gdata):
            jitter = np.random.uniform(-0.25, 0.25, len(gdata))
            ax2.scatter(np.full_like(gdata, i) + jitter,
                        gdata, alpha=0.35, s=10,
                        color=PALETTE.get(g, "#90CAF9"), zorder=3)
            ax2.text(i, np.max(gdata) + 0.3,
                     f"n={len(gdata)}\nμ={np.mean(gdata):.1f}s\nσ={np.std(gdata):.1f}s",
                     ha="center", va="bottom", fontsize=7, color=TEXT)

    ax2.set_xticks(range(1, len(GROUPS) + 1))
    ax2.set_xticklabels(GROUPS, fontsize=10)
    ax2.set_ylabel("Duration (s)", fontsize=9)
    ax2.grid(True, axis="y", linewidth=0.4)
    ax2.set_facecolor(PANEL_BG)

    plt.tight_layout()
    p2 = OUTPUT_DIR / "eda_duration_cross_group.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig2)
    print(f"  Saved → {p2.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 2 — Missing file heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_heatmap(meta: pd.DataFrame):
    """
    Heatmap: rows = GROUP, columns = audio task (col).
    Cell value = number of missing files.
    Red = many missing, green = none missing.
    """
    audio_cols = sorted(meta["col"].unique())

    # Build missing count matrix
    matrix = pd.DataFrame(index=GROUPS, columns=audio_cols, dtype=float)
    pct_matrix = pd.DataFrame(index=GROUPS, columns=audio_cols, dtype=float)

    for g in GROUPS:
        for col in audio_cols:
            subset  = meta[(meta["group"] == g) & (meta["col"] == col)]
            total   = len(subset)
            missing = (subset["exists"] == False).sum()
            matrix.loc[g, col]     = missing
            pct_matrix.loc[g, col] = (100.0 * missing / total) if total else 0.0

    matrix     = matrix.fillna(0).astype(float)
    pct_matrix = pct_matrix.fillna(0).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG)
    fig.suptitle("Missing Files per Group & Audio Task",
                 fontsize=12, color=TEXT, y=1.02)

    for ax, data, title, fmt in zip(
        axes,
        [matrix, pct_matrix],
        ["Count of Missing Files", "% Missing"],
        [".0f", ".1f"]
    ):
        im = ax.imshow(data.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=data.values.max() or 1)

        ax.set_xticks(range(len(audio_cols)))
        ax.set_xticklabels(audio_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels(GROUPS, fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, pad=6)
        ax.set_facecolor(PANEL_BG)

        # Annotate cells
        for r in range(len(GROUPS)):
            for c in range(len(audio_cols)):
                val = data.values[r, c]
                suffix = "%" if "%" in title else ""
                ax.text(c, r, f"{val:{fmt}}{suffix}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if val > data.values.max() * 0.5 else TEXT)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color=TEXT)
        cbar.ax.tick_params(labelcolor=TEXT, labelsize=7)

    plt.tight_layout()
    p = OUTPUT_DIR / "eda_missing_files_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 3 — VAD keep-rate per group
# ─────────────────────────────────────────────────────────────────────────────

def plot_vad_keeprate(meta: pd.DataFrame):
    """
    Bar chart + individual points showing what % of audio survives VAD
    for each group, broken down by audio task.

    A low keep-rate (e.g. <50%) in a particular task/group signals that
    either the VAD threshold is too aggressive or those recordings contain
    unusually long silence / noise periods.
    """
    vad_data = meta[meta["exists"] & meta["vad_keep_pct"].notna()]
    if vad_data.empty:
        print("  [SKIP] No VAD data collected.")
        return

    audio_cols = sorted(vad_data["col"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BG)
    fig.suptitle("VAD Keep-Rate: % of Audio Retained After Voice Activity Detection",
                 fontsize=12, color=TEXT, y=1.01)
    axes = axes.flatten()

    for ax, group in zip(axes, GROUPS):
        gdf   = vad_data[vad_data["group"] == group]
        color = PALETTE.get(group, "#90CAF9")

        if gdf.empty:
            ax.set_title(f"{group} — no data", color=TEXT)
            continue

        cols_present = [c for c in audio_cols if c in gdf["col"].unique()]
        means = [gdf[gdf["col"] == c]["vad_keep_pct"].mean()
                 for c in cols_present]
        stds  = [gdf[gdf["col"] == c]["vad_keep_pct"].std()
                 for c in cols_present]

        x = np.arange(len(cols_present))
        bars = ax.bar(x, means, color=color, alpha=0.7,
                      yerr=stds, capsize=4,
                      error_kw={"ecolor": TEXT, "linewidth": 0.8})

        # Individual data points
        for i, col_name in enumerate(cols_present):
            pts = gdf[gdf["col"] == col_name]["vad_keep_pct"].values
            jitter = np.random.uniform(-0.2, 0.2, len(pts))
            ax.scatter(np.full_like(pts, i) + jitter,
                       pts, alpha=0.5, s=14,
                       color="white", zorder=4)

        # Danger line at 50%
        ax.axhline(50, color="#FF5252", linewidth=1,
                   linestyle="--", alpha=0.7, label="50% threshold")

        ax.set_title(f"{group}  (n={len(gdf)} files)",
                     fontsize=10, color=color, pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(cols_present, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("VAD Keep-Rate (%)", fontsize=8)
        ax.set_ylim(0, 115)
        ax.grid(True, axis="y", linewidth=0.4)
        ax.set_facecolor(PANEL_BG)
        ax.legend(fontsize=6)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (std or 0) + 1.5,
                    f"{mean:.0f}%",
                    ha="center", va="bottom", fontsize=7, color=TEXT)

    plt.tight_layout()
    p = OUTPUT_DIR / "eda_vad_keeprate.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {p.name}")

    # ── Cross-group summary ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor=BG)
    fig2.suptitle("Overall VAD Keep-Rate by Group",
                  fontsize=11, color=TEXT)

    group_keep = [vad_data[vad_data["group"] == g]["vad_keep_pct"].dropna().values
                  for g in GROUPS]

    for i, (gdata, g) in enumerate(zip(group_keep, GROUPS)):
        color = PALETTE.get(g, "#90CAF9")
        if len(gdata) == 0:
            continue
        mean, std = np.mean(gdata), np.std(gdata)
        ax2.bar(i, mean, color=color, alpha=0.7,
                yerr=std, capsize=5,
                error_kw={"ecolor": TEXT, "linewidth": 1})
        jitter = np.random.uniform(-0.3, 0.3, len(gdata))
        ax2.scatter(np.full_like(gdata, i) + jitter,
                    gdata, alpha=0.35, s=12, color=color, zorder=3)
        ax2.text(i, mean + std + 2,
                 f"μ={mean:.0f}%\nσ={std:.0f}%\nn={len(gdata)}",
                 ha="center", va="bottom", fontsize=7, color=TEXT)

    ax2.axhline(50, color="#FF5252", linewidth=1.2,
                linestyle="--", label="50% threshold")
    ax2.set_xticks(range(len(GROUPS)))
    ax2.set_xticklabels(GROUPS, fontsize=10)
    ax2.set_ylabel("VAD Keep-Rate (%)", fontsize=9)
    ax2.set_ylim(0, 120)
    ax2.grid(True, axis="y", linewidth=0.4)
    ax2.legend(fontsize=8)
    ax2.set_facecolor(PANEL_BG)

    plt.tight_layout()
    p2 = OUTPUT_DIR / "eda_vad_keeprate_summary.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig2)
    print(f"  Saved → {p2.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(max_files: int = 9999, csv_path: str = None, output_dir: str = None, data_root: str = None):
    global OUTPUT_DIR, DATA_ROOT
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if data_root:
        DATA_ROOT = Path(data_root)
    if csv_path is None:
        csv_path = str(PROJECT_ROOT / "Data" / "data_final" /
                       "Clinical" / "clinical_all_sessions.csv")
    df = pd.read_csv(csv_path)

    # Collect metadata (this is the slow step — reads every audio file)
    meta = collect_metadata(df, max_files=max_files)

    # Cache to CSV so you don't have to rescan every time
    cache_path = OUTPUT_DIR / "eda_metadata_cache.csv"
    meta.to_csv(cache_path, index=False)
    print(f"\nMetadata cached -> {cache_path}")

    print("\n── EDA Plot 1: Duration distribution ──────────────────────")
    plot_duration_distribution(meta)

    print("\n── EDA Plot 2: Missing files heatmap ──────────────────────")
    plot_missing_heatmap(meta)

    print("\n── EDA Plot 3: VAD keep-rate ───────────────────────────────")
    plot_vad_keeprate(meta)

    print(f"\nAll EDA plots saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files",  type=int, default=9999)
    parser.add_argument("--from_cache", action="store_true")
    parser.add_argument(
        "--csv_path", default=None,
        help="Path to clinical_all_sessions.csv (defaults to PROJECT_ROOT/Data/...)"
    )
    parser.add_argument(
        "--data_root", default=None,
        help="Drive root containing the Data/ folder "
             "(e.g. /content/drive/MyDrive). Defaults to PROJECT_ROOT."
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save plots (defaults to PROJECT_ROOT/results/Plots and visuals/eda/). "
             "Set to a Drive path in Colab."
    )
    args = parser.parse_args()

    if args.data_root:
        DATA_ROOT = Path(args.data_root)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        cache_path = OUTPUT_DIR / "eda_metadata_cache.csv"
        if not cache_path.exists():
            print(f"No cache found at {cache_path}. Run without --from_cache first.")
            sys.exit(1)
        meta = pd.read_csv(cache_path)
        print(f"Loaded {len(meta)} records from cache.")
        plot_duration_distribution(meta)
        plot_missing_heatmap(meta)
        plot_vad_keeprate(meta)
    else:
        main(max_files=args.max_files,
             csv_path=args.csv_path,
             output_dir=args.output_dir,
             data_root=args.data_root)