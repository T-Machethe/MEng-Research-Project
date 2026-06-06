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
# ── Academic palette (matches results/thesis figures) ─────────────────────
BG       = "#FFFFFF"
PANEL    = "#F7F9FC"
BORDER   = "#BDC3C7"
TEXT     = "#1C2833"
MUTED    = "#6B7280"
ACCENT   = "#1A237E"

# Group colours — distinct, print-safe
PALETTE = {
    "Sept":    "#1565C0",   # royal blue
    "Fess":    "#C62828",   # deep red
    "Contr":   "#2E7D32",   # forest green
    "Tonsill": "#E65100",   # burnt orange
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "text.color":        TEXT,
    "font.family":       "sans-serif",
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
    "boxplot.whiskerprops.color":  MUTED,
    "boxplot.capprops.color":      MUTED,
    "boxplot.medianprops.color":   ACCENT,
    "boxplot.flierprops.color":    MUTED,
    "boxplot.flierprops.markeredgecolor": MUTED,
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
        ax.set_facecolor(PANEL)

        # Annotate median per column
        for i, col_data in enumerate(data_by_col, 1):
            if len(col_data):
                med = np.median(col_data)
                ax.text(i, med + 0.1, f"{med:.1f}s",
                        ha="center", va="bottom", fontsize=6, color=MUTED)

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
    ax2.set_facecolor(PANEL)

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
        ax.set_facecolor(PANEL)

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
        ax.set_facecolor(PANEL)
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
    ax2.set_facecolor(PANEL)

    plt.tight_layout()
    p2 = OUTPUT_DIR / "eda_vad_keeprate_summary.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig2)
    print(f"  Saved → {p2.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Patient and session counts  (no audio scan needed)
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_GROUPS = {
    "Vowels":    ["a","e","i","o","u"],
    "Sustained": ["a1","a2","a3"],
    "Speech":    ["speech"],
    "TDU Words": ["agua","brasero","dia","mesa"],
}
CHANNEL_COLORS = {
    "Vowels":"#4FC3F7","Sustained":"#81C784","Speech":"#FFB74D","TDU Words":"#CE93D8"
}


def plot_patient_session_counts(df: pd.DataFrame):
    df = df.copy()
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])
    audio_cols   = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle("Dataset Structure — Patients, Sessions, Recordings",
                 fontsize=13, color=TEXT, y=1.02)

    # Panel 1: unique patients per group
    ax = axes[0]
    pat_counts = df.groupby("_group")["ID"].nunique().reindex(GROUPS).fillna(0)
    colors     = [PALETTE.get(g, "#90CAF9") for g in GROUPS]
    bars       = ax.bar(GROUPS, pat_counts.values, color=colors, alpha=0.85)
    for bar, val in zip(bars, pat_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(int(val)), ha="center", va="bottom",
                fontsize=11, color=TEXT, fontweight="bold")
    ax.set_title("Unique Patients per Group", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel("Patient Count", fontsize=9); ax.grid(True,axis="y",linewidth=0.4)

    # Panel 2: recordings per session per group
    ax = axes[1]
    session_counts = df.groupby(["_group","session"]).size().unstack(fill_value=0)
    sessions   = sorted(session_counts.columns)
    x = np.arange(len(GROUPS)); w = 0.25
    ses_colors = ["#4FC3F7","#81C784","#FFB74D"]
    for si, (ses, sc) in enumerate(zip(sessions, ses_colors)):
        vals = [session_counts.loc[g, ses] if g in session_counts.index else 0
                for g in GROUPS]
        ax.bar(x + si*w, vals, w, label=f"Session {ses}",
               color=sc, alpha=0.85)
    ax.set_title("Recordings per Session per Group", fontsize=11, color=TEXT, pad=8)
    ax.set_xticks(x + w); ax.set_xticklabels(GROUPS, fontsize=10)
    ax.set_ylabel("Recording Count", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True,axis="y",linewidth=0.4)

    # Panel 3: non-empty cells per audio channel
    ax = axes[2]
    cols_ord = [c for grp in CHANNEL_GROUPS.values() for c in grp if c in df.columns]
    ch_vals  = [int(df[c].notna().sum()) for c in cols_ord]
    ch_colors = []
    for c in cols_ord:
        for grp_name, members in CHANNEL_GROUPS.items():
            if c in members:
                ch_colors.append(CHANNEL_COLORS[grp_name]); break
    bars = ax.bar(cols_ord, ch_vals, color=ch_colors, alpha=0.85)
    for bar, val in zip(bars, ch_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(val), ha="center", va="bottom", fontsize=7.5, color=TEXT)
    ax.set_title("Non-empty Recordings per Audio Channel", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel("Recording Count", fontsize=9)
    ax.set_xticklabels(cols_ord, rotation=45, ha="right", fontsize=9)
    ax.grid(True,axis="y",linewidth=0.4)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CHANNEL_COLORS[l],label=l,alpha=0.85)
               for l in CHANNEL_GROUPS]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    for a in axes: a.set_facecolor(PANEL)
    plt.tight_layout()
    p = OUTPUT_DIR / "eda_patient_session_counts.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Class balance per experiment  (no audio scan needed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_balance(df: pd.DataFrame):
    df = df.copy()
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])
    audio_cols   = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]

    def count_recs(mask):
        sub = df[mask]
        return int(sub[audio_cols].notna().sum().sum())

    # ── Exp 1: FESS Session 1 only vs all Control sessions ────────────────
    # (Exp 1 uses pre-op FESS recordings as the positive class; all three
    #  control sessions are included as the negative class.)
    exp_data = [
        ("Exp 1: CRS vs Control",
         [("FESS\n(Session 1 only)",  count_recs((df["_group"]=="Fess")&(df["session"]==1))),
          ("Control\n(all sessions)", count_recs(df["_group"]=="Contr"))],
         ["#E91E63","#4CAF50"],
         None),
        ("Exp 2: Pre-op vs Post-op (FESS)",
         [("Session 1\n(pre-op)",  count_recs((df["_group"]=="Fess")&(df["session"]==1))),
          ("Session 2+3\n(post)",  count_recs((df["_group"]=="Fess")&(df["session"]>1)))],
         ["#4FC3F7","#FF9800"],
         None),
        ("Exp 3: 3-class Trajectory (FESS)",
         [("Session 1", count_recs((df["_group"]=="Fess")&(df["session"]==1))),
          ("Session 2", count_recs((df["_group"]=="Fess")&(df["session"]==2))),
          ("Session 3", count_recs((df["_group"]=="Fess")&(df["session"]==3)))],
         ["#4FC3F7","#81C784","#FFB74D"],
         None),
        # ── Exp 4: Paired within-patient pre/post (FESS only) ─────────────
        # Recording-level balance is near-equal (353 vs 352).
        # Segment-level balance is ~30/70 pre/post due to paired windowing
        # (post-op segments are drawn from two sessions, pre-op from one).
        ("Exp 4: Paired Pre/Post (FESS)",
         [("Session 1\n(pre-op)",  count_recs((df["_group"]=="Fess")&(df["session"]==1))),
          ("Session 2\n(post-op)", count_recs((df["_group"]=="Fess")&(df["session"]==2)))],
         ["#9C27B0","#FF5722"],
         "Segment-level balance ~30/70 (pre/post)\ndue to paired windowing"),
        ("Exp 5: Generalisation",
         [("FESS (train)",         count_recs(df["_group"]=="Fess")),
          ("Septoplasty (test)",   count_recs(df["_group"]=="Sept")),
          ("Tonsillectomy (test)", count_recs(df["_group"]=="Tonsill"))],
         ["#E91E63","#2196F3","#FF9800"],
         None),
    ]

    # ── 2×3 grid — 5 panels, 6th cell hidden ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor=BG)
    fig.suptitle("Class Balance per Experiment (Recording Level)",
                 fontsize=13, color=TEXT, y=1.02)

    flat_axes = axes.flatten()
    for ax, (title, class_data, colors, footnote) in zip(flat_axes, exp_data):
        labels = [c[0] for c in class_data]
        values = [c[1] for c in class_data]
        total  = sum(values) or 1
        y      = np.arange(len(labels))
        ax.barh(y, values, color=colors[:len(labels)], alpha=0.85)
        for i, val in enumerate(values):
            ax.text(val + total * 0.01, i,
                    f"{val:,}  ({100 * val / total:.1f}%)",
                    va="center", fontsize=9, color=TEXT)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Recording Count", fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, pad=6)
        ax.set_xlim(0, max(values) * 1.40)
        ax.grid(True, axis="x", linewidth=0.4)
        ax.set_facecolor(PANEL)
        if footnote:
            ax.text(0.5, -0.18, footnote,
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color="#757575", style="italic")

    # Hide the unused 6th cell
    flat_axes[-1].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "eda_class_balance.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p.name}")



# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Age and gender distribution per group  (CSV only, fast)
# ─────────────────────────────────────────────────────────────────────────────

def plot_demographics(df: pd.DataFrame):
    """Age histogram and gender bar chart per surgical group."""
    df = df.copy()
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])

    # Detect age / gender column names (CUCO uses various spellings)
    age_col, gen_col = None, None
    for c in df.columns:
        cl = c.lower()
        if cl in ("age","edad","años") and age_col is None:
            age_col = c
        if cl in ("gender","sexo","sex","género") and gen_col is None:
            gen_col = c

    has_age = age_col is not None and df[age_col].notna().sum() > 0
    has_gen = gen_col is not None and df[gen_col].notna().sum() > 0

    if not has_age and not has_gen:
        print("  [SKIP] No Age or Gender columns found in CSV.")
        return

    n_panels = sum([has_age, has_gen, has_age and has_gen])
    fig, axes = plt.subplots(1, 2 if has_age and has_gen else 1,
                             figsize=(16 if has_age and has_gen else 8, 6),
                             facecolor=BG)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    fig.suptitle("Demographic Distribution by Surgical Group",
                 fontsize=13, color=TEXT, y=1.02)

    # Deduplicate to one row per patient
    pat_df = df.drop_duplicates(subset=["ID"])
    pat_df["_group"] = pat_df["GROUP"].map(GROUP_MAP).fillna(pat_df["GROUP"])

    ax_idx = 0

    if has_age:
        ax = axes[ax_idx]; ax_idx += 1
        ax.set_facecolor(PANEL)
        for group in GROUPS:
            ages = pat_df[pat_df["_group"]==group][age_col].dropna()
            if ages.empty: continue
            ax.hist(ages.values, bins=10, alpha=0.6,
                    color=PALETTE.get(group,"#90CAF9"), label=group, edgecolor="none")
        ax.set_xlabel("Age (years)", fontsize=10, color=TEXT)
        ax.set_ylabel("Number of Patients", fontsize=10, color=TEXT)
        ax.set_title("Age Distribution per Group", fontsize=11, color=TEXT, pad=8)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, axis="y", linewidth=0.4)

        # Print stats
        print("\n  Age statistics per group:")
        for group in GROUPS:
            ages = pat_df[pat_df["_group"]==group][age_col].dropna()
            if not ages.empty:
                print(f"    {group:<10} n={len(ages)}  "
                      f"mean={ages.mean():.1f}  std={ages.std():.1f}  "
                      f"min={ages.min():.0f}  max={ages.max():.0f}")

    if has_gen:
        ax = axes[ax_idx]; ax_idx += 1
        ax.set_facecolor(PANEL)
        gender_counts = (pat_df.groupby(["_group", gen_col])
                               .size()
                               .unstack(fill_value=0)
                               .reindex(GROUPS, fill_value=0))
        genders = gender_counts.columns.tolist()
        x = np.arange(len(GROUPS)); w = 0.35
        gen_colors = ["#4FC3F7","#F48FB1","#A5D6A7","#FFE082"]
        for gi, gen in enumerate(genders):
            bars = ax.bar(x + gi*w - w*(len(genders)-1)/2,
                          gender_counts[gen].values, w,
                          label=str(gen),
                          color=gen_colors[gi % len(gen_colors)],
                          alpha=0.85)
            for bar, val in zip(bars, gender_counts[gen].values):
                if val > 0:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                            str(int(val)), ha="center", va="bottom",
                            fontsize=9, color=TEXT, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(GROUPS, fontsize=10)
        ax.set_ylabel("Number of Patients", fontsize=10, color=TEXT)
        ax.set_title("Gender Distribution per Group", fontsize=11, color=TEXT, pad=8)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, axis="y", linewidth=0.4)

        # Check for single-gender groups
        for group in GROUPS:
            if group in gender_counts.index:
                non_zero = (gender_counts.loc[group] > 0).sum()
                if non_zero == 1:
                    print(f"  ⚠  {group} appears to be single-gender "
                          f"({gender_counts.loc[group].idxmax()})")

    plt.tight_layout()
    p = OUTPUT_DIR / "eda_demographics.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Session completion heatmap  (CSV only, fast)
# Shows which patients have recordings in which sessions.
# Critical for Exp4 (paired design needs pre AND post).
# ─────────────────────────────────────────────────────────────────────────────

def plot_session_completion(df: pd.DataFrame):
    """Patient × session completion matrix — one row per patient."""
    df = df.copy()
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])
    audio_cols = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(20, 8), facecolor=BG)
    fig.suptitle("Session Completion Matrix — Patient × Session\n"
                 "(green = ≥1 recording present, red = missing)",
                 fontsize=13, color=TEXT, y=1.02)

    for ax, group in zip(axes, GROUPS):
        gdf     = df[df["_group"]==group]
        pat_ids = sorted(gdf["ID"].unique())
        sessions = sorted(gdf["session"].dropna().unique().astype(int))

        mat = np.zeros((len(pat_ids), len(sessions)))
        for pi, pid in enumerate(pat_ids):
            for si, ses in enumerate(sessions):
                sub = gdf[(gdf["ID"]==pid) & (gdf["session"]==ses)]
                n_recs = int(sub[audio_cols].notna().sum().sum()) if not sub.empty else 0
                mat[pi, si] = min(n_recs, 1)   # 1 = present, 0 = missing

        cmap = plt.matplotlib.colors.ListedColormap(["#C62828","#2E7D32"])
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels([f"Ses {s}" for s in sessions], fontsize=9)
        ax.set_yticks(range(len(pat_ids)))
        ax.set_yticklabels([str(p) for p in pat_ids], fontsize=7)
        ax.set_title(f"{group}  (n={len(pat_ids)})",
                     fontsize=10, color=PALETTE.get(group,TEXT), pad=6)

        # Completion stats
        complete = sum(1 for pi in range(len(pat_ids))
                      if mat[pi].sum() == len(sessions))
        print(f"  {group:<10} {complete}/{len(pat_ids)} patients with all "
              f"{len(sessions)} sessions")
        if group == "Fess":
            paired = sum(1 for pi in range(len(pat_ids))
                        if mat[pi,0]>0 and mat[pi,1:].sum()>0)
            print(f"  {group:<10} {paired}/{len(pat_ids)} patients with "
                  f"pre AND post-op (usable for Exp4 paired design)")

    plt.tight_layout()
    p = OUTPUT_DIR / "eda_session_completion.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — Dataset balance summary table  (CSV only, fast)
# Produces the formal n-patients / n-recordings table matching CUCO paper
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_summary_table(df: pd.DataFrame):
    """Print and save a booktabs-style summary table (CSV only)."""
    df = df.copy()
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])
    audio_cols = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]

    rows = []
    for group in GROUPS:
        for ses in sorted(df["session"].dropna().unique().astype(int)):
            sub = df[(df["_group"]==group) & (df["session"]==ses)]
            n_pats  = sub["ID"].nunique()
            n_recs  = int(sub[audio_cols].notna().sum().sum())
            mean_r  = n_recs / n_pats if n_pats > 0 else 0
            rows.append({"Group":group,"Session":ses,
                         "Patients":n_pats,"Recordings":n_recs,
                         "Recs/Patient":f"{mean_r:.1f}"})
    summary = pd.DataFrame(rows)

    # Console table
    print("\n" + "="*60)
    print("  DATASET SUMMARY  (matches CUCO paper Table style)")
    print("="*60)
    print(summary.to_string(index=False))

    # Total row
    total_pats = df["ID"].nunique()
    total_recs = int(df[audio_cols].notna().sum().sum())
    print(f"\n  TOTAL: {total_pats} patients, {total_recs} recordings "
          f"({total_recs/total_pats:.2f} ± — per patient)")

    # Save as figure table
    fig, ax = plt.subplots(figsize=(12, max(4, len(rows)*0.4+1.5)), facecolor=BG)
    ax.axis("off"); ax.set_facecolor(BG)
    tbl = ax.table(
        cellText  = summary.values,
        colLabels = summary.columns,
        loc       = "center",
        cellLoc   = "center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL if row > 0 else BG)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(
            color=TEXT,
            fontweight="bold" if row == 0 else "normal"
        )
    ax.set_title("Dataset Balance Summary",
                 fontsize=12, color=TEXT, pad=16, loc="center")
    plt.tight_layout()
    p = OUTPUT_DIR / "eda_dataset_summary_table.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p.name}")

    # Also save as CSV for LaTeX table generation
    p_csv = OUTPUT_DIR / "eda_dataset_summary.csv"
    summary.to_csv(str(p_csv), index=False)
    print(f"  CSV   → {p_csv.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9 — Acoustic feature EDA: F0, jitter, shimmer, HNR
# Requires audio scan + parselmouth (pip install praat-parselmouth)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_acoustic_features(path: str) -> dict:
    """Extract F0, jitter, shimmer, HNR via parselmouth (Praat wrapper)."""
    try:
        import parselmouth
        from parselmouth.praat import call
        snd  = parselmouth.Sound(str(path))
        # Pitch
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        # Point process for jitter/shimmer
        pp    = call([snd, pitch], "To PointProcess (cc)")
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        # Harmonicity (HNR)
        harm   = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr    = call(harm, "Get mean", 0, 0)
        return {"f0": f0_mean if f0_mean == f0_mean else None,  # NaN check
                "jitter": jitter, "shimmer": shimmer, "hnr": hnr}
    except Exception:
        return {"f0": None, "jitter": None, "shimmer": None, "hnr": None}


def plot_acoustic_features(df: pd.DataFrame, max_files: int = 9999):
    """
    Extract and plot F0, jitter, shimmer, HNR per group for sustained vowel /a/.
    Only runs if parselmouth is installed.
    Uses vowel /a/ Session 1 only to match CUCO paper baseline analysis.
    """
    try:
        import parselmouth  # noqa: F401
    except ImportError:
        print("  [SKIP] parselmouth not installed. "
              "Run: pip install praat-parselmouth")
        print("  Acoustic EDA requires this library for F0/jitter/shimmer/HNR.")
        return

    audio_cols = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]
    df = df.copy()
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])

    # Restrict to vowel /a/, session 1 for baseline comparison
    target_col = "a"
    records = []
    scanned = 0

    print(f"  Extracting acoustic features from vowel /a/ Session 1 "
          f"(up to {max_files} files)...")

    for _, row in df[df["session"]==1].iterrows():
        group = row["_group"]
        val   = row.get(target_col)
        if pd.isna(val) or str(val).strip() == "":
            continue
        try:
            path = resolve_path(str(val), DATA_ROOT, col=target_col)
            if not path.exists():
                continue
            feats = _compute_acoustic_features(str(path))
            feats["group"] = group
            feats["id"]    = row.get("ID")
            records.append(feats)
            scanned += 1
            if scanned % 20 == 0:
                print(f"  ...{scanned} files processed")
            if scanned >= max_files:
                break
        except Exception:
            continue

    if not records:
        print("  No acoustic features extracted.")
        return

    feat_df = pd.DataFrame(records)
    print(f"  Extracted features from {len(feat_df)} files.")

    metrics  = [("f0",      "F0 (Hz)",        "Fundamental Frequency"),
                ("jitter",  "Jitter (%)",      "Local Jitter"),
                ("shimmer", "Shimmer (%)",      "Local Shimmer"),
                ("hnr",     "HNR (dB)",         "Harmonics-to-Noise Ratio")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    fig.suptitle("Acoustic Feature Distributions — Vowel /a/, Session 1 (Pre-op)\n"
                 "Comparison across surgical groups",
                 fontsize=12, color=TEXT, y=1.02)

    for ax, (col, ylabel, title) in zip(axes.flatten(), metrics):
        ax.set_facecolor(PANEL)
        data = [feat_df[feat_df["group"]==g][col].dropna().values for g in GROUPS]
        bp   = ax.boxplot(data, patch_artist=True, notch=False)
        for patch, group in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(PALETTE.get(group,"#90CAF9"))
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color(ACCENT); med.set_linewidth(2)
        for i, (d, group) in enumerate(zip(data, GROUPS), 1):
            if len(d):
                jit = np.random.uniform(-0.15, 0.15, len(d))
                ax.scatter(np.full(len(d), i)+jit, d,
                           alpha=0.45, s=15,
                           color=PALETTE.get(group,"#90CAF9"), zorder=3)
        ax.set_xticks(range(1, len(GROUPS)+1))
        ax.set_xticklabels(GROUPS, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, pad=6)
        ax.grid(True, axis="y", linewidth=0.4)

        # Print means
        for group, d in zip(GROUPS, data):
            if len(d):
                print(f"    {title} | {group}: "
                      f"mean={np.mean(d):.3f}  std={np.std(d):.3f}")

    plt.tight_layout()
    p = OUTPUT_DIR / "eda_acoustic_features.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p.name}")

    # Save raw features as CSV
    p_csv = OUTPUT_DIR / "eda_acoustic_features.csv"
    feat_df.to_csv(str(p_csv), index=False)
    print(f"  CSV   → {p_csv.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 10 — Duration consistency across sessions (per-patient paired)
# Requires audio scan
# ─────────────────────────────────────────────────────────────────────────────

def plot_duration_consistency(meta: pd.DataFrame):
    """
    For each FESS patient with both Session 1 and Session 2/3,
    scatter Session 1 mean duration vs Session 2/3 mean duration.
    Tests whether recording duration is stable within patients.
    """
    meta = meta.copy()
    meta["_group"] = meta["group"]
    fess = meta[meta["_group"] == "Fess"]

    ses1 = (fess[fess["session"]==1]
            .groupby("id")["raw_dur"].mean()
            .rename("ses1"))
    ses2 = (fess[fess["session"].isin([2,3])]
            .groupby("id")["raw_dur"].mean()
            .rename("ses2"))

    paired = pd.concat([ses1, ses2], axis=1).dropna()
    if paired.empty:
        print("  [SKIP] No paired duration data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.scatter(paired["ses1"], paired["ses2"],
               color=PALETTE["Fess"], alpha=0.7, s=60, edgecolors="white",
               linewidths=0.5, zorder=3)
    lim = [min(paired.min()), max(paired.max())]
    ax.plot(lim, lim, color=ACCENT, linestyle="--", linewidth=1.2,
            alpha=0.7, label="y = x (no change)")
    for idx, row in paired.iterrows():
        ax.annotate(str(idx), (row["ses1"], row["ses2"]),
                    fontsize=6.5, color=TEXT, alpha=0.6,
                    xytext=(2, 2), textcoords="offset points")
    corr = paired.corr().iloc[0,1]
    ax.set_xlabel("Session 1 Mean Duration (s)", fontsize=10, color=TEXT)
    ax.set_ylabel("Session 2/3 Mean Duration (s)", fontsize=10, color=TEXT)
    ax.set_title(f"FESS Patient Recording Duration: Session 1 vs Post-op\n"
                 f"Pearson r = {corr:.3f}  (n={len(paired)} patients)",
                 fontsize=11, color=TEXT, pad=10)
    ax.legend(fontsize=9); ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    p = OUTPUT_DIR / "eda_duration_consistency.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {p.name}")
    print(f"  Duration consistency (Pearson r): {corr:.3f}")



def main(max_files: int = 9999, csv_path: str = None,
         output_dir: str = None, data_root: str = None,
         counts_only: bool = False):
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

    # ── Fast plots (CSV only, no audio scan) ─────────────────────────────
    print("\n── EDA Plot 4: Patient and session counts ──────────────────")
    plot_patient_session_counts(df)

    print("\n── EDA Plot 5: Class balance per experiment ────────────────")
    plot_class_balance(df)

    print("\n── EDA Plot 6: Age and gender demographics ─────────────────")
    plot_demographics(df)

    print("\n── EDA Plot 7: Session completion matrix ───────────────────")
    plot_session_completion(df)

    print("\n── EDA Plot 8: Dataset summary table ──────────────────────")
    plot_dataset_summary_table(df)

    if counts_only:
        print(f"\n  --counts_only: skipping audio scan.\n  Plots saved to: {OUTPUT_DIR}")
        return

    # ── Slow plots (require audio scanning) ───────────────────────────────
    meta = collect_metadata(df, max_files=max_files)
    cache_path = OUTPUT_DIR / "eda_metadata_cache.csv"
    meta.to_csv(cache_path, index=False)
    print(f"\nMetadata cached → {cache_path}")

    print("\n── EDA Plot 1: Duration distribution ──────────────────────")
    plot_duration_distribution(meta)

    print("\n── EDA Plot 2: Missing files heatmap ──────────────────────")
    plot_missing_heatmap(meta)

    print("\n── EDA Plot 3: VAD keep-rate ───────────────────────────────")
    plot_vad_keeprate(meta)

    print("\n── EDA Plot 10: Duration consistency (Session 1 vs post-op) ─")
    plot_duration_consistency(meta)

    print("\n── EDA Plot 9: Acoustic features F0/jitter/shimmer/HNR ─────")
    print("   (requires parselmouth — pip install praat-parselmouth)")
    plot_acoustic_features(df, max_files=max_files)

    print(f"\nAll EDA plots saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files",   type=int, default=9999)
    parser.add_argument("--from_cache",  action="store_true",
                        help="Reload metadata cache and replot audio-scan plots")
    parser.add_argument("--counts_only", action="store_true",
                        help="Only produce plots that don't need audio scanning (~2 min)")
    parser.add_argument("--csv_path",  default=None)
    parser.add_argument("--data_root", default=None,
                        help="Drive root with Data/ folder (e.g. /content/drive/MyDrive)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save plots")
    args = parser.parse_args()

    if args.data_root:
        DATA_ROOT = Path(args.data_root)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        cache_path = OUTPUT_DIR / "eda_metadata_cache.csv"
        if not cache_path.exists():
            print(f"No cache at {cache_path}. Run without --from_cache first.")
            import sys; sys.exit(1)
        df = pd.read_csv(args.csv_path or str(
            PROJECT_ROOT/"Data"/"data_final"/"Clinical"/"clinical_all_sessions.csv"))
        meta = pd.read_csv(cache_path)
        print(f"Loaded {len(meta)} records from cache.")
        plot_patient_session_counts(df)
        plot_class_balance(df)
        plot_demographics(df)
        plot_session_completion(df)
        plot_dataset_summary_table(df)
        plot_duration_distribution(meta)
        plot_missing_heatmap(meta)
        plot_vad_keeprate(meta)
        plot_duration_consistency(meta)
    else:
        main(max_files=args.max_files,
             csv_path=args.csv_path,
             output_dir=args.output_dir,
             data_root=args.data_root,
             counts_only=args.counts_only)