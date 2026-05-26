"""
run_diagnostic_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Two-section diagnostic script:

SECTION A — Step-by-step preprocessing diagnostics
  Measures how much audio each preprocessing step removes, per group and
  audio type. Produces 4 plots and a console retention summary.

SECTION B — Window size and segment yield analysis
  Scans every audio file and reports duration statistics, then computes
  the number of 1-second segments that would be produced at 6 candidate
  window sizes. Used to justify the 1s window choice in the thesis.

All figures saved to:
    results/Plots and visuals/diagnostics/

Usage
─────
    python scripts/run_diagnostic_pipeline.py               # full scan
    python scripts/run_diagnostic_pipeline.py --max_files 80   # quick test
    python scripts/run_diagnostic_pipeline.py --from_cache     # replot only
    python scripts/run_diagnostic_pipeline.py --section b      # window analysis only
"""

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.cleaning import (
    remove_leading_silence,
    highpass_filter,
    voice_activity_detection,
)
from src.config import TARGET_SR, LEADING_TRIM_S, VAD_THRESHOLD
from src.utils.paths import resolve_path, COLUMN_TO_SUBFOLDER

# ── Style ──────────────────────────────────────────────────────────────────────
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
    "figure.facecolor": BG,   "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID, "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT, "xtick.color":      TEXT,
    "ytick.color":      TEXT, "grid.color":       GRID,
    "grid.linewidth":   0.5,  "text.color":       TEXT,
    "font.family":      "monospace",
    "legend.facecolor": PANEL_BG, "legend.edgecolor": GRID,
})

OUTPUT_DIR = PROJECT_ROOT / "results" / "Plots and visuals" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_MAP = {"FESS": "Fess", "Contr": "Contr", "Sept": "Sept", "Tonsill": "Tonsill"}
GROUPS    = ["Sept", "Fess", "Contr", "Tonsill"]

COL_ORDER      = ["a","e","i","o","u","a1","a2","a3","agua","brasero","dia","mesa","speech","un"]
WIN_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.56, 3.072]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Preprocessing pipeline diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def measure_steps(filepath: str) -> dict:
    waveform, sr = torchaudio.load(filepath)
    waveform = waveform.mean(dim=0, keepdim=True)
    raw_dur  = waveform.shape[1] / sr

    trimmed          = remove_leading_silence(waveform, sr, trim_seconds=LEADING_TRIM_S)
    after_trim_dur   = trimmed.shape[1] / sr

    if sr != TARGET_SR:
        trimmed = T.Resample(sr, TARGET_SR)(trimmed)
        sr      = TARGET_SR

    filtered         = highpass_filter(trimmed, sr)
    vad_out          = voice_activity_detection(filtered, sr, threshold=VAD_THRESHOLD)
    after_vad_dur    = vad_out.shape[1] / sr

    trim_removed_pct = 100.0 * (raw_dur - after_trim_dur) / max(raw_dur, 1e-9)
    vad_removed_pct  = 100.0 * (after_trim_dur - after_vad_dur) / max(after_trim_dur, 1e-9)
    total_kept_pct   = 100.0 * after_vad_dur / max(raw_dur, 1e-9)

    return {
        "raw_dur":            raw_dur,
        "after_trim_dur":     after_trim_dur,
        "after_vad_dur":      after_vad_dur,
        "trim_removed_pct":   trim_removed_pct,
        "vad_removed_pct":    vad_removed_pct,
        "total_kept_pct":     total_kept_pct,
        "sr":                 sr,
    }


def collect_diagnostics(df: pd.DataFrame, max_files: int = 9999) -> pd.DataFrame:
    audio_cols = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]
    records    = []
    scanned    = 0
    print(f"\nRunning step-by-step diagnostic on up to {max_files} files...")
    print(f"  Leading trim : {LEADING_TRIM_S}s")
    print(f"  VAD threshold: {VAD_THRESHOLD} RMS\n")

    for _, row in df.iterrows():
        group   = GROUP_MAP.get(str(row.get("GROUP","")).strip(), "Unknown")
        subj_id = row.get("ID","?")
        session = row.get("session","?")

        for col in audio_cols:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                path = resolve_path(str(val), PROJECT_ROOT, col=col)
            except Exception:
                continue
            if not path.exists():
                continue
            try:
                steps = measure_steps(str(path))
                records.append({"group": group, "id": subj_id,
                                 "session": session, "col": col,
                                 "file": path.name, **steps})
                scanned += 1
                if scanned % 50 == 0:
                    print(f"  ...{scanned} files processed")
                if scanned >= max_files:
                    print(f"  Reached max_files={max_files}, stopping.")
                    return pd.DataFrame(records)
            except Exception as e:
                print(f"  [SKIP] {path.name}: {e}")

    print(f"\n  Done. {scanned} files measured.")
    return pd.DataFrame(records)


def print_summary_a(diag: pd.DataFrame):
    print("\n" + "─"*72)
    print("  STEP-BY-STEP RETENTION SUMMARY")
    print("─"*72)
    print(f"  {'GROUP':<10} {'COL':<12} {'N':>4}  "
          f"{'RAW(s)':>7}  {'TRIM%':>7}  {'VAD%':>7}  {'KEPT%':>7}")
    print("─"*72)
    for group in GROUPS:
        gdf = diag[diag["group"] == group]
        if gdf.empty:
            continue
        for col in sorted(gdf["col"].unique()):
            cdf = gdf[gdf["col"] == col]
            print(f"  {group:<10} {col:<12} {len(cdf):>4}  "
                  f"{cdf['raw_dur'].mean():>7.2f}  "
                  f"{cdf['trim_removed_pct'].mean():>6.1f}%  "
                  f"{cdf['vad_removed_pct'].mean():>6.1f}%  "
                  f"{cdf['total_kept_pct'].mean():>6.1f}%")
        print(f"  {'':10} {'[ALL]':<12} {len(gdf):>4}  "
              f"{gdf['raw_dur'].mean():>7.2f}  "
              f"{gdf['trim_removed_pct'].mean():>6.1f}%  "
              f"{gdf['vad_removed_pct'].mean():>6.1f}%  "
              f"{gdf['total_kept_pct'].mean():>6.1f}%")
        print("─"*72)
    print(f"  {'OVERALL':<23} {len(diag):>4}  "
          f"{diag['raw_dur'].mean():>7.2f}  "
          f"{diag['trim_removed_pct'].mean():>6.1f}%  "
          f"{diag['vad_removed_pct'].mean():>6.1f}%  "
          f"{diag['total_kept_pct'].mean():>6.1f}%")
    print("═"*72 + "\n")


def plot_stacked_removal(diag: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    fig.suptitle("Audio Removed at Each Preprocessing Step by Group",
                 fontsize=12, color=TEXT)
    gps = [g for g in GROUPS if g in diag["group"].unique()]
    x   = np.arange(len(gps))
    w   = 0.5
    trim_v = [diag[diag["group"]==g]["trim_removed_pct"].mean() for g in gps]
    kept_v = [diag[diag["group"]==g]["total_kept_pct"].mean()   for g in gps]
    vad_v  = [100-t-k for t,k in zip(trim_v,kept_v)]
    ax.bar(x, trim_v, w, label="Removed by Trim", color="#4FC3F7", alpha=0.85)
    ax.bar(x, vad_v,  w, bottom=trim_v, label="Removed by VAD", color="#E91E63", alpha=0.85)
    ax.bar(x, kept_v, w, bottom=[t+v for t,v in zip(trim_v,vad_v)],
           label="Retained", color="#4CAF50", alpha=0.85)
    for i,(t,v,k) in enumerate(zip(trim_v,vad_v,kept_v)):
        if t>2: ax.text(i,t/2,f"{t:.1f}%",ha="center",va="center",fontsize=8,color=BG,fontweight="bold")
        if v>2: ax.text(i,t+v/2,f"{v:.1f}%",ha="center",va="center",fontsize=8,color=BG,fontweight="bold")
        ax.text(i,t+v+k/2,f"{k:.1f}%\nkept",ha="center",va="center",fontsize=8,color=BG,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(gps, fontsize=11)
    ax.set_ylabel("% of Original Audio",fontsize=9); ax.set_ylim(0,108)
    ax.legend(fontsize=8); ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)
    p = OUTPUT_DIR/"diag_stacked_removal_by_group.png"
    plt.tight_layout(); plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


def plot_retention_heatmap(diag: pd.DataFrame):
    audio_cols = sorted(diag["col"].unique())
    kept_mat = pd.DataFrame(index=GROUPS,columns=audio_cols,dtype=float)
    trim_mat = pd.DataFrame(index=GROUPS,columns=audio_cols,dtype=float)
    vad_mat  = pd.DataFrame(index=GROUPS,columns=audio_cols,dtype=float)
    for g in GROUPS:
        for col in audio_cols:
            sub = diag[(diag["group"]==g)&(diag["col"]==col)]
            if sub.empty: continue
            kept_mat.loc[g,col] = sub["total_kept_pct"].mean()
            trim_mat.loc[g,col] = sub["trim_removed_pct"].mean()
            vad_mat.loc[g,col]  = sub["vad_removed_pct"].mean()
    fig,axes = plt.subplots(1,3,figsize=(22,5),facecolor=BG)
    fig.suptitle("Retention / Removal Rates: Group × Audio Type",fontsize=12,color=TEXT,y=1.02)
    for ax,(data,title,cmap,vmin,vmax) in zip(axes,[
        (kept_mat.fillna(0).astype(float),"Total Kept (%)","RdYlGn",0,100),
        (trim_mat.fillna(0).astype(float),"Removed by Trim (%)","RdYlGn_r",0,20),
        (vad_mat.fillna(0).astype(float),"Removed by VAD (%)","RdYlGn_r",0,100),
    ]):
        im = ax.imshow(data.values,cmap=cmap,aspect="auto",vmin=vmin,vmax=vmax)
        ax.set_xticks(range(len(audio_cols))); ax.set_xticklabels(audio_cols,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS,fontsize=9)
        ax.set_title(title,fontsize=10,color=TEXT,pad=6); ax.set_facecolor(PANEL_BG)
        for r in range(len(GROUPS)):
            for c in range(len(audio_cols)):
                val = data.values[r,c]
                if not np.isnan(val):
                    ax.text(c,r,f"{val:.0f}%",ha="center",va="center",fontsize=7,
                            color="white" if val<vmax*0.4 else BG)
        cbar=plt.colorbar(im,ax=ax,shrink=0.8); cbar.ax.tick_params(labelcolor=TEXT,labelsize=7)
    plt.tight_layout()
    p = OUTPUT_DIR/"diag_retention_heatmap.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


def plot_per_col_boxplots(diag: pd.DataFrame):
    audio_cols = sorted(diag["col"].unique())
    fig,axes = plt.subplots(2,1,figsize=(16,10),facecolor=BG)
    fig.suptitle("Retention Distribution per Audio Type",fontsize=12,color=TEXT)
    for ax,col_name,color,title in [
        (axes[0],"trim_removed_pct","#4FC3F7",f"% Removed by Trim ({LEADING_TRIM_S}s) per Audio Type"),
        (axes[1],"vad_removed_pct","#E91E63", f"% Removed by VAD (threshold={VAD_THRESHOLD}) per Audio Type"),
    ]:
        data_by_col = [diag[diag["col"]==c][col_name].dropna().values for c in audio_cols]
        bp = ax.boxplot(data_by_col,patch_artist=True,notch=False,vert=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("#FFEB3B"); med.set_linewidth(2)
        for i,cdata in enumerate(data_by_col,1):
            if len(cdata):
                jitter = np.random.uniform(-0.18,0.18,len(cdata))
                ax.scatter(np.full_like(cdata,i)+jitter,cdata,alpha=0.4,s=12,color=color,zorder=3)
        ax.set_xticks(range(1,len(audio_cols)+1))
        ax.set_xticklabels(audio_cols,rotation=40,ha="right",fontsize=8)
        ax.set_ylabel("% Removed",fontsize=8); ax.set_title(title,fontsize=9,color=TEXT,pad=5)
        ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)
    plt.tight_layout()
    p = OUTPUT_DIR/"diag_per_col_boxplots.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


def plot_duration_vs_keeprate(diag: pd.DataFrame):
    fig,ax = plt.subplots(figsize=(12,7),facecolor=BG)
    fig.suptitle("Raw Duration vs Total Keep-Rate",fontsize=11,color=TEXT)
    for group in GROUPS:
        gdf   = diag[diag["group"]==group]
        color = PALETTE.get(group,"#90CAF9")
        ax.scatter(gdf["raw_dur"],gdf["total_kept_pct"],label=group,
                   color=color,alpha=0.55,s=20,edgecolors="none")
    ax.axhline(50,color="#FF5252",linewidth=1,linestyle="--",alpha=0.7,label="50% keep threshold")
    d_range    = np.linspace(0.2,diag["raw_dur"].max()+1,200)
    trim_curve = 100.0*LEADING_TRIM_S/d_range
    ax.plot(d_range,100-trim_curve,color="#4FC3F7",linewidth=1.2,linestyle="-.",
            label=f"Max kept if only trim ({LEADING_TRIM_S}s)")
    ax.set_xlabel("Raw Duration (s)",fontsize=9); ax.set_ylabel("Total Keep-Rate (%)",fontsize=9)
    ax.set_ylim(0,110); ax.set_xlim(left=0)
    ax.legend(fontsize=8,loc="lower right"); ax.grid(True,linewidth=0.4); ax.set_facecolor(PANEL_BG)
    plt.tight_layout()
    p = OUTPUT_DIR/"diag_duration_vs_keeprate.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Window size and segment yield analysis
# ══════════════════════════════════════════════════════════════════════════════

def collect_durations(df: pd.DataFrame, max_files: int = 9999) -> dict:
    """
    Scan every audio file and collect raw duration per column per group.
    Returns {col: {group: [dur, ...]}}
    """
    audio_cols = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]
    durations  = defaultdict(lambda: defaultdict(list))
    n_found = n_missing = n_skip = 0

    print(f"\nScanning up to {max_files} files for window analysis...")
    for _, row in df.iterrows():
        group = GROUP_MAP.get(str(row.get("GROUP","")).strip(), "Unknown")
        for col in audio_cols:
            rel_path = row.get(col)
            if pd.isna(rel_path) or str(rel_path).strip() == "":
                n_skip += 1
                continue
            try:
                path = resolve_path(str(rel_path), PROJECT_ROOT, col=col)
                if not path.exists():
                    n_missing += 1
                    continue
                wav, sr = torchaudio.load(str(path))
                dur = wav.shape[-1] / sr
                durations[col][group].append(dur)
                n_found += 1
                if n_found >= max_files:
                    print(f"  Reached max_files={max_files}, stopping.")
                    return dict(durations)
            except Exception:
                n_missing += 1

    print(f"  Found: {n_found}  |  Missing: {n_missing}  |  Skipped: {n_skip}")
    return dict(durations)


def print_duration_stats(durations: dict):
    cols = [c for c in COL_ORDER if c in durations]
    print("\n" + "="*100)
    print("  DURATION STATISTICS (seconds)")
    print("="*100)
    print(f"  {'col':<10} {'n':>5}  {'min':>6}  {'p10':>6}  {'median':>8}  "
          f"{'p90':>6}  {'max':>6}  {'<1s%':>7}  {'<2s%':>7}  {'<3s%':>7}")
    print("-"*100)
    for col in cols:
        all_durs = [d for g in durations[col] for d in durations[col][g]]
        if not all_durs: continue
        a = np.array(all_durs)
        print(f"  {col:<10} {len(a):>5}  {a.min():>6.2f}  {np.percentile(a,10):>6.2f}  "
              f"{np.median(a):>8.2f}  {np.percentile(a,90):>6.2f}  {a.max():>6.2f}  "
              f"{(a<1.0).mean()*100:>6.1f}%  {(a<2.0).mean()*100:>6.1f}%  "
              f"{(a<3.0).mean()*100:>6.1f}%")


def print_segment_yield(durations: dict):
    cols = [c for c in COL_ORDER if c in durations]
    print("\n" + "="*100)
    print("  MEAN SEGMENTS PER FILE  (50% overlap; files shorter than window → 1 padded segment)")
    print("="*100)
    hdr = f"  {'col':<10}" + "".join(f"  {w:.3f}s" for w in WIN_CANDIDATES)
    print(hdr); print("-"*len(hdr))
    for col in cols:
        all_durs = [d for g in durations[col] for d in durations[col][g]]
        if not all_durs: continue
        a   = np.array(all_durs)
        row = f"  {col:<10}"
        for w in WIN_CANDIDATES:
            segs = np.where(a>=w, np.floor((a-w)/(w/2)).astype(int)+1, 1)
            row += f"  {segs.mean():>6.1f}"
        print(row)


def print_window_analysis(durations: dict):
    all_flat = np.array([d for col in durations
                         for g in durations[col]
                         for d in durations[col][g]])
    print("\n" + "="*60)
    print("  WINDOW SIZE ANALYSIS  (% of files needing padding)")
    print("="*60)
    for w in WIN_CANDIDATES:
        pct   = (all_flat < w).mean() * 100
        n_pad = int((all_flat < w).sum())
        print(f"  {w:.3f}s:  {pct:5.1f}% need padding  ({n_pad} files)")


def plot_window_analysis(durations: dict):
    """
    Figure 1: Stacked bar — % needing padding vs not per window size
    Figure 2: Mean segments per file per channel (heatmap)
    """
    all_flat = np.array([d for col in durations
                         for g in durations[col]
                         for d in durations[col][g]])

    # ── Fig 1: padding need per window ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG)
    pcts_pad = [(all_flat < w).mean()*100 for w in WIN_CANDIDATES]
    pcts_ok  = [100-p for p in pcts_pad]
    x = np.arange(len(WIN_CANDIDATES))
    ax.bar(x, pcts_ok,  0.55, label="No padding needed", color="#4CAF50", alpha=0.85)
    ax.bar(x, pcts_pad, 0.55, bottom=pcts_ok, label="Needs padding", color="#E91E63", alpha=0.85)
    for i,(pad,ok) in enumerate(zip(pcts_pad,pcts_ok)):
        ax.text(i,ok+pad/2,f"{pad:.1f}%",ha="center",va="center",fontsize=9,color=BG,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{w:.3f}s" for w in WIN_CANDIDATES],fontsize=10)
    ax.set_ylabel("% of Files",fontsize=9); ax.set_ylim(0,110)
    ax.set_title("Proportion of Files Requiring Padding per Window Size",fontsize=11,color=TEXT,pad=8)
    ax.legend(fontsize=9); ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)
    plt.tight_layout()
    p = OUTPUT_DIR/"diag_window_padding_analysis.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")

    # ── Fig 2: Mean segments per file per channel ─────────────────────────
    cols = [c for c in COL_ORDER if c in durations]
    mat  = np.zeros((len(cols), len(WIN_CANDIDATES)))
    for ci,col in enumerate(cols):
        all_durs = [d for g in durations[col] for d in durations[col][g]]
        if not all_durs: continue
        a = np.array(all_durs)
        for wi,w in enumerate(WIN_CANDIDATES):
            segs = np.where(a>=w, np.floor((a-w)/(w/2)).astype(int)+1, 1)
            mat[ci,wi] = segs.mean()

    fig, ax = plt.subplots(figsize=(12,7), facecolor=BG)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(WIN_CANDIDATES)))
    ax.set_xticklabels([f"{w:.3f}s" for w in WIN_CANDIDATES], fontsize=10)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=9)
    ax.set_xlabel("Window Size", fontsize=10); ax.set_ylabel("Audio Channel", fontsize=10)
    ax.set_title("Mean Segments per File by Window Size and Channel\n(50% overlap; short files → 1 padded segment)",
                 fontsize=11, color=TEXT, pad=10)
    for r in range(len(cols)):
        for c in range(len(WIN_CANDIDATES)):
            ax.text(c, r, f"{mat[r,c]:.1f}", ha="center", va="center",
                    fontsize=9, color="white" if mat[r,c]>mat.max()*0.6 else TEXT,
                    fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Mean Segments per File", fontsize=9, color=TEXT)
    cbar.ax.tick_params(labelcolor=TEXT, labelsize=8)
    plt.tight_layout()
    p = OUTPUT_DIR/"diag_segment_yield_heatmap.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")

    # ── Fig 3: Duration distribution overlay with window thresholds ───────
    fig, ax = plt.subplots(figsize=(12,5), facecolor=BG)
    ax.hist(all_flat, bins=60, color=ACCENT, alpha=0.7, edgecolor="none",
            label=f"All files (n={len(all_flat)})")
    for w,col in zip([1.0,2.0,3.072], ["#69F0AE","#FF9800","#FF5252"]):
        ax.axvline(w, color=col, linewidth=1.8, linestyle="--",
                   label=f"{w:.3f}s  ({(all_flat<w).mean()*100:.1f}% need padding)")
    ax.set_xlabel("Raw File Duration (seconds)", fontsize=10)
    ax.set_ylabel("Number of Files", fontsize=10)
    ax.set_title("File Duration Distribution with Window Size Thresholds",
                 fontsize=11, color=TEXT, pad=8)
    ax.legend(fontsize=9); ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)
    plt.tight_layout()
    p = OUTPUT_DIR/"diag_duration_distribution_windows.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(max_files: int = 9999, section: str = "all"):
    csv_path = (PROJECT_ROOT / "Data" / "data_final" /
                "Clinical" / "clinical_all_sessions.csv")
    df = pd.read_csv(csv_path)

    if section in ("all", "a"):
        print("\n══ SECTION A: Preprocessing Pipeline Diagnostics ══")
        diag = collect_diagnostics(df, max_files=max_files)
        cache = OUTPUT_DIR/"diagnostic_cache.csv"
        diag.to_csv(cache, index=False)
        print(f"\nCached → {cache}")
        print_summary_a(diag)
        print("\n── Plot 1: Stacked removal ─"); plot_stacked_removal(diag)
        print("── Plot 2: Retention heatmap ─"); plot_retention_heatmap(diag)
        print("── Plot 3: Per-column boxplots ─"); plot_per_col_boxplots(diag)
        print("── Plot 4: Duration vs keep-rate ─"); plot_duration_vs_keeprate(diag)

    if section in ("all", "b"):
        print("\n══ SECTION B: Window Size and Segment Yield Analysis ══")
        durations = collect_durations(df, max_files=max_files)
        print_duration_stats(durations)
        print_segment_yield(durations)
        print_window_analysis(durations)
        print("\n── Window analysis plots ─")
        plot_window_analysis(durations)

    print(f"\nAll diagnostic outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files", type=int, default=9999)
    parser.add_argument("--from_cache", action="store_true")
    parser.add_argument("--section", choices=["all","a","b"], default="all",
                        help="'a'=preprocessing only, 'b'=window analysis only, 'all'=both")
    args = parser.parse_args()

    if args.from_cache:
        cache = OUTPUT_DIR/"diagnostic_cache.csv"
        if not cache.exists():
            print("No cache found. Run without --from_cache first.")
            sys.exit(1)
        diag = pd.read_csv(cache)
        print(f"Loaded {len(diag)} records from cache.")
        print_summary_a(diag)
        plot_stacked_removal(diag); plot_retention_heatmap(diag)
        plot_per_col_boxplots(diag); plot_duration_vs_keeprate(diag)
    else:
        main(max_files=args.max_files, section=args.section)