"""
run_exploratory_DA.py
─────────────────────────────────────────────────────────────────────────────
Exploratory Data Analysis for the sinusitis clinical audio dataset.

Produces publication-quality figures covering:
  1. Duration distribution per group and audio type  (existing)
  2. Missing file counts per group and column heatmap  (existing)
  3. VAD keep-rate per group  (existing)
  4. Patient and session count per group  (NEW — for methods section)
  5. Segment count per audio channel after preprocessing  (NEW)
  6. Class balance visualisation per experiment  (NEW)

All figures saved to:
    results/Plots and visuals/eda/

Usage
─────
    python scripts/run_exploratory_DA.py
    python scripts/run_exploratory_DA.py --max_files 300
    python scripts/run_exploratory_DA.py --from_cache
    python scripts/run_exploratory_DA.py --counts_only   # plots 4-6 only (fast, no audio scan)
"""

import argparse
import sys
import warnings
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
from src.config import TARGET_SR
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
    "figure.facecolor":  BG,   "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID, "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT, "xtick.color":       TEXT,
    "ytick.color":       TEXT, "grid.color":        GRID,
    "grid.linewidth":    0.5,  "text.color":        TEXT,
    "font.family":       "monospace",
    "legend.facecolor":  PANEL_BG, "legend.edgecolor": GRID,
    "boxplot.medianprops.color": ACCENT,
})

OUTPUT_DIR = PROJECT_ROOT / "results" / "Plots and visuals" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_MAP = {"FESS": "Fess", "Contr": "Contr", "Sept": "Sept", "Tonsill": "Tonsill"}
GROUPS    = ["Sept", "Fess", "Contr", "Tonsill"]

CHANNEL_GROUPS = {
    "Vowels":    ["a","e","i","o","u"],
    "Sustained": ["a1","a2","a3"],
    "Speech":    ["speech"],
    "TDU Words": ["agua","brasero","dia","mesa"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data collection  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def collect_metadata(df: pd.DataFrame, max_files: int = 9999) -> pd.DataFrame:
    audio_cols = [c for c in COLUMN_TO_SUBFOLDER.keys() if c in df.columns]
    records    = []
    scanned    = 0
    print(f"\nScanning up to {max_files} files for EDA metadata...")

    for _, row in df.iterrows():
        raw_group = str(row.get("GROUP","")).strip()
        group     = GROUP_MAP.get(raw_group, raw_group)
        subj_id   = row.get("ID","?")
        session   = row.get("session","?")

        for col in audio_cols:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                records.append({"group":group,"id":subj_id,"session":session,
                                 "col":col,"exists":False,"raw_dur":np.nan,
                                 "vad_dur":np.nan,"vad_keep_pct":np.nan})
                continue
            try:
                path = resolve_path(str(val), PROJECT_ROOT, col=col)
            except Exception:
                records.append({"group":group,"id":subj_id,"session":session,
                                 "col":col,"exists":False,"raw_dur":np.nan,
                                 "vad_dur":np.nan,"vad_keep_pct":np.nan})
                continue
            if not path.exists():
                records.append({"group":group,"id":subj_id,"session":session,
                                 "col":col,"exists":False,"raw_dur":np.nan,
                                 "vad_dur":np.nan,"vad_keep_pct":np.nan})
                continue
            try:
                info     = torchaudio.info(str(path))
                raw_dur  = info.num_frames / info.sample_rate
                wav, sr  = torchaudio.load(str(path))
                wav      = wav.mean(dim=0, keepdim=True)
                wav      = remove_leading_silence(wav, sr)
                if sr != TARGET_SR:
                    wav  = T.Resample(sr, TARGET_SR)(wav); sr = TARGET_SR
                wav      = highpass_filter(wav, sr)
                vad_wav  = voice_activity_detection(wav, sr)
                vad_dur  = vad_wav.shape[1] / sr
                keep_pct = 100.0 * vad_dur / max(raw_dur, 1e-6)
                records.append({"group":group,"id":subj_id,"session":session,
                                 "col":col,"exists":True,"raw_dur":raw_dur,
                                 "vad_dur":vad_dur,"vad_keep_pct":keep_pct})
                scanned += 1
                if scanned % 50 == 0:
                    print(f"  ...{scanned} files scanned")
                if scanned >= max_files:
                    print(f"  Reached max_files={max_files}, stopping.")
                    return pd.DataFrame(records)
            except Exception:
                records.append({"group":group,"id":subj_id,"session":session,
                                 "col":col,"exists":True,"raw_dur":np.nan,
                                 "vad_dur":np.nan,"vad_keep_pct":np.nan})

    print(f"  Done. {scanned} files scanned.")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 1 — Duration distribution  (existing, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_duration_distribution(meta: pd.DataFrame):
    exist      = meta[meta["exists"] & meta["raw_dur"].notna()]
    if exist.empty:
        print("  [SKIP] No duration data."); return
    audio_cols = sorted(exist["col"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(18,12), facecolor=BG)
    fig.suptitle("Raw Duration Distribution by Group & Audio Task",
                 fontsize=13, color=TEXT, y=1.01)
    for ax, group in zip(axes.flatten(), GROUPS):
        gdf   = exist[exist["group"]==group]
        color = PALETTE.get(group,"#90CAF9")
        if gdf.empty:
            ax.set_title(f"{group} — no data", color=TEXT); continue
        cols_p   = [c for c in audio_cols if c in gdf["col"].unique()]
        data_col = [gdf[gdf["col"]==c]["raw_dur"].dropna().values for c in cols_p]
        bp = ax.boxplot(data_col, patch_artist=True, notch=False, vert=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color(ACCENT); med.set_linewidth(1.8)
        for i,(cd,cn) in enumerate(zip(data_col,cols_p),1):
            jitter = np.random.uniform(-0.18,0.18,len(cd))
            ax.scatter(np.full_like(cd,i)+jitter,cd,alpha=0.45,s=12,color=color,zorder=3)
        ax.set_title(f"{group}  (n={len(gdf)} files)",fontsize=10,color=color,pad=5)
        ax.set_xticks(range(1,len(cols_p)+1))
        ax.set_xticklabels(cols_p,rotation=45,ha="right",fontsize=7)
        ax.set_ylabel("Duration (s)",fontsize=8); ax.grid(True,axis="y",linewidth=0.4)
        ax.set_facecolor(PANEL_BG)
        for i,cd in enumerate(data_col,1):
            if len(cd):
                ax.text(i,np.median(cd)+0.1,f"{np.median(cd):.1f}s",
                        ha="center",va="bottom",fontsize=6,color=ACCENT)
    plt.tight_layout()
    p = OUTPUT_DIR/"eda_duration_by_group_task.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 2 — Missing file heatmap  (existing, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_heatmap(meta: pd.DataFrame):
    audio_cols = sorted(meta["col"].unique())
    matrix     = pd.DataFrame(index=GROUPS, columns=audio_cols, dtype=float)
    pct_matrix = pd.DataFrame(index=GROUPS, columns=audio_cols, dtype=float)
    for g in GROUPS:
        for col in audio_cols:
            subset  = meta[(meta["group"]==g)&(meta["col"]==col)]
            total   = len(subset)
            missing = (subset["exists"]==False).sum()
            matrix.loc[g,col]     = missing
            pct_matrix.loc[g,col] = (100.0*missing/total) if total else 0.0
    matrix     = matrix.fillna(0).astype(float)
    pct_matrix = pct_matrix.fillna(0).astype(float)

    fig, axes = plt.subplots(1,2,figsize=(18,6),facecolor=BG)
    fig.suptitle("Missing Files per Group & Audio Task",fontsize=12,color=TEXT,y=1.02)
    for ax,data,title in zip(axes,[matrix,pct_matrix],
                              ["Count of Missing Files","% Missing"]):
        im = ax.imshow(data.values,cmap="RdYlGn_r",aspect="auto",
                       vmin=0,vmax=data.values.max() or 1)
        ax.set_xticks(range(len(audio_cols)))
        ax.set_xticklabels(audio_cols,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS,fontsize=9)
        ax.set_title(title,fontsize=10,color=TEXT,pad=6); ax.set_facecolor(PANEL_BG)
        fmt = ".1f" if "%" in title else ".0f"
        for r in range(len(GROUPS)):
            for c in range(len(audio_cols)):
                val = data.values[r,c]
                suffix = "%" if "%" in title else ""
                ax.text(c,r,f"{val:{fmt}}{suffix}",ha="center",va="center",
                        fontsize=7,color="white" if val>data.values.max()*0.5 else TEXT)
        cbar = plt.colorbar(im,ax=ax,shrink=0.8)
        cbar.ax.tick_params(labelcolor=TEXT,labelsize=7)
    plt.tight_layout()
    p = OUTPUT_DIR/"eda_missing_files_heatmap.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 3 — VAD keep-rate  (existing, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_vad_keeprate(meta: pd.DataFrame):
    vad_data   = meta[meta["exists"] & meta["vad_keep_pct"].notna()]
    if vad_data.empty:
        print("  [SKIP] No VAD data."); return
    audio_cols = sorted(vad_data["col"].unique())

    fig, axes  = plt.subplots(2,2,figsize=(18,12),facecolor=BG)
    fig.suptitle("VAD Keep-Rate: % of Audio Retained After VAD",
                 fontsize=12,color=TEXT,y=1.01)
    for ax,group in zip(axes.flatten(),GROUPS):
        gdf   = vad_data[vad_data["group"]==group]
        color = PALETTE.get(group,"#90CAF9")
        if gdf.empty:
            ax.set_title(f"{group} — no data",color=TEXT); continue
        cols_p = [c for c in audio_cols if c in gdf["col"].unique()]
        means  = [gdf[gdf["col"]==c]["vad_keep_pct"].mean() for c in cols_p]
        stds   = [gdf[gdf["col"]==c]["vad_keep_pct"].std()  for c in cols_p]
        x      = np.arange(len(cols_p))
        bars   = ax.bar(x,means,color=color,alpha=0.7,yerr=stds,capsize=4,
                        error_kw={"ecolor":TEXT,"linewidth":0.8})
        for i,cn in enumerate(cols_p):
            pts = gdf[gdf["col"]==cn]["vad_keep_pct"].values
            ax.scatter(np.full_like(pts,i)+np.random.uniform(-0.2,0.2,len(pts)),
                       pts,alpha=0.5,s=14,color="white",zorder=4)
        ax.axhline(50,color="#FF5252",linewidth=1,linestyle="--",alpha=0.7)
        ax.set_title(f"{group}  (n={len(gdf)} files)",fontsize=10,color=color,pad=5)
        ax.set_xticks(x); ax.set_xticklabels(cols_p,rotation=45,ha="right",fontsize=7)
        ax.set_ylabel("VAD Keep-Rate (%)",fontsize=8); ax.set_ylim(0,115)
        ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)
        for bar,m,s in zip(bars,means,stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(s or 0)+1.5,
                    f"{m:.0f}%",ha="center",va="bottom",fontsize=7,color=TEXT)
    plt.tight_layout()
    p = OUTPUT_DIR/"eda_vad_keeprate.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 4 — Patient and session counts per group  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def plot_patient_session_counts(df: pd.DataFrame):
    """
    Panel 1: Number of unique patients per group.
    Panel 2: Number of recordings per session per group.
    Panel 3: Recordings per audio channel (all groups combined).
    """
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])
    audio_cols   = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(18,6), facecolor=BG)
    fig.suptitle("Dataset Structure — Patients, Sessions, Recordings",
                 fontsize=13, color=TEXT, y=1.02)

    # Panel 1: patients per group
    ax = axes[0]
    pat_counts = df.groupby("_group")["ID"].nunique().reindex(GROUPS).fillna(0)
    colors     = [PALETTE.get(g,"#90CAF9") for g in GROUPS]
    bars       = ax.bar(GROUPS, pat_counts.values, color=colors, alpha=0.85, edgecolor="none")
    for bar, val in zip(bars, pat_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(int(val)), ha="center", va="bottom", fontsize=11,
                color=TEXT, fontweight="bold")
    ax.set_title("Unique Patients per Group", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel("Patient Count", fontsize=9); ax.grid(True,axis="y",linewidth=0.4)
    ax.set_facecolor(PANEL_BG)

    # Panel 2: recordings per session per group
    ax = axes[1]
    session_counts = df.groupby(["_group","session"]).size().unstack(fill_value=0)
    sessions       = sorted(session_counts.columns)
    x = np.arange(len(GROUPS)); w = 0.25
    ses_colors = ["#4FC3F7","#81C784","#FFB74D"]
    for si, (ses, sc) in enumerate(zip(sessions, ses_colors)):
        vals = [session_counts.loc[g,ses] if g in session_counts.index else 0 for g in GROUPS]
        ax.bar(x + si*w, vals, w, label=f"Session {ses}",
               color=sc, alpha=0.85, edgecolor="none")
    ax.set_title("Recordings per Session per Group", fontsize=11, color=TEXT, pad=8)
    ax.set_xticks(x + w); ax.set_xticklabels(GROUPS, fontsize=10)
    ax.set_ylabel("Recording Count", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)

    # Panel 3: recordings per audio channel (non-empty cells)
    ax = axes[2]
    ch_counts = {}
    for col in audio_cols:
        n = df[col].notna().sum() if col in df.columns else 0
        ch_counts[col] = n
    cols_ord  = [c for c in ["a","e","i","o","u","a1","a2","a3",
                              "agua","brasero","dia","mesa","speech"]
                 if c in ch_counts]
    ch_vals   = [ch_counts.get(c,0) for c in cols_ord]
    ch_colors = []
    for c in cols_ord:
        for grp_name, members in CHANNEL_GROUPS.items():
            if c in members:
                ch_colors.append({"Vowels":"#4FC3F7","Sustained":"#81C784",
                                   "Speech":"#FFB74D","TDU Words":"#CE93D8"}.get(grp_name,"#90CAF9"))
                break
    bars = ax.bar(cols_ord, ch_vals, color=ch_colors, alpha=0.85, edgecolor="none")
    for bar, val in zip(bars, ch_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(val), ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_title("Non-empty Recordings per Audio Channel", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel("Recording Count", fontsize=9)
    ax.set_xticklabels(cols_ord, rotation=45, ha="right", fontsize=9)
    ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)
    # Channel group legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c,label=l,alpha=0.85) for l,c in
               [("Vowels","#4FC3F7"),("Sustained","#81C784"),
                ("Speech","#FFB74D"),("TDU Words","#CE93D8")]]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    p = OUTPUT_DIR/"eda_patient_session_counts.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 5 — Segment count per channel after preprocessing  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def plot_segment_counts(segment_dir: Path):
    """
    Load saved .pt segment files, count segments per channel per group.
    Requires segment_dir to contain files named like: {col}_{group}_{id}_*.pt
    Falls back gracefully if segment_dir doesn't exist.
    """
    if not segment_dir.exists():
        print(f"  [SKIP] Segment dir not found: {segment_dir}")
        print("  Run preprocessing first, then rerun with --counts_only")
        return

    counts = {g: {col: 0 for cols in CHANNEL_GROUPS.values() for col in cols}
              for g in GROUPS}

    # Count .pt files per channel and group
    for pt_file in segment_dir.rglob("*.pt"):
        parts = pt_file.stem.split("_")
        if len(parts) < 2: continue
        # Try to infer col and group from filename or parent dir
        parent = pt_file.parent.name
        for g in GROUPS:
            if g.lower() in parent.lower() or g.lower() in pt_file.stem.lower():
                for col in [c for cols in CHANNEL_GROUPS.values() for c in cols]:
                    if col == parent or col in pt_file.stem.split("_"):
                        counts[g][col] += 1
                        break
                break

    # If can't parse, just count total .pt files
    total = sum(sum(v.values()) for v in counts.values())
    if total == 0:
        print("  [INFO] Could not parse segment filenames. Showing total count only.")
        n_total = len(list(segment_dir.rglob("*.pt")))
        print(f"  Total .pt segments found: {n_total}")
        return

    fig, ax = plt.subplots(figsize=(14,6), facecolor=BG)
    fig.suptitle("Segment Count per Audio Channel and Group",
                 fontsize=12, color=TEXT)
    cols_ord = [c for cols in CHANNEL_GROUPS.values() for c in cols
                if any(counts[g][c]>0 for g in GROUPS)]
    x = np.arange(len(cols_ord)); w = 0.2
    for gi,g in enumerate(GROUPS):
        vals = [counts[g].get(c,0) for c in cols_ord]
        ax.bar(x+gi*w, vals, w, label=g, color=PALETTE.get(g,"#90CAF9"),
               alpha=0.85, edgecolor="none")
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(cols_ord,rotation=45,ha="right",fontsize=9)
    ax.set_ylabel("Segment Count",fontsize=9); ax.legend(fontsize=9)
    ax.set_title("Segments per Channel per Group (after 1s windowing)",
                 fontsize=10,color=TEXT,pad=6)
    ax.grid(True,axis="y",linewidth=0.4); ax.set_facecolor(PANEL_BG)
    plt.tight_layout()
    p = OUTPUT_DIR/"eda_segment_counts.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA Plot 6 — Class balance per experiment  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_balance(df: pd.DataFrame):
    """
    Horizontal bar charts showing class balance for each of the 5 experiments.
    Exp1: FESS vs Control (binary)
    Exp2: Pre-op vs Post-op (session 1 vs 2+3 for FESS)
    Exp3: Session 1/2/3 (FESS only)
    Exp4: Paired change (FESS, implicit 30/70 balance)
    Exp5: FESS (train) vs Septoplasty+Tonsillectomy (test)
    """
    df["_group"] = df["GROUP"].map(GROUP_MAP).fillna(df["GROUP"])
    audio_cols   = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]

    def count_recordings(mask):
        sub = df[mask]
        return int(sub[audio_cols].notna().sum().sum())

    exp_data = [
        ("Exp 1: CRS vs Control",
         [("FESS",    count_recordings(df["_group"]=="Fess")),
          ("Control", count_recordings(df["_group"]=="Contr"))],
         ["#E91E63","#4CAF50"]),
        ("Exp 2: Pre-op vs Post-op (FESS)",
         [("Session 1\n(pre-op)",  count_recordings((df["_group"]=="Fess")&(df["session"]==1))),
          ("Session 2+3\n(post)",  count_recordings((df["_group"]=="Fess")&(df["session"]>1)))],
         ["#4FC3F7","#FF9800"]),
        ("Exp 3: 3-class Trajectory (FESS)",
         [("Session 1",count_recordings((df["_group"]=="Fess")&(df["session"]==1))),
          ("Session 2",count_recordings((df["_group"]=="Fess")&(df["session"]==2))),
          ("Session 3",count_recordings((df["_group"]=="Fess")&(df["session"]==3)))],
         ["#4FC3F7","#81C784","#FFB74D"]),
        ("Exp 5: Generalisation",
         [("FESS\n(train)",           count_recordings(df["_group"]=="Fess")),
          ("Septoplasty\n(test)",      count_recordings(df["_group"]=="Sept")),
          ("Tonsillectomy\n(test)",    count_recordings(df["_group"]=="Tonsill"))],
         ["#E91E63","#2196F3","#FF9800"]),
    ]

    fig, axes = plt.subplots(2,2,figsize=(14,10),facecolor=BG)
    fig.suptitle("Class Balance per Experiment",fontsize=13,color=TEXT,y=1.02)
    for ax, (title, class_data, colors) in zip(axes.flatten(), exp_data):
        labels = [c[0] for c in class_data]
        values = [c[1] for c in class_data]
        total  = sum(values) or 1
        y      = np.arange(len(labels))
        bars   = ax.barh(y, values, color=colors[:len(labels)], alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width()+total*0.01, bar.get_y()+bar.get_height()/2,
                    f"{val:,}  ({100*val/total:.1f}%)",
                    va="center", fontsize=9, color=TEXT)
        ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=10)
        ax.set_xlabel("Recording Count",fontsize=9)
        ax.set_title(title,fontsize=10,color=TEXT,pad=6)
        ax.set_xlim(0,max(values)*1.35)
        ax.grid(True,axis="x",linewidth=0.4); ax.set_facecolor(PANEL_BG)
    plt.tight_layout()
    p = OUTPUT_DIR/"eda_class_balance.png"
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(max_files: int = 9999, counts_only: bool = False):
    csv_path = (PROJECT_ROOT / "Data" / "data_final" /
                "Clinical" / "clinical_all_sessions.csv")
    df = pd.read_csv(csv_path)

    # Plots 4-6 don't need audio scanning
    print("\n── EDA Plot 4: Patient and session counts ─────────────────")
    plot_patient_session_counts(df)

    print("\n── EDA Plot 5: Segment counts per channel ─────────────────")
    seg_dir = PROJECT_ROOT / "clean_audio_3s"
    plot_segment_counts(seg_dir)

    print("\n── EDA Plot 6: Class balance per experiment ────────────────")
    plot_class_balance(df)

    if counts_only:
        print(f"\nCounts-only mode: skipping audio scan. All plots saved to:\n  {OUTPUT_DIR}")
        return

    # Plots 1-3 require full audio scan
    cache_path = OUTPUT_DIR/"eda_metadata_cache.csv"
    meta = collect_metadata(df, max_files=max_files)
    meta.to_csv(cache_path, index=False)
    print(f"\nMetadata cached → {cache_path}")

    print("\n── EDA Plot 1: Duration distribution ──────────────────────")
    plot_duration_distribution(meta)

    print("\n── EDA Plot 2: Missing files heatmap ──────────────────────")
    plot_missing_heatmap(meta)

    print("\n── EDA Plot 3: VAD keep-rate ───────────────────────────────")
    plot_vad_keeprate(meta)

    print(f"\nAll EDA plots saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files",   type=int, default=9999)
    parser.add_argument("--from_cache",  action="store_true")
    parser.add_argument("--counts_only", action="store_true",
                        help="Run only plots 4-6 (no audio scan needed)")
    args = parser.parse_args()

    if args.from_cache:
        cache_path = OUTPUT_DIR/"eda_metadata_cache.csv"
        if not cache_path.exists():
            print("No cache found. Run without --from_cache first."); sys.exit(1)
        df = pd.read_csv(
            PROJECT_ROOT/"Data"/"data_final"/"Clinical"/"clinical_all_sessions.csv"
        )
        meta = pd.read_csv(cache_path)
        print(f"Loaded {len(meta)} records from cache.")
        plot_patient_session_counts(df)
        plot_class_balance(df)
        plot_duration_distribution(meta)
        plot_missing_heatmap(meta)
        plot_vad_keeprate(meta)
    else:
        main(max_files=args.max_files, counts_only=args.counts_only)