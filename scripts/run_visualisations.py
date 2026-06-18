"""
run_visualisations.py
─────────────────────────────────────────────────────────────────────────────
Generates all 12 programmatic thesis figures from backbone_comparison JSON
files. Figures are saved to OUTPUT_DIR (configurable, default: Drive results).

Figures produced
────────────────
Results chapter:
  1  thesis_fig_cross_exp_heatmap        5×5 macro-F1, all models × exps
  2  thesis_fig_exp1_performance_v2      Exp1 F1+AUC bars, MLP+SVM pairs
  3  thesis_fig_exp1_confusion           2×2 confusion matrices
  4  thesis_fig_exp1_audio_heatmap_v2    5-model × 13-channel F1 (incl XLS-R)
  5  thesis_fig_audio_slope              Channel rank inversion Exp1→Exp2
  6  thesis_fig_exp5_audio_heatmap       5-model × 13-channel Exp5
  7  thesis_fig_svm_gain_final           ΔF1 heatmap with SVM F1 subscripts
  8  thesis_fig_cross_exp_bars           Grouped bars AUC+F1, 3 strategies × 5 exps

Appendix:
  9  thesis_fig_exp2_audio_heatmap
  10 thesis_fig_exp3_audio_heatmap
  11 thesis_fig9_scratch_vs_finetune     Cross-experiment F1 line chart
  12 thesis_fig11_auc                    AUC profile, 4 strategies

Ablation (read from ablation_results.json):
  13 Thesis_fig_ablation_freeze          Layer freezing ablation
  14 Thesis_fig_ablation_loss            Loss function ablation
  15 Thesis_fig_ablation_decay           LR decay factor ablation
  16 Thesis_fig_ablation_summary         Best-per-condition vs XLS-R baseline

Usage
─────
    python scripts/run_visualisations.py --results_dir /path/to/results
    python scripts/run_visualisations.py --results_dir /path/to/results --figure 2
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Palette ────────────────────────────────────────────────────────────────────
BG            = "#FFFFFF"
PANEL         = "#F7F9FC"
BORDER        = "#BDC3C7"
TEXT          = "#1C2833"
MUTED         = "#6B7280"
ACCENT        = "#1A237E"
SCRATCH_W2V   = "#1565C0"
SCRATCH_WLM   = "#42A5F5"
FINETUNE_W2V  = "#E65100"
FINETUNE_WLM  = "#FF8A65"
XLSR_COLOR    = "#6A1B9A"
XLSR_SCRATCH_COLOR = "#AB47BC"
SVM_COLOR     = "#00695C"
CHANCE_COLOR  = "#9E9E9E"

# ── Ablation palette ───────────────────────────────────────────────────────────
ABL_MLP    = "#2196F3"   # blue  — MLP head bars
ABL_SVM    = "#4CAF50"   # green — SVM probe bars
ABL_CUR_EC = "#1a1a2e"   # dark outline for current config

ABL_SAVE_NAMES = {
    "freeze": "13. Thesis_fig_ablation_freeze",
    "loss":   "14. Thesis_fig_ablation_loss",
    "decay":  "15. Thesis_fig_ablation_decay",
}

ABL_TITLE_MAP = {
    "freeze": "XLS-R Layer Freezing Ablation — Experiment 1",
    "loss":   "XLS-R Loss Function Ablation — Experiment 1",
    "decay":  "XLS-R LR Decay Factor Ablation — Experiment 1",
}

CMAP_DIV = LinearSegmentedColormap.from_list(
    "crs", ["#C62828","#FFECB3","#1B5E20"], N=256)
CMAP_RdGn = LinearSegmentedColormap.from_list(
    "rdgn", ["#B71C1C","#FFECB3","#1B5E20"], N=256)

# ── Model / experiment layout ──────────────────────────────────────────────────
EXP_FOLDERS = {
    "1": "exp1_backbone_comparison",
    "2": "exp2_backbone_comparison",
    "3": "exp3_backbone_comparison",
    "4": "exp4_backbone_comparison",
    "5": "exp5_backbone_comparison",
}

EXP_LABELS = {
    "1": "Exp1\nBinary",
    "2": "Exp2\nSession",
    "3": "Exp3\nTrajectory",
    "4": "Exp4\nPaired",
    "5": "Exp5\nGeneralisation",
}

MODEL_ORDER  = ["wav2vec2_scratch","wav2vec2_finetune","wavlm_scratch",
                "wavlm_finetune","xlsr_scratch","xlsr_finetune"]
MODEL_LABELS = ["w2v2\nScratch","w2v2\nFinetune","WavLM\nScratch",
                "WavLM\nFinetune","XLS-R\nScratch","XLS-R\nFinetune"]
MODEL_COLORS = [SCRATCH_W2V, FINETUNE_W2V, SCRATCH_WLM, FINETUNE_WLM,
                XLSR_SCRATCH_COLOR, XLSR_COLOR]

AUDIO_TYPES  = ["a","e","i","o","u","a1","a2","a3","speech","agua","brasero","dia","mesa"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all(results_dir: Path) -> dict:
    """Load all available backbone_comparison.json files. Returns {exp_key: data}."""
    data = {}
    for k, folder in EXP_FOLDERS.items():
        p = results_dir / folder / "backbone_comparison.json"
        if p.exists():
            data[k] = load_json(p)
    return data


def g(res: dict, key: str):
    """Get metric with _macro fallback."""
    if not res: return None
    v = res.get(key) or res.get(key+"_macro")
    return float(v) if isinstance(v,(int,float)) else None


def sv(res: dict, key: str):
    """Get SVM metric."""
    svm = res.get("svm",{}) if res else {}
    if not svm: return None
    v = svm.get(key) or svm.get(key+"_macro")
    return float(v) if isinstance(v,(int,float)) else None


def pat(res: dict, channel: str, metric: str = "f1_macro"):
    """Get per_audio_type metric for a channel."""
    if not res: return None
    pt  = res.get("test/per_audio_type",{})
    ch  = pt.get(channel,{})
    v   = ch.get(metric)
    return float(v) if isinstance(v,(int,float)) else None


def savefig(fig, out: Path, name: str, dpi: int = 200):
    fig.savefig(str(out/f"{name}.png"), bbox_inches="tight",
                facecolor=BG, dpi=dpi)
    plt.close(fig)
    print(f"  Saved → {name}")


def ax_style(ax):
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, zorder=1)

# ── Ablation helpers ───────────────────────────────────────────────────────────

def abl_is_complete(entry: dict) -> bool:
    return (entry.get("status", "") == "complete"
            and entry.get("test_f1") is not None)


def abl_filter_complete(entries: list) -> list:
    return [e for e in entries if abl_is_complete(e)]


def abl_title(key: str) -> str:
    return ABL_TITLE_MAP.get(key, f"{key.capitalize()} Ablation — Experiment 1")


def abl_save_name(key: str) -> str:
    return ABL_SAVE_NAMES.get(key, f"Thesis_fig_ablation_{key}")


def savefig_abl(fig, out: Path, name: str):
    """PNG-only save at 180 DPI with #f8f9fa background."""
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / f"{name}.png"), bbox_inches="tight",
                facecolor="#f8f9fa", dpi=180)
    plt.close(fig)
    print(f"  Saved → {name}")
    
    
# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Cross-experiment macro-F1 heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_cross_exp_heatmap(all_data: dict, out: Path):
    exp_keys   = ["1","2","3","4","5"]
    exp_labels = [EXP_LABELS[k].replace("\n"," ") for k in exp_keys]

    # Known values from completed runs
    known = {
        "wav2vec2_scratch":  [0.761,0.509,0.389,0.412,0.516],
        "wav2vec2_finetune": [0.567,0.247,0.321,0.412,0.256],
        "wavlm_scratch":     [0.647,0.516,0.293,None, 0.520],
        "wavlm_finetune":    [0.579,0.448,0.342,None, 0.365],
        "xlsr_finetune":     [0.696,0.247,None, None, None ],
    }

    # Overlay with any fresh JSON data
    mat = np.full((len(MODEL_ORDER), len(exp_keys)), np.nan)
    for ri, mk in enumerate(MODEL_ORDER):
        for ci, ek in enumerate(exp_keys):
            res = all_data.get(ek,{}).get(mk,{})
            v   = g(res,"test/f1_macro")
            if v is not None:
                mat[ri,ci] = v
            elif known.get(mk,[None]*5)[ci] is not None:
                mat[ri,ci] = known[mk][ci]

    fig, ax = plt.subplots(figsize=(11,5.5), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(mat, cmap=CMAP_RdGn, vmin=0.20, vmax=0.82,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(exp_keys)))
    ax.set_xticklabels(exp_labels, fontsize=10)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_LABELS, fontsize=10)
    ax.axhline(1.5, color="white", linewidth=2)
    ax.axhline(3.5, color="white", linewidth=2)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri,ci]
            if not np.isnan(v):
                tc = "white" if v<0.35 or v>0.70 else TEXT
                ax.text(ci,ri,f"{v:.3f}",ha="center",va="center",
                        fontsize=10,color=tc,fontweight="bold")
            else:
                ax.text(ci,ri,"—",ha="center",va="center",fontsize=11,color=MUTED)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Test F1 (macro)", fontsize=9, color=TEXT)
    ax.set_title("Cross-Experiment Test F1 — All Models",
                 fontsize=12,fontweight="bold",color=ACCENT,pad=18)
    plt.tight_layout()
    savefig(fig, out, "1. Thesis_fig_cross_exp_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Exp1 performance bar chart (v2 includes XLS-R)
# ══════════════════════════════════════════════════════════════════════════════

def fig_exp1_performance_v2(all_data: dict, out: Path):
    exp1 = all_data.get("1",{})

    f1s  = [g(exp1.get(m,{}), "test/f1_macro")  or 0 for m in MODEL_ORDER]
    aucs = [g(exp1.get(m,{}), "test/roc_auc")   or 0 for m in MODEL_ORDER]
    sf1s = [sv(exp1.get(m,{}), "test/f1_macro")     for m in MODEL_ORDER]
    saucs= [sv(exp1.get(m,{}), "test/roc_auc")      for m in MODEL_ORDER]

    # Known fallbacks
    known_f1  = [0.761,0.567,0.647,0.579,None, 0.696]
    known_auc = [0.831,0.665,0.767,0.661,None, 0.791]
    known_sf1 = [None, 0.562,None, 0.584,None, 0.685]
    known_sau = [None, 0.637,None, 0.653,None, 0.777]

    f1s  = [v if v else known_f1[i]  for i,v in enumerate(f1s)]
    aucs = [v if v else known_auc[i] for i,v in enumerate(aucs)]
    sf1s = [v if v else known_sf1[i] for i,v in enumerate(sf1s)]
    saucs= [v if v else known_sau[i] for i,v in enumerate(saucs)]

    fig, axes = plt.subplots(1,2,figsize=(14,5.5),facecolor=BG)
    fig.suptitle("Experiment 1: Binary Sinusitis Detection — Test Performance",
                 fontsize=13,fontweight="bold",color=ACCENT,y=1.01)
    x = np.arange(len(MODEL_ORDER)); w = 0.32

    for ax, vals, svals, ylabel in [
        (axes[0], f1s,  sf1s,  "Test F1 (macro)"),
        (axes[1], aucs, saucs, "Test AUC"),
    ]:
        ax_style(ax)
        bars = ax.bar(x-w/2, vals, w, color=MODEL_COLORS, alpha=0.88,
                      edgecolor="white", linewidth=0.8, label="MLP head", zorder=3)
        for i,(sv_val,has) in enumerate([(v,v is not None) for v in svals]):
            if has:
                ax.bar(x[i]+w/2, sv_val, w, color=SVM_COLOR, alpha=0.75,
                       edgecolor="white", linewidth=0.8, zorder=3)
                ax.text(x[i]+w/2, sv_val+0.01, f"{sv_val:.3f}",
                        ha="center",va="bottom",fontsize=8,color=SVM_COLOR,fontweight="bold")
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                    ha="center",va="bottom",fontsize=8.5,color=TEXT,fontweight="bold")
        ax.axhline(0.5,color=CHANCE_COLOR,linestyle="--",linewidth=1.2,alpha=0.6,zorder=2)
        ax.text(len(MODEL_ORDER)-0.5,0.515,"chance",color=CHANCE_COLOR,fontsize=8,va="bottom")
        ax.set_ylim(0,1.0); ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS,fontsize=9.5)
        ax.set_ylabel(ylabel,fontsize=11,color=TEXT)
        ax.set_title(ylabel,fontsize=11,color=TEXT,pad=8)

    legend_handles = [
    mpatches.Patch(color=SCRATCH_W2V,       label="wav2vec2 Scratch"),
    mpatches.Patch(color=FINETUNE_W2V,      label="wav2vec2 FT (MLP)"),
    mpatches.Patch(color=SCRATCH_WLM,       label="WavLM Scratch"),
    mpatches.Patch(color=FINETUNE_WLM,      label="WavLM FT (MLP)"),
    mpatches.Patch(color=XLSR_SCRATCH_COLOR,label="XLS-R Scratch"),
    mpatches.Patch(color=XLSR_COLOR,        label="XLS-R FT (MLP)"),
    mpatches.Patch(color=SVM_COLOR,         label="FT (SVM probe)"),
    
]
    fig.legend(handles=legend_handles,loc="lower center",ncol=4,
            fontsize=9,framealpha=0.9,edgecolor=BORDER,bbox_to_anchor=(0.5,-0.10))
    plt.tight_layout()
    savefig(fig, out, "2. Thesis_fig_exp1_performance_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Exp1 confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

def fig_exp1_confusion(all_data: dict, out: Path):
    exp1 = all_data.get("1",{})
    # 5 models in a 2×3 grid (bottom-right cell left empty)
    cms = [
    ("wav2vec2\nScratch",  [[1232,170],[169,304]]),
    ("WavLM\nScratch",     (exp1.get("wavlm_scratch",{}).get("test/confusion_matrix")
                             or [[1323,79],[324,149]])),
    ("XLS-R\nScratch",    exp1.get("xlsr_scratch",{}).get("test/confusion_matrix")),
    ("wav2vec2\nFinetune", exp1.get("wav2vec2_finetune",{}).get("test/confusion_matrix")),
    ("WavLM\nFinetune",   exp1.get("wavlm_finetune",{}).get("test/confusion_matrix")),
    ("XLS-R\nFinetune",   exp1.get("xlsr_finetune",{}).get("test/confusion_matrix")),
    ]
    blues = LinearSegmentedColormap.from_list("b",["#E3F2FD","#1565C0"],N=256)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG)
    fig.suptitle("Experiment 1 — Test Set Confusion Matrices (All Models)",
                 fontsize=13, fontweight="bold", color=ACCENT, y=1.01)
    all_axes = axes.flatten()
    for idx, (lbl, cm) in enumerate(cms):
        ax = all_axes[idx]
        if cm is None or any(v is None for row in cm for v in row):
            ax.set_facecolor(PANEL)
            ax.text(0.5,0.5,"Pending",ha="center",va="center",
                    transform=ax.transAxes,fontsize=11,color=MUTED,style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lbl,fontsize=11,fontweight="bold",color=MUTED,pad=8)
            continue
        cm_arr = np.array(cm); total = cm_arr.sum()
        im = ax.imshow(cm_arr, cmap=blues, interpolation="nearest")
        ax.set_title(lbl, fontsize=11, fontweight="bold", color=TEXT, pad=8)
        ax.set_xlabel("Predicted", fontsize=10, color=TEXT)
        ax.set_ylabel("True", fontsize=10, color=TEXT)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Control","FESS"]); ax.set_yticklabels(["Control","FESS"])
        thresh = cm_arr.max()/2.0
        for i in range(2):
            for j in range(2):
                pct = 100*cm_arr[i,j]/total
                ax.text(j,i,f"{cm_arr[i,j]}\n({pct:.1f}%)",
                        ha="center",va="center",fontsize=11,
                        color="white" if cm_arr[i,j]>thresh else TEXT,
                        fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
   
    plt.tight_layout()
    savefig(fig, out, "3. Thesis_fig_exp1_confusion")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Audio type heatmap helper (used for Exp1,2,3,5)
# ══════════════════════════════════════════════════════════════════════════════

def fig_audio_heatmap(exp_data: dict, exp_name: str, out: Path, fname: str):
    heat = np.full((len(MODEL_ORDER), len(AUDIO_TYPES)), np.nan)
    for ri, mk in enumerate(MODEL_ORDER):
        res = exp_data.get(mk,{})
        for ci, at in enumerate(AUDIO_TYPES):
            v = pat(res, at)
            if v is not None:
                heat[ri, ci] = v

    fig, ax = plt.subplots(figsize=(14,4.5), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(heat, cmap=CMAP_DIV, vmin=0.15, vmax=0.90,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(AUDIO_TYPES)))
    ax.set_xticklabels(AUDIO_TYPES, fontsize=10.5)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_LABELS, fontsize=10)
    # Group separators
    for sep in [4.5, 7.5, 8.5]:
        ax.axvline(sep, color="white", linewidth=2)
    # Group labels placed below x-axis via transforms so they don't overlap cells
    for x_pos, grp_label in [(2,"Vowels"),(6,"Sustained"),(8,"Speech"),(10.5,"TDU Words")]:
        ax.annotate(grp_label,
                    xy=(x_pos, len(MODEL_ORDER)-0.5),
                    xytext=(x_pos, len(MODEL_ORDER)+0.55),
                    ha="center", fontsize=8.5, color=MUTED,
                    fontweight="bold", annotation_clip=False)
    for ri in range(heat.shape[0]):
        for ci in range(heat.shape[1]):
            v = heat[ri,ci]
            if not np.isnan(v):
                tc = "white" if v<0.35 or v>0.75 else TEXT
                ax.text(ci,ri,f"{v:.2f}",ha="center",va="center",
                        fontsize=8.5,color=tc,fontweight="bold")
            else:
                ax.text(ci,ri,"—",ha="center",va="center",fontsize=9,color=MUTED)
    ax.axhline(1.5,color="white",linewidth=2)
    ax.axhline(3.5,color="white",linewidth=2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Test F1 (macro)",fontsize=9,color=TEXT)
    ax.set_title(f"{exp_name} — Per-Audio-Type Test F1 by Model",
                 fontsize=12,fontweight="bold",color=ACCENT,pad=28)
    plt.tight_layout()
    savefig(fig, out, fname)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Audio channel slope chart (rank inversion Exp1→Exp2)
# ══════════════════════════════════════════════════════════════════════════════

def fig_audio_slope(all_data: dict, out: Path):
    """
    Bump chart: rank of each audio channel group across Exp1 and Exp2.
    Rank 1 = most informative. Crossing lines signal rank inversions.
    """
    groups    = ["Vowels","Sustained","Speech","TDU Words"]
    exp1_vals = [0.704, 0.562, 0.769, 0.832]   # F1 Exp1 wav2vec2-scratch
    exp2_vals = [0.588, 0.621, 0.481, 0.583]   # F1 Exp2 wav2vec2-scratch
    gcolors   = ["#1976D2","#388E3C","#F57C00","#7B1FA2"]

    import scipy.stats as _ss
    exp1_ranks = _ss.rankdata([-v for v in exp1_vals]).astype(int)  # rank 1 = highest F1
    exp2_ranks = _ss.rankdata([-v for v in exp2_vals]).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for grp, r1, r2, v1, v2, col in zip(groups, exp1_ranks, exp2_ranks,
                                          exp1_vals, exp2_vals, gcolors):
        ax.plot([0, 1], [r1, r2], color=col, linewidth=2.8, alpha=0.9,
                marker="o", markersize=11, zorder=3, solid_capstyle="round")
        # Left labels: rank + F1 value
        ax.text(-0.06, r1, f"#{r1}  {grp}\nF1={v1:.3f}",
                ha="right", va="center", fontsize=9.5, color=col, fontweight="bold")
        # Right labels: rank + F1 value
        ax.text(1.06, r2, f"#{r2}  {grp}\nF1={v2:.3f}",
                ha="left", va="center", fontsize=9.5, color=col, fontweight="bold")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(4.6, 0.4)   # inverted so rank 1 is at the top
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Experiment 1\n(CRS vs Control)", "Experiment 2\n(Pre vs Post-op)"],
        fontsize=11, fontweight="bold", color=TEXT
    )
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1st\n(most informative)", "2nd", "3rd",
                         "4th\n(least informative)"],
                        fontsize=8.5, color=MUTED)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.set_title(
        "Audio Channel Rank by Diagnostic Value\n"
        "Crossing lines indicate a rank inversion between experiments",
        fontsize=12, fontweight="bold", color=ACCENT, pad=14
    )
    ax.text(0.5, 4.45,
            "wav2vec2-scratch  ·  rank 1 = highest Test F1",
            ha="center", fontsize=8, color=MUTED, style="italic",
            transform=ax.transData)
    plt.tight_layout()
    savefig(fig, out, "5. Thesis_fig_audio_slope")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — SVM gain heatmap (ΔF1 = SVM − MLP, with SVM F1 subscript)
# ══════════════════════════════════════════════════════════════════════════════

def fig_svm_gain_final(all_data: dict, out: Path):
    ft_models  = ["wav2vec2_finetune","wavlm_finetune","xlsr_finetune"]
    ft_labels  = ["wav2vec2\nFinetune","WavLM\nFinetune","XLS-R\nFinetune"]
    exp_keys   = ["1","2","3","4","5"]
    exp_labels = [EXP_LABELS[k].replace("\n"," ") for k in exp_keys]

    # Known gains and SVM F1 values
    known_gain = {
        "wav2vec2_finetune": [-0.005, 0.234, 0.031, 0.084, 0.238],
        "wavlm_finetune":    [ 0.005, 0.082, 0.029, None,  0.172],
        "xlsr_finetune":     [-0.011, 0.323, None,  None,  None ],
    }
    known_svm = {
        "wav2vec2_finetune": [0.562, 0.481, 0.352, 0.496, 0.494],
        "wavlm_finetune":    [0.584, 0.530, 0.371, None,  0.537],
        "xlsr_finetune":     [0.685, 0.570, None,  None,  None ],
    }

    gain_matrix = np.full((len(ft_models), len(exp_keys)), np.nan)
    svm_matrix  = np.full((len(ft_models), len(exp_keys)), np.nan)

    for ri, mk in enumerate(ft_models):
        for ci, ek in enumerate(exp_keys):
            res    = all_data.get(ek,{}).get(mk,{})
            mlp_f1 = g(res,"test/f1_macro")
            svm_f1 = sv(res,"test/f1_macro")
            if mlp_f1 is not None and svm_f1 is not None:
                gain_matrix[ri,ci] = svm_f1 - mlp_f1
                svm_matrix[ri,ci]  = svm_f1
            elif known_gain[mk][ci] is not None:
                gain_matrix[ri,ci] = known_gain[mk][ci]
                svm_matrix[ri,ci]  = known_svm[mk][ci]

    vmax = 0.35
    fig, ax = plt.subplots(figsize=(11,5), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(gain_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(exp_keys)))
    ax.set_xticklabels(exp_labels, fontsize=10)
    ax.set_yticks(np.arange(len(ft_models)))
    ax.set_yticklabels(ft_labels, fontsize=10)
    ax.set_xlabel("Experiment",fontsize=11,color=TEXT)
    ax.set_ylabel("Finetune Model",fontsize=11,color=TEXT)
    ax.set_title("SVM Linear Probe Gain over MLP Head  (ΔF1 = SVM F1 − MLP F1)",
                 fontsize=12,fontweight="bold",color=ACCENT,pad=12)

    for ri in range(len(ft_models)):
        for ci in range(len(exp_keys)):
            v = gain_matrix[ri,ci]
            if np.isnan(v):
                ax.text(ci,ri,"n/a",ha="center",va="center",fontsize=9,color=MUTED)
            else:
                sign = "+" if v>=0 else ""
                bg_light = abs(v)<0.08
                svm_val  = svm_matrix[ri,ci]
                label    = f"{sign}{v:.3f}"
                if not np.isnan(svm_val):
                    label += f"\n(SVM={svm_val:.3f})"
                ax.text(ci,ri,label,ha="center",va="center",fontsize=8.5,
                        fontweight="bold",color=TEXT if bg_light else "white")

    cbar = plt.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
    cbar.set_label("ΔF1 (SVM − MLP)",fontsize=9,color=TEXT)
    plt.tight_layout()
    savefig(fig, out, "7. Thesis_fig_svm_gain_final")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Cross-experiment grouped horizontal bars (3 strategies × 5 exps)
# ══════════════════════════════════════════════════════════════════════════════

def fig_cross_exp_bars(all_data: dict, out: Path):
    exp_keys = ["1","2","3","4","5"]

    strategies = [
    ("w2v2 Scratch",   "wav2vec2_scratch", "get", SCRATCH_W2V),
    ("WavLM Scratch",  "wavlm_scratch",    "get", SCRATCH_WLM),
    ("XLS-R Scratch",  "xlsr_scratch",     "get", XLSR_SCRATCH_COLOR),
    ("WavLM FT-SVM",   "wavlm_finetune",   "svm", SVM_COLOR),

    ]

    known_f1 = {
        "wav2vec2_scratch":  [0.761,0.509,0.389,0.412,0.516],
        "wavlm_scratch":     [0.647,0.516,0.293,None, 0.520],
        "wavlm_finetune_svm":[0.584,0.530,0.371,None, 0.537],
    }
    known_auc = {
        "wav2vec2_scratch":  [0.831,0.517,0.535,0.557,0.525],
        "wavlm_scratch":     [0.767,0.523,0.536,None, 0.541],
        "wavlm_finetune_svm":[0.653,0.555,0.541,None, 0.559],
    }

    fig, axes = plt.subplots(1,2,figsize=(16,7),facecolor=BG)
    fig.suptitle("Cross-Experiment Performance: Key Strategies",
                 fontsize=13,fontweight="bold",color=ACCENT,y=1.01)

    for ax, metric_name, ylabel in [(axes[0],"F1","Test F1 (macro)"),
                                     (axes[1],"AUC","Test AUC")]:
        ax_style(ax)
        exp_labels_short = ["E1\nBinary","E2\nSession","E3\nTraj",
                             "E4\nPaired","E5\nGen"]
        n_s = len(strategies)
        x   = np.arange(len(exp_keys))
        w   = 0.20

        for si, (label, mk, mode, color) in enumerate(strategies):
            vals = []
            kd   = "wavlm_finetune_svm" if mode=="svm" else mk
            kmap = known_f1 if metric_name=="F1" else known_auc
            for ci, ek in enumerate(exp_keys):
                res = all_data.get(ek,{}).get(mk,{})
                if mode == "svm":
                    v = sv(res, "test/f1_macro" if metric_name=="F1" else "test/roc_auc")
                else:
                    v = g(res, "test/f1_macro" if metric_name=="F1" else "test/roc_auc")
                if v is None and kmap.get(kd,[None]*5)[ci] is not None:
                    v = kmap[kd][ci]
                vals.append(v)

            xs_plot = [x[i]+si*w for i,v in enumerate(vals) if v is not None]
            ys_plot = [vals[i]   for i,v in enumerate(vals) if v is not None]
            ax.bar(xs_plot, ys_plot, w, color=color, alpha=0.85,
                   label=label, edgecolor="white", linewidth=0.8, zorder=3)
            for xi, yi in zip(xs_plot, ys_plot):
                ax.text(xi, yi+0.01, f"{yi:.3f}", ha="center",
                        fontsize=7.5, color=color, fontweight="bold", va="bottom")

        ax.axhline(0.5,color=CHANCE_COLOR,linestyle="--",linewidth=1.2,alpha=0.6)
        ax.set_xticks(x + w * (n_s - 1) / 2); ax.set_xticklabels(exp_labels_short,fontsize=10)
        ax.set_ylabel(ylabel,fontsize=11,color=TEXT)
        ax.set_ylim(0,1.0); ax.set_title(ylabel,fontsize=11,color=TEXT,pad=8)
        ax.legend(fontsize=9,framealpha=0.9,edgecolor=BORDER)

    plt.tight_layout()
    savefig(fig, out, "8. Thesis_fig_cross_exp_bars")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Cross-experiment F1 line chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_scratch_vs_finetune(all_data: dict, out: Path):
    """
    Small-multiples: one panel per backbone family.
    Each panel shows scratch, finetune-MLP and finetune-SVM F1 across 5 exps.
    Cleaner than a single 8-line chart.
    """
    exp_keys   = ["1","2","3","4","5"]
    exp_labels = [EXP_LABELS[k].replace("\n"," ") for k in exp_keys]
    x          = np.arange(len(exp_keys))

    families = [
        {
            "title":  "wav2vec2",
            "series": [
                ("Scratch",   "wav2vec2_scratch",  g,  SCRATCH_W2V, "-",  "o"),
                ("FT (MLP)",  "wav2vec2_finetune", g,  FINETUNE_W2V,"--", "^"),
                ("FT (SVM)",  "wav2vec2_finetune", sv, SVM_COLOR,   ":",  "P"),
            ],
        },
        {
            "title":  "WavLM",
            "series": [
                ("Scratch",  "wavlm_scratch",  g,  SCRATCH_WLM,  "-",  "s"),
                ("FT (MLP)", "wavlm_finetune", g,  FINETUNE_WLM, "--", "v"),
                ("FT (SVM)", "wavlm_finetune", sv, SVM_COLOR,    ":",  "P"),
            ],
        },
        {
            "title":  "XLS-R",
            "series": [
                ("Scratch",  "xlsr_scratch",  g,  XLSR_SCRATCH_COLOR, "-",  "o"),
                ("FT (MLP)", "xlsr_finetune", g,  XLSR_COLOR,         "--", "D"),
                ("FT (SVM)", "xlsr_finetune", sv, SVM_COLOR,          ":",  "P"),
            ],
        },
    ]

    known = {
        ("wav2vec2_scratch",  g): [0.761,0.509,0.389,0.412,0.516],
        ("wav2vec2_finetune", g): [0.567,0.247,0.321,0.412,0.256],
        ("wav2vec2_finetune",sv): [0.562,0.481,0.352,0.496,0.494],
        ("wavlm_scratch",     g): [0.647,0.516,0.293,None, 0.520],
        ("wavlm_finetune",    g): [0.579,0.448,0.342,None, 0.365],
        ("wavlm_finetune",   sv): [0.584,0.530,0.371,None, 0.537],
        ("xlsr_scratch",      g): [None, None, None, None, None ],
        ("xlsr_finetune",     g): [0.696,0.247,None, None, None ],
        ("xlsr_finetune",    sv): [0.685,0.570,None, None, None ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG, sharey=True)
    fig.suptitle("Cross-Experiment Test F1 by Backbone Family",
                 fontsize=13, fontweight="bold", color=ACCENT, y=1.02)

    for ax, fam in zip(axes, families):
        ax.set_facecolor(PANEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for label, mk, mfn, color, ls, marker in fam["series"]:
            kb   = known.get((mk, mfn), [None]*5)
            vals = []
            for ci, ek in enumerate(exp_keys):
                res = all_data.get(ek,{}).get(mk,{})
                v   = mfn(res, "test/f1_macro")
                if v is None: v = kb[ci]
                vals.append(v)
            has = [i for i,v in enumerate(vals) if v is not None]
            if not has: continue
            xs  = [x[i] for i in has]
            ys  = [vals[i] for i in has]
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.2,
                    marker=marker, markersize=7, label=label, alpha=0.9)
            for xi, yi in zip(xs, ys):
                ax.annotate(f"{yi:.3f}", xy=(xi,yi), xytext=(0,8),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=color, fontweight="bold")

        ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--",
                   linewidth=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=8.5, rotation=15, ha="right")
        ax.set_ylim(0.15, 0.95)
        ax.set_title(fam["title"], fontsize=11, fontweight="bold",
                     color=TEXT, pad=8)
        ax.set_xlabel("Experiment", fontsize=9, color=TEXT)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper right",
                  framealpha=0.9, edgecolor=BORDER)

    axes[0].set_ylabel("Test F1 (macro)", fontsize=11, color=TEXT)
    plt.tight_layout()
    savefig(fig, out, "11. Thesis_fig_scratch_vs_finetune")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 12 — AUC profile
# ══════════════════════════════════════════════════════════════════════════════

def fig_radar_auc(all_data: dict, out: Path):
    """
    Grouped dot plot: strategies on y-axis, AUC on x-axis, one dot per
    experiment per strategy. Cleaner than radar — precise values, missing
    data simply absent, easy cross-strategy comparison within each experiment.
    """
    exp_keys    = ["1","2","3","4","5"]
    exp_labels  = [EXP_LABELS[k].replace("\n"," ") for k in exp_keys]

    strategies = [
        ("wav2vec2 Scratch", "wav2vec2_scratch", g,  SCRATCH_W2V),
        ("WavLM Scratch",    "wavlm_scratch",    g,  SCRATCH_WLM),
        ("XLS-R Scratch",    "xlsr_scratch",     g,  XLSR_SCRATCH_COLOR),
        ("XLS-R FT (MLP)",   "xlsr_finetune",    g,  XLSR_COLOR),
        ("WavLM FT (SVM)",   "wavlm_finetune",   sv, SVM_COLOR),
    ]
    known_auc = {
        ("wav2vec2_scratch", g):  [0.831,0.517,0.535,0.557,0.525],
        ("wavlm_scratch",    g):  [0.767,0.523,0.536,None, 0.541],
        ("xlsr_scratch",     g):  [None, None, None, None, None ],
        ("xlsr_finetune",    g):  [0.791,0.566,None, None, None ],
        ("wavlm_finetune",   sv): [0.653,0.555,0.541,None, 0.559],
    }

    n_strat = len(strategies)
    n_exp   = len(exp_keys)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_positions = np.arange(n_strat)
    offsets     = np.linspace(-0.3, 0.3, n_exp)  # vertical jitter per exp

    for si, (label, mk, mfn, color) in enumerate(strategies):
        kb = known_auc.get((mk, mfn), [None]*5)
        for ci, (ek, el) in enumerate(zip(exp_keys, exp_labels)):
            res = all_data.get(ek,{}).get(mk,{})
            v   = mfn(res,"test/roc_auc")
            if v is None: v = kb[ci]
            if v is None: continue
            y_pos = y_positions[si] + offsets[ci]
            sc = ax.scatter(v, y_pos, color=color, s=90, zorder=4,
                            alpha=0.9, edgecolors="white", linewidths=0.6)
            ax.text(v + 0.006, y_pos, f"{v:.3f}",
                    va="center", fontsize=7.5, color=color, fontweight="bold")

        # Horizontal range bar
        vals = [known_auc.get((mk,mfn),[None]*5)[ci] or 0
                for ci in range(n_exp)
                if (known_auc.get((mk,mfn),[None]*5)[ci]) is not None]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y_positions[si], y_positions[si]],
                    color=color, linewidth=1, alpha=0.3, zorder=2)

    # Experiment legend
    exp_handles = [
        plt.scatter([], [], s=80, color="#555555",
                    alpha=0.6+ci*0.08, label=el, edgecolors="white", linewidths=0.6)
        for ci, el in enumerate(exp_labels)
    ]

    ax.axvline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=1.2,
               alpha=0.6, label="Chance (AUC=0.5)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in strategies], fontsize=10, color=TEXT)
    ax.set_xlabel("Test AUC", fontsize=11, color=TEXT)
    ax.set_xlim(0.44, 0.96)
    ax.set_title("AUC Profile by Transfer Learning Strategy and Experiment",
                 fontsize=12, fontweight="bold", color=ACCENT, pad=14)
    ax.grid(True, axis="x", alpha=0.25)

    # Experiment dot legend
    for ci, el in enumerate(exp_labels):
        ax.scatter([], [], s=80, color=MUTED,
                   alpha=0.5+ci*0.1, label=el,
                   edgecolors="white", linewidths=0.6)
    ax.axvline(0.5, color=CHANCE_COLOR, linestyle="--",
               linewidth=1.2, alpha=0.6)

    # Strategy colour legend
    strategy_handles = [
        mpatches.Patch(color=s[3], label=s[0]) for s in strategies
    ]
    ax.legend(handles=strategy_handles, fontsize=8.5,
              loc="lower right", framealpha=0.9, edgecolor=BORDER)

    plt.tight_layout()
    savefig(fig, out, "12. Thesis_fig_auc_profile")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 13/14/15 — per-condition ablation bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation_condition(key: str, entries: list, out: Path) -> bool:
    """
    Horizontal two-panel bar chart for one ablation condition.
    Left panel: Test F1 (macro).  Right panel: Test AUC.
    Upper bar = SVM probe (green), lower bar = MLP head (blue).
    Returns True if saved, False if skipped.
    """
    complete   = abl_filter_complete(entries)
    n_total    = len(entries)
    n_complete = len(complete)

    if n_complete == 0:
        print(f"Ablation [{key}]: {n_total} entries, "
              f"0 complete → skipped (no complete entries)")
        return False

    labels   = [e["label"]        for e in complete]
    mlp_f1s  = [e["test_f1"]      for e in complete]
    mlp_aucs = [e.get("test_auc") for e in complete]
    svm_f1s  = [e.get("svm_f1")   for e in complete]
    svm_aucs = [e.get("svm_auc")  for e in complete]

    current_idx = next(
        (i for i, e in enumerate(complete)
         if "current" in e["label"].lower()), None)

    all_vals = [v for v in mlp_f1s + mlp_aucs + svm_f1s + svm_aucs
                if v is not None]
    if not all_vals:
        print(f"Ablation [{key}]: no plottable values → skipped")
        return False
    x_min = round(min(all_vals) - 0.05, 2)
    x_max = round(max(all_vals) + 0.08, 2)

    fig, (ax_f1, ax_auc) = plt.subplots(
        1, 2, figsize=(13, 1.8 + 0.75 * n_complete),
        sharey=True, facecolor="#f8f9fa")
    fig.suptitle(abl_title(key), fontsize=12, fontweight="bold",
                 color=ACCENT, y=1.01)

    bar_h = 0.30
    y_pos = np.arange(n_complete)
    y_svm = y_pos - 0.18   # upper (higher after invert)
    y_mlp = y_pos + 0.18   # lower

    for ax, mlp_vals, svm_vals, xlabel in [
        (ax_f1,  mlp_f1s,  svm_f1s,  "Test F1 (macro)"),
        (ax_auc, mlp_aucs, svm_aucs, "Test AUC"),
    ]:
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25, zorder=1)

        for i, (mval, sval) in enumerate(zip(mlp_vals, svm_vals)):
            is_cur = (i == current_idx)
            ec = ABL_CUR_EC if is_cur else "white"
            lw = 2.0        if is_cur else 0.6
            if mval is not None:
                ax.barh(y_mlp[i], mval, bar_h, color=ABL_MLP, alpha=0.85,
                        edgecolor=ec, linewidth=lw, zorder=3)
                ax.text(mval + 0.003, y_mlp[i], f"{mval:.3f}",
                        va="center", ha="left", fontsize=8,
                        fontweight="bold", color=ABL_MLP)
            if sval is not None:
                ax.barh(y_svm[i], sval, bar_h, color=ABL_SVM, alpha=0.85,
                        edgecolor=ec, linewidth=lw, zorder=3)
                ax.text(sval + 0.003, y_svm[i], f"{sval:.3f}",
                        va="center", ha="left", fontsize=8,
                        fontweight="bold", color=ABL_SVM)

        if x_min <= 0.500 <= x_max:
            ax.axvline(0.500, color=CHANCE_COLOR, linestyle="--",
                    linewidth=1.0, alpha=0.7, zorder=2)
            ax.text(0.500, 1.01, "chance",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=CHANCE_COLOR)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(xlabel, fontsize=10, color=TEXT)

    ax_f1.set_yticks(y_pos)
    ax_f1.set_yticklabels(labels, fontsize=9)
    ax_f1.invert_yaxis()
    ax_auc.tick_params(labelleft=False)

    bottom_offset = -0.10 if n_complete <= 6 else -0.05
    fig.legend(handles=[
        mpatches.Patch(color=ABL_MLP, label="MLP head"),
        mpatches.Patch(color=ABL_SVM, label="SVM probe"),
        mpatches.Patch(facecolor="white", edgecolor=ABL_CUR_EC,
                       linewidth=2.0, label="Current config"),
    ], loc="lower center", ncol=3, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, bottom_offset))

    plt.tight_layout()
    name = abl_save_name(key)
    savefig_abl(fig, out, name)

    fig_num = name.split(".")[0].strip()
    print(f"Ablation [{key}]: {n_total} entries, "
          f"{n_complete} complete → saved figure {fig_num}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Figure 16 — ablation summary comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation_summary(ablation_data: dict, out: Path):
    """
    Single horizontal bar chart: best result per condition vs XLS-R baseline.
    Only generated when at least two conditions have complete data.
    """
    completed = {k: abl_filter_complete(v) for k, v in ablation_data.items()
                 if abl_filter_complete(v)}
    n_done = len(completed)

    if n_done < 2:
        print(f"Ablation summary: skipped ({n_done} condition(s) complete, need ≥2)")
        return

    # XLS-R baseline from [CURRENT] entry in freeze condition
    baseline_mlp_auc = baseline_svm_auc = None
    baseline_label = "XLS-R baseline"
    if "freeze" in completed:
        for e in completed["freeze"]:
            if "current" in e["label"].lower():
                baseline_mlp_auc = e.get("test_auc")
                baseline_svm_auc = e.get("svm_auc")
                clean = (e["label"].replace("[CURRENT]", "")
                                   .replace("(CURRENT)", "").strip())
                baseline_label = f"XLS-R baseline\n({clean})"
                break

    COND_FMT = {
        "freeze": "Freeze ablation\n(best: {best})",
        "loss":   "Loss ablation\n(best: {best})",
        "decay":  "Decay ablation\n(best: \u03bb={best})",
    }
    rows = []
    for key, entries in completed.items():
        best = max(entries, key=lambda e: e.get("test_auc") or 0.0)
        best_clean = (best["label"].replace("[CURRENT]", "")
                                   .replace("(CURRENT)", "").strip())
        fmt = COND_FMT.get(key, "{key_cap} ablation\n(best: {best})")
        rows.append({
            "label":   fmt.format(best=best_clean, key_cap=key.capitalize()),
            "mlp_auc": best.get("test_auc"),
            "svm_auc": best.get("svm_auc"),
        })

    all_rows = [{"label": baseline_label,
                 "mlp_auc": baseline_mlp_auc,
                 "svm_auc": baseline_svm_auc}] + rows
    n_rows   = len(all_rows)

    fig, ax = plt.subplots(figsize=(11, 1.8 + 0.75 * n_rows), facecolor="#f8f9fa")
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, zorder=1)

    bar_h = 0.30
    y_pos = np.arange(n_rows)
    y_svm = y_pos - 0.18
    y_mlp = y_pos + 0.18

    for i, row in enumerate(all_rows):
        mval, sval = row["mlp_auc"], row["svm_auc"]
        ec = ABL_CUR_EC if i == 0 else "white"
        lw = 2.0        if i == 0 else 0.6
        if mval is not None:
            ax.barh(y_mlp[i], mval, bar_h, color=ABL_MLP, alpha=0.85,
                    edgecolor=ec, linewidth=lw, zorder=3)
            ax.text(mval + 0.003, y_mlp[i], f"{mval:.3f}",
                    va="center", ha="left", fontsize=8,
                    fontweight="bold", color=ABL_MLP)
        if sval is not None:
            ax.barh(y_svm[i], sval, bar_h, color=ABL_SVM, alpha=0.85,
                    edgecolor=ec, linewidth=lw, zorder=3)
            ax.text(sval + 0.003, y_svm[i], f"{sval:.3f}",
                    va="center", ha="left", fontsize=8,
                    fontweight="bold", color=ABL_SVM)

    if baseline_mlp_auc is not None:
        ax.axvline(baseline_mlp_auc, color=ACCENT, linestyle="--",
                   linewidth=1.2, alpha=0.75, zorder=4)
        ax.text(baseline_mlp_auc + 0.003, 1.01, "XLS-R baseline (freeze=4)",
                transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=7.5, color=ACCENT)
  
    all_vals = [r[k] for r in all_rows for k in ("mlp_auc", "svm_auc")
                if r[k] is not None]
    
    if all_vals:
        x_min = round(min(all_vals) - 0.05, 2)
        x_max = round(max(all_vals) + 0.08, 2)
        ax.set_xlim(x_min, x_max)
    else:
        x_min, x_max = 0.0, 1.0

    if x_min <= 0.500 <= x_max:
        ax.axvline(0.500, color=CHANCE_COLOR, linestyle=":",
                linewidth=1.0, alpha=0.6, zorder=2)
        ax.text(0.500, 1.01, "chance", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=CHANCE_COLOR)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in all_rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Test AUC", fontsize=10, color=TEXT)
    ax.set_title("Ablation Summary — Best Result per Condition vs XLS-R Baseline",
                 fontsize=12, fontweight="bold", color=ACCENT, pad=14)
    fig.legend(handles=[
        mpatches.Patch(color=ABL_MLP, label="MLP head (best)"),
        mpatches.Patch(color=ABL_SVM, label="SVM probe (best)"),
        mpatches.Patch(facecolor="white", edgecolor=ABL_CUR_EC,
                       linewidth=2.0, label="XLS-R baseline"),
    ], loc="lower center", ncol=3, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    savefig_abl(fig, out, "16. Thesis_fig_ablation_summary")
    print(f"Ablation summary: generated ({n_done} conditions complete)")
    
# ══════════════════════════════════════════════════════════════════════════════
# Synthesis helpers — shared by Figures 19 and 20
# ══════════════════════════════════════════════════════════════════════════════

# Fallback known values [E1, E2, E3, E4, E5]
_KNOWN_VALS = {
    ("wav2vec2_scratch",  "f1"):      [0.761, 0.509, 0.389, 0.412, 0.516],
    ("wav2vec2_scratch",  "auc"):     [0.831, 0.517, 0.535, 0.557, 0.525],
    ("wavlm_scratch",     "f1"):      [0.647, 0.516, 0.293, 0.504, 0.520],
    ("wavlm_scratch",     "auc"):     [0.767, 0.523, 0.536, 0.555, 0.541],
    ("wav2vec2_finetune", "f1"):      [0.567, 0.247, 0.321, 0.412, 0.256],
    ("wav2vec2_finetune", "auc"):     [0.665, 0.505, 0.489, 0.525, 0.507],
    ("wav2vec2_finetune", "svm_f1"):  [0.562, 0.481, 0.352, 0.496, 0.494],
    ("wav2vec2_finetune", "svm_auc"): [0.637, 0.499, 0.507, 0.537, 0.502],
    ("wavlm_finetune",    "f1"):      [0.579, 0.448, 0.342, 0.502, 0.365],
    ("wavlm_finetune",    "auc"):     [0.661, 0.574, 0.508, 0.591, 0.528],
    ("wavlm_finetune",    "svm_f1"):  [0.584, 0.530, 0.371, 0.530, 0.537],
    ("wavlm_finetune",    "svm_auc"): [0.653, 0.556, 0.541, 0.577, 0.559],
    ("xlsr_finetune",     "f1"):      [0.696, 0.247, 0.432, 0.412, 0.395],
    ("xlsr_finetune",     "auc"):     [0.791, 0.566, 0.634, 0.536, 0.556],
    ("xlsr_finetune",     "svm_f1"):  [0.685, 0.570, 0.414, 0.552, 0.517],
    ("xlsr_finetune",     "svm_auc"): [0.777, 0.660, 0.615, 0.591, 0.547],
}

_EXP_KEYS_S    = ["1", "2", "3", "4", "5"]
_EXP_XLABELS_S = ["E1\nBinary", "E2\nSession", "E3\nTrajectory",
                  "E4\nPaired", "E5\nGeneralisation"]
_SCRATCH_MODELS  = ["wav2vec2_scratch", "wavlm_scratch", "xlsr_scratch"]
_FT_MLP_MODELS   = ["wav2vec2_finetune", "wavlm_finetune", "xlsr_finetune"]

_MODEL_SHORT = {
    ("wav2vec2_scratch",  False): "w2v2-S",
    ("wavlm_scratch",     False): "WavLM-S",
    ("xlsr_scratch",      False): "XLS-R-S",
    ("wav2vec2_finetune", False): "w2v2-FT",
    ("wavlm_finetune",    False): "WavLM-FT",
    ("xlsr_finetune",     False): "XLS-R-FT",
    ("wav2vec2_finetune", True):  "w2v2-FT-SVM",
    ("wavlm_finetune",    True):  "WavLM-FT-SVM",
    ("xlsr_finetune",     True):  "XLS-R-FT-SVM",
}

# Cell background colours: dark backbone colours for MLP, lighter for SVM probe
_CELL_COLORS = {
    ("wav2vec2_scratch",  False): "#1565C0",
    ("wavlm_scratch",     False): "#004D40",
    ("xlsr_scratch",      False): "#AB47BC", 
    ("wav2vec2_finetune", False): "#1565C0",
    ("wavlm_finetune",    False): "#004D40",
    ("xlsr_finetune",     False): "#4A148C",
    ("wav2vec2_finetune", True):  "#90CAF9",
    ("wavlm_finetune",    True):  "#80CBC4",
    ("xlsr_finetune",     True):  "#CE93D8",
}

_DARK_CELL = {k: v for k, v in [
    (("wav2vec2_scratch",  False), True),
    (("wavlm_scratch",     False), True),
    (("xlsr_scratch",      False), True),
    (("wav2vec2_finetune", False), True),
    (("wavlm_finetune",    False), True),
    (("xlsr_finetune",     False), True),
    (("wav2vec2_finetune", True),  False),
    (("wavlm_finetune",    True),  False),
    (("xlsr_finetune",     True),  False),
]}


def _synth_val(all_data, exp, model, metric):
    """Metric lookup with _KNOWN_VALS fallback. metric: f1|auc|svm_f1|svm_auc."""
    res = all_data.get(exp, {}).get(model, {})
    v = (g(res, "test/f1_macro")  if metric == "f1"      else
         g(res, "test/roc_auc")   if metric == "auc"     else
         sv(res, "test/f1_macro") if metric == "svm_f1"  else
         sv(res, "test/roc_auc")  if metric == "svm_auc" else None)
    if v is not None:
        return v
    row = _KNOWN_VALS.get((model, metric), [None] * 5)
    return row[_EXP_KEYS_S.index(exp)]


def _synthesis_table(all_data):
    """
    Returns (winners, strat_f1, strat_auc).

    winners[exp][category] = {"f1", "auc", "model", "is_svm"}
      category ∈ {overall, scratch, ft_mlp, svm}

    strat_f1 / strat_auc: {strategy: [v_e1, v_e2, v_e3, v_e4, v_e5]}
    """
    def _best_mlp(exp, models):
        best = {"f1": None, "auc": None, "model": None, "is_svm": False}
        for m in models:
            v = _synth_val(all_data, exp, m, "f1")
            if v is not None and (best["f1"] is None or v > best["f1"]):
                best = {"f1": v, "auc": _synth_val(all_data, exp, m, "auc"),
                        "model": m, "is_svm": False}
        return best

    def _best_svm(exp):
        best = {"f1": None, "auc": None, "model": None, "is_svm": True}
        for m in _FT_MLP_MODELS:
            v = _synth_val(all_data, exp, m, "svm_f1")
            if v is not None and (best["f1"] is None or v > best["f1"]):
                best = {"f1": v, "auc": _synth_val(all_data, exp, m, "svm_auc"),
                        "model": m, "is_svm": True}
        return best

    winners   = {}
    strat_f1  = {"scratch": [], "ft_mlp": [], "svm": []}
    strat_auc = {"scratch": [], "ft_mlp": [], "svm": []}

    for exp in _EXP_KEYS_S:
        sc  = _best_mlp(exp, _SCRATCH_MODELS)
        ft  = _best_mlp(exp, _FT_MLP_MODELS)
        sv_ = _best_svm(exp)
        ov  = max((c for c in [sc, ft, sv_] if c["f1"] is not None),
                  key=lambda c: c["f1"],
                  default={"f1": None, "auc": None, "model": None, "is_svm": False})
        winners[exp] = {"overall": ov, "scratch": sc, "ft_mlp": ft, "svm": sv_}
        strat_f1["scratch"].append(sc["f1"])
        strat_f1["ft_mlp"].append(ft["f1"])
        strat_f1["svm"].append(sv_["f1"])
        strat_auc["scratch"].append(sc["auc"])
        strat_auc["ft_mlp"].append(ft["auc"])
        strat_auc["svm"].append(sv_["auc"])

    return winners, strat_f1, strat_auc


# ══════════════════════════════════════════════════════════════════════════════
# Figure 19 — Cross-experiment strategy lines
# ══════════════════════════════════════════════════════════════════════════════

def fig_synthesis_strategy_lines(all_data: dict, out: Path):
    """
    Two-panel line chart: best scratch / best FT-MLP / best SVM probe
    tracked across all five experiments.
    """
    _, strat_f1, strat_auc = _synthesis_table(all_data)

    fig, (ax_f1, ax_auc) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Cross-Experiment Performance by Strategy",
                 fontsize=13, fontweight="bold", color=ACCENT, y=1.01)

    x = np.arange(5)
    lines = [
        ("scratch", "Best scratch",   SCRATCH_W2V,  "-",  "o"),
        ("ft_mlp",  "Best FT-MLP",    FINETUNE_W2V, "--", "o"),
        ("svm",     "Best SVM probe", SVM_COLOR,    ":",  "o"),
    ]

    panels = [
        (ax_f1,  strat_f1,  "Test F1 (macro)",
         [(0.500, "chance (binary)"), (0.333, "chance (3-class)")]),
        (ax_auc, strat_auc, "Test AUC",
         [(0.500, "chance")]),
    ]

    for ax, vals_dict, ylabel, chance_lines in panels:
        ax.set_facecolor(PANEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.20, zorder=1)

        for level, label in chance_lines:
            ax.axhline(level, color=CHANCE_COLOR, linestyle="--",
                       linewidth=1.0, alpha=0.7, zorder=2)
            ax.text(4.55, level, label, va="center", ha="left",
                    fontsize=7.5, color=CHANCE_COLOR)

        for key, label, color, ls, marker in lines:
            ys = vals_dict[key]
            xi = [i for i, v in enumerate(ys) if v is not None]
            yi = [v for v in ys if v is not None]
            ax.plot([x[i] for i in xi], yi,
                    color=color, linestyle=ls, linewidth=2.0,
                    marker=marker, markersize=7, zorder=4, label=label)
            for pos, (i, v) in enumerate(zip(xi, yi)):
                off = 0.018 if pos % 2 == 0 else -0.025
                ax.text(x[i], v + off, f"{v:.3f}",
                        ha="center",
                        va="bottom" if off > 0 else "top",
                        fontsize=7.5, color=color,
                        fontweight="bold", zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(_EXP_XLABELS_S, fontsize=9)
        ax.set_xlabel("Experiment", fontsize=10, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=10, color=TEXT)
        ax.set_xlim(-0.4, 5.2)
        all_v = [v for ys in vals_dict.values() for v in ys if v is not None]
        if all_v:
            ax.set_ylim(round(min(all_v) - 0.06, 2),
                        round(max(all_v) + 0.08, 2))

    ax_f1.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(str(out / "19. Thesis_fig_synthesis_strategy_lines.png"),
                bbox_inches="tight", facecolor=BG, dpi=200)
    plt.close(fig)
    print("  Saved → 19. Thesis_fig_synthesis_strategy_lines")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 20 — Cross-experiment winner map
# ══════════════════════════════════════════════════════════════════════════════

def fig_synthesis_winner_map(all_data: dict, out: Path):
    """
    4×5 winner grid. Cell background encodes F1 performance (YlGn scale).
    Thin left strip encodes backbone (3 colours). Text shows model name,
    F1, AUC. Separates performance and identity into distinct visual channels.
    """
    winners, _, _ = _synthesis_table(all_data)

    ROWS = [
        ("overall", "Best overall"),
        ("scratch", "Best scratch"),
        ("ft_mlp",  "Best FT-MLP"),
        ("svm",     "Best SVM probe"),
    ]

    BACKBONE_STRIP = {
        "wav2vec2_scratch":  "#1565C0",
        "wavlm_scratch":     "#00695C",
        "wav2vec2_finetune": "#1565C0",
        "wavlm_finetune":    "#00695C",
        "xlsr_finetune":     "#6A1B9A",
    }

    n_rows, n_cols = len(ROWS), len(_EXP_KEYS_S)

    # Normalise F1 across all cells for the performance colormap
    all_f1 = [winners[exp][cat]["f1"]
              for exp in _EXP_KEYS_S for cat, _ in ROWS
              if winners[exp][cat]["f1"] is not None]
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0.25, vmax=max(all_f1) + 0.02 if all_f1 else 0.80)
    cmap = plt.cm.YlGn

    fig, ax = plt.subplots(figsize=(15, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -1.0)
    ax.axis("off")

    for ri, (cat_key, _) in enumerate(ROWS):
        for ci, exp in enumerate(_EXP_KEYS_S):
            w     = winners[exp][cat_key]
            model = w["model"]
            f1_v  = w["f1"]
            auc_v = w["auc"]

            # Cell background — performance intensity
            rgba = cmap(norm(f1_v)) if f1_v is not None else (0.94, 0.94, 0.94, 1)
            brightness = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            tc = "white" if brightness < 0.52 else TEXT

            ax.add_patch(mpatches.Rectangle(
                (ci - 0.48, ri - 0.48), 0.96, 0.96,
                facecolor=rgba, edgecolor="white", linewidth=1.8, zorder=2))

            # Backbone strip — thin left edge (3 colours only)
            strip_c = BACKBONE_STRIP.get(model, MUTED) if model else MUTED
            ax.add_patch(mpatches.Rectangle(
                (ci - 0.48, ri - 0.48), 0.07, 0.96,
                facecolor=strip_c, edgecolor="none", zorder=3))

            # Text: winner name → F1 → AUC, shifted right of strip
            short   = _MODEL_SHORT.get((model, w["is_svm"]), "—") if model else "—"
            f1_str  = f"F1  {f1_v:.3f}"  if f1_v  is not None else "F1  —"
            auc_str = f"AUC {auc_v:.3f}" if auc_v is not None else "AUC —"
            tx = ci + 0.06
            ax.text(tx, ri - 0.16, short,   ha="center", va="center",
                    fontsize=9, fontweight="bold", color=tc, zorder=4)
            ax.text(tx, ri + 0.10, f1_str,  ha="center", va="center",
                    fontsize=8,   color=tc, zorder=4)
            ax.text(tx, ri + 0.30, auc_str, ha="center", va="center",
                    fontsize=7.5, color=tc, zorder=4)

    # Column headers — inside axes so they cannot overlap the title
    for ci, lbl in enumerate(_EXP_XLABELS_S):
        ax.text(ci, -0.75, lbl, ha="center", va="center",
                fontsize=9, fontweight="bold", color=TEXT)

    # Row labels
    for ri, (_, cat_label) in enumerate(ROWS):
        ax.text(-0.55, ri, cat_label, ha="right", va="center",
                fontsize=9, color=TEXT)

    # Backbone legend (left-strip colours, 3 items only)
    bb_patches = [
        mpatches.Patch(color="#1565C0", label="wav2vec2"),
        mpatches.Patch(color="#00695C", label="WavLM"),
        mpatches.Patch(color="#6A1B9A", label="XLS-R"),
    ]
    ax.legend(handles=bb_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=9, framealpha=0.9,
              title="Backbone (left strip)", title_fontsize=8)

    # Colorbar for performance scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.015, pad=0.02, aspect=20)
    cbar.set_label("Test F1 (macro)", fontsize=9, color=TEXT)
    cbar.ax.tick_params(labelsize=8)

    # Use suptitle so it never competes with column headers inside the axes
    fig.suptitle("Cross-Experiment Winner Map",
                 fontsize=13, fontweight="bold", color=ACCENT, y=1.07)
    plt.tight_layout()
    fig.savefig(str(out / "20. Thesis_fig_synthesis_winner_map.png"),
                bbox_inches="tight", facecolor=BG, dpi=200)
    plt.close(fig)
    print("  Saved → 20. Thesis_fig_synthesis_winner_map")
    
    
    """
    4-row × 5-column grid showing the winning configuration per
    experiment per category. Cells colour-coded by backbone.
    """
    winners, _, _ = _synthesis_table(all_data)

    ROWS = [
        ("overall", "Best overall"),
        ("scratch", "Best scratch"),
        ("ft_mlp",  "Best FT-MLP"),
        ("svm",     "Best SVM probe"),
    ]
    n_rows, n_cols = len(ROWS), len(_EXP_KEYS_S)

    fig, ax = plt.subplots(
        figsize=(14, max(4.5, 1.4 * n_rows)), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -1.0)
    ax.axis("off")

    for ri, (cat_key, _) in enumerate(ROWS):
        for ci, exp in enumerate(_EXP_KEYS_S):
            w      = winners[exp][cat_key]
            model  = w["model"]
            is_svm = w["is_svm"]
            f1_v   = w["f1"]
            auc_v  = w["auc"]

            cell_c = _CELL_COLORS.get((model, is_svm), PANEL) if model else PANEL
            dark   = _DARK_CELL.get((model, is_svm), False)   if model else False
            tc     = "white" if dark else TEXT

            ax.add_patch(mpatches.Rectangle(
                (ci - 0.48, ri - 0.48), 0.96, 0.96,
                facecolor=cell_c, edgecolor="white", linewidth=2.0, zorder=2))

            short   = _MODEL_SHORT.get((model, is_svm), "—") if model else "—"
            f1_str  = f"F1  {f1_v:.3f}"  if f1_v  is not None else "F1  —"
            auc_str = f"AUC {auc_v:.3f}" if auc_v is not None else "AUC —"

            ax.text(ci, ri - 0.18, short,   ha="center", va="center",
                    fontsize=9,   fontweight="bold", color=tc, zorder=3)
            ax.text(ci, ri + 0.08, f1_str,  ha="center", va="center",
                    fontsize=7.5, color=tc, zorder=3)
            ax.text(ci, ri + 0.28, auc_str, ha="center", va="center",
                    fontsize=7.5, color=tc, zorder=3)

    # Column headers
    for ci, lbl in enumerate(_EXP_XLABELS_S):
        ax.text(ci, -0.62, lbl, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TEXT)

    # Row labels
    for ri, (_, cat_label) in enumerate(ROWS):
        ax.text(-0.55, ri, cat_label, ha="right", va="center",
                fontsize=9, color=TEXT)

    legend_patches = [
        mpatches.Patch(color="#1565C0", label="wav2vec2 (MLP)"),
        mpatches.Patch(color="#004D40", label="WavLM (MLP)"),
        mpatches.Patch(color="#4A148C", label="XLS-R (MLP)"),
        mpatches.Patch(color="#AB47BC", label="XLS-R Scratch"),
        mpatches.Patch(color="#90CAF9", label="wav2vec2 (SVM probe)"),
        mpatches.Patch(color="#80CBC4", label="WavLM (SVM probe)"),
        mpatches.Patch(color="#CE93D8", label="XLS-R (SVM probe)"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9,
              framealpha=0.9, title="Backbone / Strategy", title_fontsize=9)

    ax.set_title("Cross-Experiment Winner Map",
                 fontsize=13, fontweight="bold", color=ACCENT, pad=18)
    plt.tight_layout()
    fig.savefig(str(out / "20. Thesis_fig_synthesis_winner_map.png"),
                bbox_inches="tight", facecolor=BG, dpi=200)
    plt.close(fig)
    print("  Saved → 20. Thesis_fig_synthesis_winner_map")
    
# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

FIGURE_MAP = {
    1:  ("Cross-experiment heatmap",       fig_cross_exp_heatmap),
    2:  ("Exp1 performance bars v2",       fig_exp1_performance_v2),
    3:  ("Exp1 confusion matrices",        fig_exp1_confusion),
    4:  ("Exp1 audio heatmap v2",          None),   # handled specially
    5:  ("Audio slope chart",             fig_audio_slope),
    6:  ("Exp5 audio heatmap",            None),
    7:  ("SVM gain heatmap",              fig_svm_gain_final),
    8:  ("Cross-exp grouped bars",        fig_cross_exp_bars),
    9:  ("Exp2 audio heatmap (appendix)", None),
    10: ("Exp3 audio heatmap (appendix)", None),
    11: ("Scratch vs FT line chart",      fig_scratch_vs_finetune),
    12: ("Radar AUC profile",             fig_radar_auc),
    13: ("Ablation: freeze depth",        None),   # handled via ablation_data
    14: ("Ablation: loss function",       None),
    15: ("Ablation: LR decay",            None),
    16: ("Ablation: summary comparison",  None),
    19: ("Synthesis strategy lines",      fig_synthesis_strategy_lines),
    20: ("Synthesis winner map",          fig_synthesis_winner_map),
}


def run_all(all_data: dict, out: Path, figure: int = 0,
            ablation_data: dict = None):
    def should_run(n): return figure == 0 or figure == n

    if should_run(1):
        print("\n── Figure 1: Cross-exp heatmap ─────────────────────────────")
        fig_cross_exp_heatmap(all_data, out)

    if should_run(2):
        print("\n── Figure 2: Exp1 performance bars v2 ──────────────────────")
        fig_exp1_performance_v2(all_data, out)

    if should_run(3):
        print("\n── Figure 3: Exp1 confusion matrices ───────────────────────")
        fig_exp1_confusion(all_data, out)

    if should_run(4):
        print("\n── Figure 4: Exp1 audio heatmap (v2 incl XLS-R) ───────────")
        fig_audio_heatmap(all_data.get("1",{}), "Experiment 1",
                          out, "4. Thesis_fig_exp1_audio_heatmap")

    if should_run(5):
        print("\n── Figure 5: Audio slope chart ──────────────────────────────")
        fig_audio_slope(all_data, out)

    if should_run(6):
        print("\n── Figure 6: Exp5 audio heatmap ────────────────────────────")
        fig_audio_heatmap(all_data.get("5",{}), "Experiment 5",
                          out, "6. Thesis_fig_exp5_audio_heatmap")

    if should_run(7):
        print("\n── Figure 7: SVM gain heatmap ───────────────────────────────")
        fig_svm_gain_final(all_data, out)

    if should_run(8):
        print("\n── Figure 8: Cross-exp grouped bars ────────────────────────")
        fig_cross_exp_bars(all_data, out)

    if should_run(9):
        print("\n── Figure 9 (App): Exp2 audio heatmap ──────────────────────")
        fig_audio_heatmap(all_data.get("2",{}), "Experiment 2",
                          out, "9. Thesis_fig_exp2_audio_heatmap")

    if should_run(10):
        print("\n── Figure 10 (App): Exp3 audio heatmap ─────────────────────")
        fig_audio_heatmap(all_data.get("3",{}), "Experiment 3",
                          out, "10. Thesis_fig_exp3_audio_heatmap")

    if should_run(11):
        print("\n── Figure 11 (App): Scratch vs FT line chart ───────────────")
        fig_scratch_vs_finetune(all_data, out)

    if should_run(12):
        print("\n── Figure 12 (App): Radar AUC profile ──────────────────────")
        fig_radar_auc(all_data, out)

    if should_run(19):
        print("\n── Figure 19: Synthesis strategy lines ─────────────────────")
        fig_synthesis_strategy_lines(all_data, out)

    if should_run(20):
        print("\n── Figure 20: Synthesis winner map ─────────────────────────")
        fig_synthesis_winner_map(all_data, out)

    # ── Ablation figures (13–16) — require ablation_results.json ─────────────
    if ablation_data is None:
        if any(should_run(n) for n in [13, 14, 15, 16]):
            print("\n[ablation] No ablation_results.json loaded — skipping figures 13–16.")
    else:
        ABL_CONDITION_MAP = {"freeze": 13, "loss": 14, "decay": 15}

        if figure == 0:
            # Run all conditions present in the JSON
            print("\n── Figures 13–15: Ablation condition charts ────────────────")
            for key, entries in ablation_data.items():
                fig_ablation_condition(key, entries, out)
        else:
            # Run only the specifically requested condition
            for cond, fig_num in ABL_CONDITION_MAP.items():
                if should_run(fig_num) and cond in ablation_data:
                    print(f"\n── Figure {fig_num}: Ablation [{cond}] ──────────────────────")
                    fig_ablation_condition(cond, ablation_data[cond], out)

        if should_run(16):
            print("\n── Figure 16: Ablation summary ─────────────────────────────")
            fig_ablation_summary(ablation_data, out)

    print(f"\nAll figures saved to:\n  {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="/content/drive/MyDrive/MSc_Sinusitis_results",
        help="Path to results directory containing expN_backbone_comparison folders"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save figures (defaults to results_dir)"
    )
    parser.add_argument(
        "--figure", type=int, default=0,
        help="Generate a specific figure only (1-16). 0 = all."
    )
    args   = parser.parse_args()
    res_d  = Path(args.results_dir)
    out_d  = Path(args.output_dir) if args.output_dir else res_d
    out_d.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {res_d}")
    all_data = load_all(res_d)
    print(f"Loaded {len(all_data)} experiment JSON files: {list(all_data.keys())}")

    # Load ablation JSON if present (optional — figures 1–12 run without it)
    # NEW — checks results_dir first, then results_dir/ablation/
    abl_path = res_d / "ablation_results.json"
    if not abl_path.exists():
        abl_path = res_d / "ablation" / "ablation_results.json"

    if abl_path.exists():
        with open(abl_path) as f:
            ablation_data = json.load(f)
        print(f"Loaded ablation JSON ({abl_path}): {list(ablation_data.keys())}")
    else:
        ablation_data = None
        print("No ablation_results.json found — figures 13–16 will be skipped.")

    run_all(all_data, out_d, figure=args.figure, ablation_data=ablation_data)