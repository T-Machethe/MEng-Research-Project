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
  12 thesis_fig11_radar_auc              Radar AUC profile, 4 strategies

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
SVM_COLOR     = "#00695C"
CHANCE_COLOR  = "#9E9E9E"

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
                "wavlm_finetune","xlsr_finetune"]
MODEL_LABELS = ["w2v2\nScratch","w2v2\nFinetune","WavLM\nScratch",
                "WavLM\nFinetune","XLS-R\nFinetune"]
MODEL_COLORS = [SCRATCH_W2V, FINETUNE_W2V, SCRATCH_WLM, FINETUNE_WLM, XLSR_COLOR]

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
    for ext in ["pdf","png"]:
        fig.savefig(str(out/f"{name}.{ext}"), bbox_inches="tight",
                    facecolor=BG, dpi=dpi if ext=="png" else None)
    plt.close(fig)
    print(f"  Saved → {name}")


def ax_style(ax):
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, zorder=1)


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
    ax.set_title("Cross-Experiment Test F1 Summary\n* = partial / pending results",
                 fontsize=12,fontweight="bold",color=ACCENT,pad=12)
    ax.text(-0.58, 0.5,  "Scratch",      va="center",ha="right",fontsize=9,
            color=SCRATCH_W2V, fontweight="bold",rotation=90)
    ax.text(-0.58, 2.5,  "Finetune",     va="center",ha="right",fontsize=9,
            color=FINETUNE_W2V,fontweight="bold",rotation=90)
    ax.text(-0.58, 4,    "Multilingual", va="center",ha="right",fontsize=9,
            color=XLSR_COLOR,  fontweight="bold",rotation=90)
    plt.tight_layout()
    savefig(fig, out, "thesis_fig_cross_exp_heatmap")


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
    known_f1  = [0.761,0.567,0.647,0.579,0.696]
    known_auc = [0.831,0.665,0.767,0.661,0.791]
    known_sf1 = [None, 0.562,None, 0.584,0.685]
    known_sau = [None, 0.637,None, 0.653,0.777]
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
        mpatches.Patch(color=SCRATCH_W2V,  label="wav2vec2 Scratch"),
        mpatches.Patch(color=FINETUNE_W2V, label="wav2vec2 FT (MLP)"),
        mpatches.Patch(color=SCRATCH_WLM,  label="WavLM Scratch"),
        mpatches.Patch(color=FINETUNE_WLM, label="WavLM FT (MLP)"),
        mpatches.Patch(color=XLSR_COLOR,   label="XLS-R FT (MLP)"),
        mpatches.Patch(color=SVM_COLOR,    label="FT (SVM probe)"),
    ]
    fig.legend(handles=legend_handles,loc="lower center",ncol=3,
               fontsize=9,framealpha=0.9,edgecolor=BORDER,bbox_to_anchor=(0.5,-0.10))
    plt.tight_layout()
    savefig(fig, out, "thesis_fig_exp1_performance_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Exp1 confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

def fig_exp1_confusion(all_data: dict, out: Path):
    exp1 = all_data.get("1",{})
    cms  = {
        "wav2vec2\nScratch":  [[1232,170],[169,304]],
        "wav2vec2\nFinetune": (exp1.get("wav2vec2_finetune",{}).get("test/confusion_matrix")
                               or [[None,None],[None,None]]),
        "WavLM\nScratch":     (exp1.get("wavlm_scratch",{}).get("test/confusion_matrix")
                               or [[1323,79],[324,149]]),
        "WavLM\nFinetune":    (exp1.get("wavlm_finetune",{}).get("test/confusion_matrix")
                               or [[None,None],[None,None]]),
    }
    blues = LinearSegmentedColormap.from_list("b",["#E3F2FD","#1565C0"],N=256)
    fig, axes = plt.subplots(2,2,figsize=(10,8),facecolor=BG)
    fig.suptitle("Experiment 1 — Test Set Confusion Matrices",
                 fontsize=13,fontweight="bold",color=ACCENT)
    for ax,(lbl,cm) in zip(axes.flatten(),cms.items()):
        if cm is None or any(v is None for row in cm for v in row):
            ax.axis("off"); continue
        cm_arr = np.array(cm); total = cm_arr.sum()
        im = ax.imshow(cm_arr,cmap=blues,interpolation="nearest")
        ax.set_title(lbl,fontsize=11,fontweight="bold",color=TEXT,pad=8)
        ax.set_xlabel("Predicted",fontsize=10,color=TEXT)
        ax.set_ylabel("True",fontsize=10,color=TEXT)
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
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    plt.tight_layout()
    savefig(fig, out, "thesis_fig_exp1_confusion")


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
    ax.text(2,   -0.9, "Vowels",    ha="center",fontsize=9,color=TEXT,fontweight="bold")
    ax.text(6,   -0.9, "Sustained", ha="center",fontsize=9,color=TEXT,fontweight="bold")
    ax.text(8,   -0.9, "Speech",    ha="center",fontsize=9,color=TEXT,fontweight="bold")
    ax.text(10.5,-0.9, "TDU Words", ha="center",fontsize=9,color=TEXT,fontweight="bold")
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
                 fontsize=12,fontweight="bold",color=ACCENT,pad=14)
    plt.tight_layout()
    savefig(fig, out, fname)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Audio channel slope chart (rank inversion Exp1→Exp2)
# ══════════════════════════════════════════════════════════════════════════════

def fig_audio_slope(all_data: dict, out: Path):
    groups    = ["Vowels","Sustained","Speech","TDU Words"]
    exp1_vals = [0.704, 0.562, 0.769, 0.832]
    exp2_vals = [0.588, 0.621, 0.481, 0.583]
    gcolors   = ["#1976D2","#388E3C","#F57C00","#7B1FA2"]

    fig, ax = plt.subplots(figsize=(8,6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for grp, v1, v2, col in zip(groups, exp1_vals, exp2_vals, gcolors):
        ax.plot([0,1],[v1,v2],color=col,linewidth=2.2,alpha=0.85,
                marker="o",markersize=8,zorder=3)
        ax.text(-0.06, v1, f"{v1:.3f}", ha="right",va="center",
                fontsize=9.5,color=col,fontweight="bold")
        ax.text(1.06,  v2, f"{v2:.3f}", ha="left", va="center",
                fontsize=9.5,color=col,fontweight="bold")
        mid_y = (v1+v2)/2
        ax.text(0.5, mid_y+0.015, grp, ha="center",fontsize=9,color=col,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",facecolor="white",
                          edgecolor=col,alpha=0.85,linewidth=1.2))

    ax.axhline(0.5,color=CHANCE_COLOR,linestyle="--",alpha=0.5,linewidth=1)
    ax.set_xlim(-0.25,1.25); ax.set_ylim(0.35,1.0)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Experiment 1\n(CRS vs Control)",
                         "Experiment 2\n(Pre vs Post-op)"],
                        fontsize=11,fontweight="bold",color=TEXT)
    ax.set_ylabel("Test F1 (macro) — wav2vec2 Scratch",fontsize=10,color=TEXT)
    ax.set_title("Audio Channel Diagnostic Value Shifts\nBetween Experimental Conditions",
                 fontsize=12,fontweight="bold",color=ACCENT,pad=10)
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    savefig(fig, out, "thesis_fig_audio_slope")


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
    savefig(fig, out, "thesis_fig_svm_gain_final")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Cross-experiment grouped horizontal bars (3 strategies × 5 exps)
# ══════════════════════════════════════════════════════════════════════════════

def fig_cross_exp_bars(all_data: dict, out: Path):
    exp_keys = ["1","2","3","4","5"]

    strategies = [
        ("w2v2 Scratch",  "wav2vec2_scratch", "get", SCRATCH_W2V),
        ("WavLM Scratch", "wavlm_scratch",    "get", SCRATCH_WLM),
        ("WavLM FT-SVM",  "wavlm_finetune",   "svm", SVM_COLOR),
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
    fig.suptitle("Cross-Experiment Performance: 3 Key Strategies",
                 fontsize=13,fontweight="bold",color=ACCENT,y=1.01)

    for ax, metric_name, ylabel in [(axes[0],"F1","Test F1 (macro)"),
                                     (axes[1],"AUC","Test AUC")]:
        ax_style(ax)
        exp_labels_short = ["E1\nBinary","E2\nSession","E3\nTraj",
                             "E4\nPaired","E5\nGen"]
        n_s = len(strategies)
        x   = np.arange(len(exp_keys))
        w   = 0.25

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
        ax.set_xticks(x+w); ax.set_xticklabels(exp_labels_short,fontsize=10)
        ax.set_ylabel(ylabel,fontsize=11,color=TEXT)
        ax.set_ylim(0,1.0); ax.set_title(ylabel,fontsize=11,color=TEXT,pad=8)
        ax.legend(fontsize=9,framealpha=0.9,edgecolor=BORDER)

    plt.tight_layout()
    savefig(fig, out, "thesis_fig_cross_exp_bars")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Cross-experiment F1 line chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_scratch_vs_finetune(all_data: dict, out: Path):
    exp_keys   = ["1","2","3","4","5"]
    x          = np.arange(len(exp_keys))
    x_labels   = [EXP_LABELS[k] for k in exp_keys]

    series = [
        ("wav2vec2 Scratch", "wav2vec2_scratch", g,  SCRATCH_W2V, "-",   "o"),
        ("WavLM Scratch",    "wavlm_scratch",    g,  SCRATCH_WLM, "-",   "s"),
        ("wav2vec2 FT",      "wav2vec2_finetune",g,  FINETUNE_W2V,"--",  "^"),
        ("WavLM FT",         "wavlm_finetune",   g,  FINETUNE_WLM,"--",  "v"),
        ("XLS-R FT",         "xlsr_finetune",    g,  XLSR_COLOR,  "-.",  "D"),
        ("WavLM FT-SVM",     "wavlm_finetune",   sv, SVM_COLOR,   ":",   "P"),
    ]
    known = {
        ("wav2vec2_scratch",g):   [0.761,0.509,0.389,0.412,0.516],
        ("wavlm_scratch",g):      [0.647,0.516,0.293,None, 0.520],
        ("wav2vec2_finetune",g):  [0.567,0.247,0.321,0.412,0.256],
        ("wavlm_finetune",g):     [0.579,0.448,0.342,None, 0.365],
        ("xlsr_finetune",g):      [0.696,0.247,None, None, None ],
        ("wavlm_finetune",sv):    [0.584,0.530,0.371,None, 0.537],
    }

    fig, ax = plt.subplots(figsize=(13,6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for label, mk, mfn, color, ls, marker in series:
        vals = []
        kb   = known.get((mk, mfn), [None]*5)
        for ci, ek in enumerate(exp_keys):
            res = all_data.get(ek,{}).get(mk,{})
            v   = mfn(res,"test/f1_macro")
            if v is None: v = kb[ci]
            vals.append(v)
        has = [i for i,v in enumerate(vals) if v is not None]
        if not has: continue
        xs = [x[i] for i in has]; ys = [vals[i] for i in has]
        ax.plot(xs,ys,color=color,linestyle=ls,linewidth=2.2,
                marker=marker,markersize=7,label=label,alpha=0.9)
        for xi,yi in zip(xs,ys):
            ax.annotate(f"{yi:.3f}",xy=(xi,yi),xytext=(0,9),
                        textcoords="offset points",ha="center",
                        fontsize=7.5,color=color,fontweight="bold")

    ax.axhline(0.5,color=CHANCE_COLOR,linestyle="--",linewidth=1,alpha=0.5,label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(x_labels,fontsize=10)
    ax.set_ylim(0.15,0.95)
    ax.set_ylabel("Test F1 (macro)",fontsize=11,color=TEXT)
    ax.set_xlabel("Experiment",fontsize=11,color=TEXT)
    ax.set_title("Cross-Experiment Test F1: Scratch vs Finetune vs SVM Probe",
                 fontsize=13,fontweight="bold",color=ACCENT,pad=14)
    ax.legend(fontsize=8.5,loc="upper right",framealpha=0.9,edgecolor=BORDER,ncol=2)
    ax.grid(True,axis="y",alpha=0.3)
    plt.tight_layout()
    savefig(fig, out, "thesis_fig9_scratch_vs_finetune")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 12 — Radar AUC profile
# ══════════════════════════════════════════════════════════════════════════════

def fig_radar_auc(all_data: dict, out: Path):
    exp_keys  = ["1","2","3","4","5"]
    exp_short = ["Exp1\nBinary","Exp2\nSession","Exp3\nTraj.","Exp4\nPaired","Exp5\nGen."]
    N       = len(exp_keys)
    angles  = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]

    strategies = [
        ("wav2vec2 Scratch","wav2vec2_scratch",g,  SCRATCH_W2V,"-", 0.9),
        ("WavLM Scratch",   "wavlm_scratch",   g,  SCRATCH_WLM,"-", 0.9),
        ("XLS-R FT (MLP)",  "xlsr_finetune",   g,  XLSR_COLOR, "--",0.85),
        ("WavLM FT (SVM)",  "wavlm_finetune",  sv, SVM_COLOR,  "-.",0.85),
    ]
    known_auc = {
        ("wav2vec2_scratch",g):  [0.831,0.517,0.535,0.557,0.525],
        ("wavlm_scratch",g):     [0.767,0.523,0.536,None, 0.541],
        ("xlsr_finetune",g):     [0.791,0.566,None, None, None ],
        ("wavlm_finetune",sv):   [0.653,0.555,0.541,None, 0.559],
    }

    fig, ax = plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True),facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(exp_short,fontsize=9.5,color=TEXT)
    ax.set_ylim(0.4,0.9)
    ax.set_yticks([0.5,0.6,0.7,0.8,0.9])
    ax.set_yticklabels(["0.5","0.6","0.7","0.8","0.9"],fontsize=7.5,color=MUTED)
    ax.grid(color=BORDER,linewidth=0.8,alpha=0.7)

    for label, mk, mfn, color, ls, alpha in strategies:
        kb   = known_auc.get((mk,mfn),[None]*5)
        vals = []
        for ci, ek in enumerate(exp_keys):
            res = all_data.get(ek,{}).get(mk,{})
            v   = mfn(res,"test/roc_auc")
            if v is None: v = kb[ci]
            vals.append(v if v is not None else 0.5)
        vals += vals[:1]
        ax.plot(angles,vals,color=color,linestyle=ls,linewidth=2.2,alpha=alpha,label=label)
        ax.fill(angles,vals,color=color,alpha=0.08)
        for angle,val,ek in zip(angles[:-1],vals[:-1],exp_keys):
            res    = all_data.get(ek,{}).get(mk,{})
            actual = mfn(res,"test/roc_auc")
            ax.scatter([angle],[val],color=color,s=50,
                       marker="o" if actual is not None else "x",zorder=5,alpha=alpha)

    ax.set_title("Multi-Experiment AUC Profile\nby Transfer Learning Strategy",
                 fontsize=12,fontweight="bold",color=ACCENT,pad=20,va="bottom")
    ax.legend(loc="lower left",bbox_to_anchor=(-0.35,-0.12),
              fontsize=9,framealpha=0.9,edgecolor=BORDER)
    plt.tight_layout()
    savefig(fig, out, "thesis_fig11_radar_auc")


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
}


def run_all(all_data: dict, out: Path, figure: int = 0):
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
                          out, "thesis_fig_exp1_audio_heatmap_v2")

    if should_run(5):
        print("\n── Figure 5: Audio slope chart ──────────────────────────────")
        fig_audio_slope(all_data, out)

    if should_run(6):
        print("\n── Figure 6: Exp5 audio heatmap ────────────────────────────")
        fig_audio_heatmap(all_data.get("5",{}), "Experiment 5",
                          out, "thesis_fig_exp5_audio_heatmap")

    if should_run(7):
        print("\n── Figure 7: SVM gain heatmap ───────────────────────────────")
        fig_svm_gain_final(all_data, out)

    if should_run(8):
        print("\n── Figure 8: Cross-exp grouped bars ────────────────────────")
        fig_cross_exp_bars(all_data, out)

    if should_run(9):
        print("\n── Figure 9 (App): Exp2 audio heatmap ──────────────────────")
        fig_audio_heatmap(all_data.get("2",{}), "Experiment 2",
                          out, "thesis_fig_exp2_audio_heatmap")

    if should_run(10):
        print("\n── Figure 10 (App): Exp3 audio heatmap ─────────────────────")
        fig_audio_heatmap(all_data.get("3",{}), "Experiment 3",
                          out, "thesis_fig_exp3_audio_heatmap")

    if should_run(11):
        print("\n── Figure 11 (App): Scratch vs FT line chart ───────────────")
        fig_scratch_vs_finetune(all_data, out)

    if should_run(12):
        print("\n── Figure 12 (App): Radar AUC profile ──────────────────────")
        fig_radar_auc(all_data, out)

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
        help="Generate a specific figure only (1-12). 0 = all."
    )
    args   = parser.parse_args()
    res_d  = Path(args.results_dir)
    out_d  = Path(args.output_dir) if args.output_dir else res_d
    out_d.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {res_d}")
    all_data = load_all(res_d)
    print(f"Loaded {len(all_data)} experiment JSON files: {list(all_data.keys())}")

    run_all(all_data, out_d, figure=args.figure)