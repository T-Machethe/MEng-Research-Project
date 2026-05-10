"""
diagnose_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Diagnostic script that runs each preprocessing step independently and
measures exactly how much audio each step removes.

Reports per-step retention broken down by:
  - Group (Sept, Fess, Contr, Tonsill)
  - Audio type / column (a, e, speech, a1, agua, etc.)

Outputs
───────
  Console table  — per-step retention summary
  CSV            — full per-file breakdown for custom analysis
  4 plots        — saved to Plots and visuals/diagnostics/
    1. Stacked bar: % removed by each step per group
    2. Heatmap: retention rate per group × audio type
    3. Box plots: per-step retention distribution by audio type
    4. Scatter: raw duration vs VAD keep-rate (coloured by group)

Usage
─────
    python scripts/diagnose_pipeline.py
    python scripts/diagnose_pipeline.py --max_files 80   # quick test
    python scripts/diagnose_pipeline.py --from_cache     # replot only
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
from src.config import TARGET_SR, LEADING_TRIM_S, VAD_THRESHOLD
from src.utils.paths import resolve_path, COLUMN_TO_SUBFOLDER

# ── Style ─────────────────────────────────────────────────────────────────────
BG       = "#0B0E14"
PANEL_BG = "#13161F"
TEXT     = "#DDE1EE"
GRID     = "#252836"

PALETTE = {
    "Sept":    "#2196F3",
    "Fess":    "#E91E63",
    "Contr":   "#4CAF50",
    "Tonsill": "#FF9800",
}
STEP_COLORS = {
    "after_trim":      "#4FC3F7",  # light blue
    "after_highpass":  "#81C784",  # light green  (duration unchanged but shown)
    "after_vad":       "#E91E63",  # pink/red
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "grid.color":       GRID,
    "grid.linewidth":   0.5,
    "text.color":       TEXT,
    "font.family":      "monospace",
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID,
})

OUTPUT_DIR = PROJECT_ROOT /"results"/ "Plots and visuals" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_MAP = {
    "FESS": "Fess", "Contr": "Contr",
    "Sept": "Sept", "Tonsill": "Tonsill",
}
GROUPS = ["Sept", "Fess", "Contr", "Tonsill"]


# ─────────────────────────────────────────────────────────────────────────────
# Step-by-step measurement for a single file
# ─────────────────────────────────────────────────────────────────────────────

def measure_steps(filepath: str) -> dict:
    """
    Load one file and measure the duration (seconds) at each pipeline stage.

    Returns
    -------
    dict with keys:
        raw_dur          – original duration
        after_trim_dur   – after leading silence removal
        after_vad_dur    – after VAD  (high-pass doesn't change duration)
        trim_removed_pct – % removed by trim alone
        vad_removed_pct  – % removed by VAD alone (relative to after_trim)
        total_kept_pct   – % of original that survives both steps
        sr               – final sample rate
    """
    waveform, sr = torchaudio.load(filepath)
    waveform = waveform.mean(dim=0, keepdim=True)          # mono [1, T]

    raw_dur = waveform.shape[1] / sr

    # ── Step 1: Leading trim ──────────────────────────────────────────────
    trimmed = remove_leading_silence(waveform, sr,
                                     trim_seconds=LEADING_TRIM_S)
    after_trim_dur = trimmed.shape[1] / sr

    # ── Step 2: Resample ──────────────────────────────────────────────────
    if sr != TARGET_SR:
        trimmed = T.Resample(sr, TARGET_SR)(trimmed)
        sr = TARGET_SR

    # ── Step 3: High-pass filter (duration unchanged) ────────────────────
    filtered = highpass_filter(trimmed, sr)
    after_highpass_dur = filtered.shape[1] / sr   # same as after_trim_dur

    # ── Step 4: VAD ───────────────────────────────────────────────────────
    vad_out = voice_activity_detection(filtered, sr,
                                       threshold=VAD_THRESHOLD)
    after_vad_dur = vad_out.shape[1] / sr

    # ── Compute per-step contribution ────────────────────────────────────
    trim_removed_pct = 100.0 * (raw_dur - after_trim_dur) / max(raw_dur, 1e-9)
    # VAD removal relative to what remained after trim
    vad_removed_pct  = 100.0 * (after_trim_dur - after_vad_dur) / max(after_trim_dur, 1e-9)
    total_kept_pct   = 100.0 * after_vad_dur / max(raw_dur, 1e-9)

    return {
        "raw_dur":           raw_dur,
        "after_trim_dur":    after_trim_dur,
        "after_highpass_dur":after_highpass_dur,
        "after_vad_dur":     after_vad_dur,
        "trim_removed_pct":  trim_removed_pct,
        "vad_removed_pct":   vad_removed_pct,
        "total_kept_pct":    total_kept_pct,
        "sr":                sr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Collect diagnostics across dataset
# ─────────────────────────────────────────────────────────────────────────────

def collect_diagnostics(df: pd.DataFrame,
                         max_files: int = 9999) -> pd.DataFrame:
    audio_cols = [c for c in COLUMN_TO_SUBFOLDER if c in df.columns]
    records    = []
    scanned    = 0

    print(f"\nRunning step-by-step diagnostic on up to {max_files} files...")
    print(f"  Leading trim : {LEADING_TRIM_S}s")
    print(f"  VAD threshold: {VAD_THRESHOLD} RMS\n")

    for _, row in df.iterrows():
        raw_group = str(row.get("GROUP", "")).strip()
        group     = GROUP_MAP.get(raw_group, raw_group)
        subj_id   = row.get("ID", "?")
        session   = row.get("session", "?")

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
                records.append({
                    "group":   group,
                    "id":      subj_id,
                    "session": session,
                    "col":     col,
                    "file":    path.name,
                    **steps,
                })
                scanned += 1
                if scanned % 50 == 0:
                    print(f"  ...{scanned} files processed")
                if scanned >= max_files:
                    print(f"  Reached max_files={max_files}, stopping.")
                    return pd.DataFrame(records)
            except Exception as e:
                print(f"  [SKIP] {path.name}: {e}")
                continue

    print(f"\n  Done. {scanned} files measured.")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(diag: pd.DataFrame):
    """
    Print a clean per-group × per-step retention table to the console.
    This is the fastest way to see which step is causing the most loss.
    """
    print("\n" + "-" * 72)
    print("  STEP-BY-STEP RETENTION SUMMARY")
    print("-" * 72)
    print(f"  {'GROUP':<10} {'COL':<12} {'N':>4}  "
          f"{'RAW(s)':>7}  {'TRIM%':>7}  {'VAD%':>7}  {'KEPT%':>7}")
    print("-" * 72)

    for group in GROUPS:
        gdf = diag[diag["group"] == group]
        if gdf.empty:
            continue
        for col in sorted(gdf["col"].unique()):
            cdf = gdf[gdf["col"] == col]
            print(
                f"  {group:<10} {col:<12} {len(cdf):>4}  "
                f"{cdf['raw_dur'].mean():>7.2f}  "
                f"{cdf['trim_removed_pct'].mean():>6.1f}%  "
                f"{cdf['vad_removed_pct'].mean():>6.1f}%  "
                f"{cdf['total_kept_pct'].mean():>6.1f}%"
            )
        # Group subtotal
        print(
            f"  {'':10} {'[ALL]':<12} {len(gdf):>4}  "
            f"{gdf['raw_dur'].mean():>7.2f}  "
            f"{gdf['trim_removed_pct'].mean():>6.1f}%  "
            f"{gdf['vad_removed_pct'].mean():>6.1f}%  "
            f"{gdf['total_kept_pct'].mean():>6.1f}%"
        )
        print("─" * 72)

    # Grand total
    print(
        f"  {'OVERALL':<23} {len(diag):>4}  "
        f"{diag['raw_dur'].mean():>7.2f}  "
        f"{diag['trim_removed_pct'].mean():>6.1f}%  "
        f"{diag['vad_removed_pct'].mean():>6.1f}%  "
        f"{diag['total_kept_pct'].mean():>6.1f}%"
    )
    print("═" * 72)
    print()

    # Highlight worst offenders
    print("  TOP 10 WORST-RETAINED FILES (lowest total_kept_pct):")
    print("─" * 72)
    worst = diag.nsmallest(10, "total_kept_pct")[
        ["group", "col", "file", "raw_dur",
         "trim_removed_pct", "vad_removed_pct", "total_kept_pct"]
    ]
    for _, r in worst.iterrows():
        print(f"  [{r['group']:<7}] {r['col']:<10}  "
              f"raw={r['raw_dur']:.2f}s  "
              f"trim={r['trim_removed_pct']:.0f}%  "
              f"vad={r['vad_removed_pct']:.0f}%  "
              f"kept={r['total_kept_pct']:.0f}%  "
              f"{r['file']}")
    print("═" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Stacked bar: % removed per step per group
# ─────────────────────────────────────────────────────────────────────────────

def plot_stacked_removal(diag: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    fig.suptitle("Audio Removed at Each Preprocessing Step by Group",
                 fontsize=12, color=TEXT)

    groups_present = [g for g in GROUPS if g in diag["group"].unique()]
    x = np.arange(len(groups_present))
    width = 0.5

    trim_vals = [diag[diag["group"] == g]["trim_removed_pct"].mean()
                 for g in groups_present]
    # VAD % is relative to post-trim; convert to % of original for stacking
    vad_abs_vals = [
        diag[diag["group"] == g]["total_kept_pct"].mean()
        for g in groups_present
    ]
    # What VAD removes as % of original = 100 - trim_removed - total_kept
    vad_removed_abs = [
        100 - t - k for t, k in zip(trim_vals, vad_abs_vals)
    ]
    kept_vals = vad_abs_vals

    bars_trim = ax.bar(x, trim_vals, width,
                       label=f"Removed by Trim ({LEADING_TRIM_S}s leading)",
                       color="#4FC3F7", alpha=0.85)
    bars_vad  = ax.bar(x, vad_removed_abs, width,
                       bottom=trim_vals,
                       label=f"Removed by VAD (threshold={VAD_THRESHOLD})",
                       color="#E91E63", alpha=0.85)
    bars_kept = ax.bar(x, kept_vals, width,
                       bottom=[t + v for t, v in zip(trim_vals, vad_removed_abs)],
                       label="Retained",
                       color="#4CAF50", alpha=0.85)

    # Annotate each segment
    for i, (t, v, k) in enumerate(zip(trim_vals, vad_removed_abs, kept_vals)):
        if t > 2:
            ax.text(i, t / 2, f"{t:.1f}%", ha="center", va="center",
                    fontsize=8, color=BG, fontweight="bold")
        if v > 2:
            ax.text(i, t + v / 2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=8, color=BG, fontweight="bold")
        ax.text(i, t + v + k / 2, f"{k:.1f}%\nkept",
                ha="center", va="center", fontsize=8,
                color=BG, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups_present, fontsize=11)
    ax.set_ylabel("% of Original Audio", fontsize=9)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", linewidth=0.4)
    ax.set_facecolor(PANEL_BG)

    p = OUTPUT_DIR / "diag_stacked_removal_by_group.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Heatmap: total retention % per group × audio type
# ─────────────────────────────────────────────────────────────────────────────

def plot_retention_heatmap(diag: pd.DataFrame):
    audio_cols = sorted(diag["col"].unique())

    # Build matrices
    kept_mat  = pd.DataFrame(index=GROUPS, columns=audio_cols, dtype=float)
    trim_mat  = pd.DataFrame(index=GROUPS, columns=audio_cols, dtype=float)
    vad_mat   = pd.DataFrame(index=GROUPS, columns=audio_cols, dtype=float)

    for g in GROUPS:
        for col in audio_cols:
            sub = diag[(diag["group"] == g) & (diag["col"] == col)]
            if sub.empty:
                continue
            kept_mat.loc[g, col] = sub["total_kept_pct"].mean()
            trim_mat.loc[g, col] = sub["trim_removed_pct"].mean()
            vad_mat.loc[g, col]  = sub["vad_removed_pct"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(22, 5), facecolor=BG)
    fig.suptitle("Retention / Removal Rates: Group × Audio Type",
                 fontsize=12, color=TEXT, y=1.02)

    datasets = [
        (kept_mat.fillna(0).astype(float),  "Total Kept (%)",          "RdYlGn",    0, 100),
        (trim_mat.fillna(0).astype(float),  "Removed by Trim (%)",     "RdYlGn_r",  0, 20),
        (vad_mat.fillna(0).astype(float),   "Removed by VAD (%)",      "RdYlGn_r",  0, 100),
    ]

    for ax, (data, title, cmap, vmin, vmax) in zip(axes, datasets):
        im = ax.imshow(data.values, cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(audio_cols)))
        ax.set_xticklabels(audio_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels(GROUPS, fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, pad=6)
        ax.set_facecolor(PANEL_BG)

        for r in range(len(GROUPS)):
            for c in range(len(audio_cols)):
                val = data.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.0f}%",
                            ha="center", va="center", fontsize=7,
                            color="white" if val < vmax * 0.4 else BG)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelcolor=TEXT, labelsize=7)

    plt.tight_layout()
    p = OUTPUT_DIR / "diag_retention_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Box plots: per-step retention by audio type
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_col_boxplots(diag: pd.DataFrame):
    audio_cols = sorted(diag["col"].unique())
    x = np.arange(len(audio_cols))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG)
    fig.suptitle("Retention Distribution per Audio Type",
                 fontsize=12, color=TEXT)

    for ax, col_name, color, title in [
        (axes[0], "trim_removed_pct", "#4FC3F7",
         f"% Removed by Leading Trim ({LEADING_TRIM_S}s) per Audio Type"),
        (axes[1], "vad_removed_pct",  "#E91E63",
         f"% Removed by VAD (threshold={VAD_THRESHOLD}) per Audio Type"),
    ]:
        data_by_col = [diag[diag["col"] == c][col_name].dropna().values
                       for c in audio_cols]

        bp = ax.boxplot(data_by_col, patch_artist=True,
                        notch=False, vert=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("#FFEB3B")
            median.set_linewidth(2)

        for i, cdata in enumerate(data_by_col, 1):
            if len(cdata):
                jitter = np.random.uniform(-0.18, 0.18, len(cdata))
                ax.scatter(np.full_like(cdata, i) + jitter, cdata,
                           alpha=0.4, s=12, color=color, zorder=3)
                ax.text(i, np.nanmax(cdata) + 1.5 if len(cdata) else 0,
                        f"μ={np.mean(cdata):.0f}%",
                        ha="center", va="bottom", fontsize=6.5, color=TEXT)

        ax.set_xticks(range(1, len(audio_cols) + 1))
        ax.set_xticklabels(audio_cols, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("% of Audio Removed", fontsize=8)
        ax.set_title(title, fontsize=9, color=TEXT, pad=5)
        ax.grid(True, axis="y", linewidth=0.4)
        ax.set_facecolor(PANEL_BG)

    plt.tight_layout()
    p = OUTPUT_DIR / "diag_per_col_boxplots.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Scatter: raw duration vs VAD keep-rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_duration_vs_keeprate(diag: pd.DataFrame):
    """
    If short files consistently have lower keep-rates, the trim alone
    (removing a fixed 0.15s) is disproportionately harming them.
    If long files also have low keep-rates, VAD is the main culprit.
    """
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    fig.suptitle("Raw Duration vs Total Keep-Rate\n"
                 "(reveals whether trim hurts short files or VAD hurts all files)",
                 fontsize=11, color=TEXT)

    for group in GROUPS:
        gdf   = diag[diag["group"] == group]
        color = PALETTE.get(group, "#90CAF9")
        ax.scatter(
            gdf["raw_dur"], gdf["total_kept_pct"],
            label=group, color=color,
            alpha=0.55, s=20, edgecolors="none",
        )

    # Reference lines
    ax.axhline(50, color="#FF5252", linewidth=1, linestyle="--",
               alpha=0.7, label="50% keep threshold")
    ax.axhline(30, color="#FF9800", linewidth=1, linestyle=":",
               alpha=0.7, label="30% keep (current avg)")

    # Trim-only impact curve: for a file of duration D,
    # trim removes LEADING_TRIM_S seconds regardless → trim_pct = 0.15/D * 100
    d_range = np.linspace(0.2, diag["raw_dur"].max() + 1, 200)
    trim_curve = 100.0 * LEADING_TRIM_S / d_range
    ax.plot(d_range, 100 - trim_curve, color="#4FC3F7",
            linewidth=1.2, linestyle="-.",
            label=f"Max possible kept% if only trim applied ({LEADING_TRIM_S}s)")

    ax.set_xlabel("Raw Duration (s)", fontsize=9)
    ax.set_ylabel("Total Keep-Rate (%)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linewidth=0.4)
    ax.set_facecolor(PANEL_BG)

    plt.tight_layout()
    p = OUTPUT_DIR / "diag_duration_vs_keeprate.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_plots(diag: pd.DataFrame):
    print("\n── Plot 1: Stacked removal by group ────────────────────────")
    plot_stacked_removal(diag)

    print("── Plot 2: Retention heatmap ───────────────────────────────")
    plot_retention_heatmap(diag)

    print("── Plot 3: Per audio-type box plots ───────────────────────")
    plot_per_col_boxplots(diag)

    print("── Plot 4: Duration vs keep-rate scatter ──────────────────")
    plot_duration_vs_keeprate(diag)

    print(f"\nAll diagnostic plots saved to:\n  {OUTPUT_DIR}")


def main(max_files: int = 9999):
    csv_path = (PROJECT_ROOT / "Data" / "data_final" /
                "Clinical" / "clinical_all_sessions.csv")
    df = pd.read_csv(csv_path)

    diag = collect_diagnostics(df, max_files=max_files)

    cache_path = OUTPUT_DIR / "diagnostic_cache.csv"
    diag.to_csv(cache_path, index=False)
    print(f"\nDiagnostic data cached -> {cache_path}")

    print_summary(diag)
    run_plots(diag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files", type=int, default=9999,
                        help="Max files to scan (use ~80 for a quick test)")
    parser.add_argument("--from_cache", action="store_true",
                        help="Skip scanning, load diagnostic_cache.csv and replot")
    args = parser.parse_args()

    if args.from_cache:
        cache_path = OUTPUT_DIR / "diagnostic_cache.csv"
        if not cache_path.exists():
            print("No cache found. Run without --from_cache first.")
            sys.exit(1)
        diag = pd.read_csv(cache_path)
        print(f"Loaded {len(diag)} records from cache.")
        print_summary(diag)
        run_plots(diag)
    else:
        main(max_files=args.max_files)