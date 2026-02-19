#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot panels (c)–(f) with publication-style formatting for OPV properties.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 可选：scipy KDE
try:
    from scipy.stats import gaussian_kde
    _have_scipy = True
except Exception:
    _have_scipy = False
    from sklearn.neighbors import KernelDensity

def _kde_curve(xvals: np.ndarray, grid: np.ndarray):
    xvals = np.asarray(xvals, dtype=float)
    if xvals.size < 2:
        return np.zeros_like(grid)
    if _have_scipy:
        kde = gaussian_kde(xvals)
        return kde(grid)
    kde = KernelDensity(kernel="gaussian", bandwidth=1.06*np.std(xvals)*(xvals.size ** (-1/5)))
    kde.fit(xvals.reshape(-1,1))
    logd = kde.score_samples(grid.reshape(-1,1))
    return np.exp(logd)

# ---------------------- Style ----------------------
DEF_FIGSIZE = (7.2, 5.2)
DEF_DPI = 300
LW_LINE = 2.0
LW_LINE_CONNECT = 2.0 
ALPHA_FILL = 0.45
ALPHA_LINE = 0.95
NCOLOR_TRAIN = "#1f77b4"      # blue
NCOLOR_GEN = "#ffb000"        # yellow (color-blind friendly)
GRID_ALPHA = 0.15
BINS = 50

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "figure.dpi": DEF_DPI,
    "savefig.dpi": DEF_DPI,
})

# ---------------------- IO helpers ----------------------
def _norm(s: str) -> str:
    return s.replace("\ufeff", "").strip()

def read_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.rename(columns={c: _norm(c) for c in df.columns}, inplace=True)
    return df

def pick_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s

def panel_hist(train_vals, gen_vals, xlabel, title, outpath, bins=BINS, figsize=DEF_FIGSIZE):
    if len(train_vals) > 0 and len(gen_vals) > 0:
        all_vals = np.concatenate([train_vals, gen_vals])
        vmin = np.percentile(all_vals, 0.5)
        vmax = np.percentile(all_vals, 99.5)
        margin = (vmax - vmin) * 0.1
        edges = np.linspace(vmin - margin, vmax + margin, bins)
    else:
        edges = bins

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.hist(train_vals, bins=edges, density=True, alpha=ALPHA_FILL,
            color=NCOLOR_TRAIN, label="Training set", edgecolor="none")
    ax.hist(gen_vals, bins=edges, density=True, alpha=ALPHA_FILL,
            color=NCOLOR_GEN, label="Generated", edgecolor="none")

    if isinstance(edges, (list, np.ndarray)):
        lo, hi = edges[0], edges[-1]
    else:
        lo, hi = np.min(np.r_[train_vals, gen_vals]), np.max(np.r_[train_vals, gen_vals])
    
    grid = np.linspace(lo, hi, 512)
    
    if len(train_vals) > 1:
        ax.plot(grid, _kde_curve(train_vals, grid), color=NCOLOR_TRAIN, linewidth=LW_LINE, label="Training KDE")
    if len(gen_vals) > 1:
        ax.plot(grid, _kde_curve(gen_vals, grid), color=NCOLOR_GEN, linewidth=LW_LINE, label="Generated KDE")

    if len(train_vals) > 0:
        ax.axvline(np.median(train_vals), color=NCOLOR_TRAIN, linestyle="--", linewidth=LW_LINE, label="Training median")
    if len(gen_vals) > 0:
        ax.axvline(np.median(gen_vals), color=NCOLOR_GEN, linestyle="-.", linewidth=LW_LINE, label="Generated median")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)

    handles, labels = ax.get_legend_handles_labels()
    unique = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            unique.append((h, l))
            seen.add(l)
    if unique:
        ax.legend([h for h, _ in unique], [l for _, l in unique], frameon=False)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    print(f"[Saved] {outpath.resolve()}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot publication-style panels for OPV.")
    ap.add_argument("--train", required=True, help="Training set CSV")
    ap.add_argument("--gen", required=True, help="Generated set CSV")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--bins", type=int, default=BINS, help="Number of bins")
    args = ap.parse_args()

    train_df = read_table(args.train)
    gen_df = read_table(args.gen)

    def get_col_data(df, keywords):
        for k in keywords:
            if k in df.columns: return pick_numeric(df[k])
        for col in df.columns:
            if keywords[0].lower() in col.lower(): return pick_numeric(df[col])
        print(f"⚠️ Warning: Not found {keywords}")
        return pd.Series([], dtype=float)

    # === OPV Properties ===
    # HOMO - 优先找 phys_homo
    homo_tr = get_col_data(train_df, ["homo", "homo_eV"])
    homo_ge = get_col_data(gen_df,   ["pred_homo", "homo_phys_eV"])

    # Gap - 优先找 phys_gap
    gap_tr = get_col_data(train_df, ["gap", "gap_eV"])
    gap_ge = get_col_data(gen_df,   ["pred_gap", "gap_phys_eV"])

    # MolMR - 优先找 phys_MolMR
    mr_tr = get_col_data(train_df, ["MolMR", "molmr"])
    mr_ge = get_col_data(gen_df,   ["pred_MolMR", "MolMR_phys_a.u."])

    # LogP - 优先找 phys_LogP
    logp_tr = get_col_data(train_df, ["LogP", "logp"])
    logp_ge = get_col_data(gen_df,   ["pred_LogP", "LogP_phys_a.u."])

    outdir = Path(args.outdir)

    if len(homo_tr)>0: panel_hist(homo_tr.to_numpy(), homo_ge.to_numpy(), "HOMO (eV)", "(c) Distribution of HOMO", outdir/"figure_c_homo.png", args.bins)
    if len(gap_tr)>0:  panel_hist(gap_tr.to_numpy(), gap_ge.to_numpy(), "Gap (eV)", "(d) Distribution of Gap", outdir/"figure_d_gap.png", args.bins)
    if len(mr_tr)>0:   panel_hist(mr_tr.to_numpy(), mr_ge.to_numpy(), "MolMR (a.u.)", "(e) Distribution of MolMR", outdir/"figure_e_molmr.png", args.bins)
    if len(logp_tr)>0: panel_hist(logp_tr.to_numpy(), logp_ge.to_numpy(), "LogP", "(f) Distribution of LogP", outdir/"figure_f_logp.png", args.bins)

    print("[Done] OPV Panels exported.")

if __name__ == "__main__":
    main()