#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot panels (c)–(f) with publication-style formatting,
and draw a line connecting the histogram bars for each dataset.
"""
# 放在文件顶部
import numpy as np
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
        kde = gaussian_kde(xvals)  # Scott's Rule 带宽（默认）
        return kde(grid)
    # fallback: sklearn
    kde = KernelDensity(kernel="gaussian", bandwidth=1.06*np.std(xvals)*(xvals.size ** (-1/5)))  # Silverman 近似
    kde.fit(xvals.reshape(-1,1))
    logd = kde.score_samples(grid.reshape(-1,1))
    return np.exp(logd)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------- Style ----------------------
DEF_FIGSIZE = (7.2, 5.2)      # inches
DEF_DPI = 300
LW_LINE = 1.8
LW_LINE_CONNECT = 2.0         # 连线宽度
ALPHA_FILL = 0.55
ALPHA_LINE = 0.95
NCOLOR_TRAIN = "#1f77b4"      # blue
NCOLOR_GEN = "#ffb000"        # yellow (color-blind friendly)
GRID_ALPHA = 0.15
BINS = 40                     # default number of bins (auto edges set per panel)

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
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

# ---------------------- Plotting ----------------------
def panel_hist(
    train_vals: np.ndarray,
    gen_vals: np.ndarray,
    xlabel: str,
    title: str,
    outpath: Path,
    bins: int = BINS,
    figsize=DEF_FIGSIZE,
):
    # 统一 bin 边界（两数据集联合范围）
    if len(train_vals) > 0 and len(gen_vals) > 0:
        vmin = float(np.min([train_vals.min(), gen_vals.min()]))
        vmax = float(np.max([train_vals.max(), gen_vals.max()]))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
            edges = np.linspace(vmin, vmax, bins)
        else:
            edges = bins
    else:
        edges = bins

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # 叠加直方图（密度归一化）
    ax.hist(train_vals, bins=edges, density=True, alpha=ALPHA_FILL,
            color=NCOLOR_TRAIN, label="Training set", edgecolor="none")
    ax.hist(gen_vals, bins=edges, density=True, alpha=ALPHA_FILL,
            color=NCOLOR_GEN, label="Generated", edgecolor="none")

    # 设定与直方图同范围的光滑曲线网格
    if isinstance(edges, (list, np.ndarray)):
        lo, hi = edges[0], edges[-1]
    else:
        lo, hi = np.min(np.r_[train_vals, gen_vals]), np.max(np.r_[train_vals, gen_vals])
    grid = np.linspace(lo, hi, 512)
    
    # 训练集 KDE（蓝色）
    density_tr = _kde_curve(train_vals, grid)
    ax.plot(grid, density_tr, color=NCOLOR_TRAIN, linewidth=2.0, alpha=ALPHA_LINE, label="Training KDE")
    
    # 生成集 KDE（黄色）
    density_ge = _kde_curve(gen_vals, grid)
    ax.plot(grid, density_ge, color=NCOLOR_GEN, linewidth=2.0, alpha=ALPHA_LINE, label="Generated KDE")


    # 中位数标注
    if len(train_vals) > 0:
        ax.axvline(np.median(train_vals), color=NCOLOR_TRAIN, linestyle="--",
                   linewidth=LW_LINE, alpha=ALPHA_LINE, label="Training median")
    if len(gen_vals) > 0:
        ax.axvline(np.median(gen_vals), color=NCOLOR_GEN, linestyle="-.",
                   linewidth=LW_LINE, alpha=ALPHA_LINE, label="Generated median")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=0.8)
    # 去重 legend（避免重复 handle）
    handles, labels = ax.get_legend_handles_labels()
    unique = []
    seen = set()
    for h, l in zip(handles, labels):
        if l == "_nolegend_":
            continue
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    if unique:
        ax.legend([h for h, _ in unique], [l for _, l in unique], frameon=False)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath.resolve()}")

def main():
    ap = argparse.ArgumentParser(description="Plot publication-style panels (c)–(f).")
    ap.add_argument("--train", required=True, help="Training set CSV (with r0, Q, D, P)")
    ap.add_argument("--gen", required=True, help="Generated set CSV (with *_phys_* columns)")
    ap.add_argument("--outdir", default="fig_c_to_f_pub", help="Output directory")
    ap.add_argument("--bins", type=int, default=BINS, help="Number of bins")
    args = ap.parse_args()

    train_df = read_table(args.train)
    gen_df = read_table(args.gen)

    # 列存在性校验 + 读取
    need_train = ["D", "P", "Q", "r0"]
    need_gen   = ["D_phys_km_s", "P_phys_kbar", "EG_phys_m_s", "r0_phys_g_cm3"]
    for c in need_train:
        if c not in train_df.columns:
            raise SystemExit(f"Column '{c}' not found in training CSV.")
    for c in need_gen:
        if c not in gen_df.columns:
            raise SystemExit(f"Column '{c}' not found in generated CSV.")

    D_tr  = pick_numeric(train_df["D"])
    P_tr  = pick_numeric(train_df["P"])
    Q_tr  = pick_numeric(train_df["Q"])               # m/s
    r0_tr = pick_numeric(train_df["r0"])

    D_ge  = pick_numeric(gen_df["D_phys_km_s"])       # km/s
    P_ge  = pick_numeric(gen_df["P_phys_kbar"])       # kbar
    EG_ge = pick_numeric(gen_df["EG_phys_m_s"] / 1000.0)  # m/s -> km/s
    r0_ge = pick_numeric(gen_df["r0_phys_g_cm3"])     # g/cm^3

    outdir = Path(args.outdir)

    # c — D (km/s)
    panel_hist(
        train_vals=D_tr.to_numpy(),
        gen_vals=D_ge.to_numpy(),
        xlabel="Detonation velocity D (km/s)",
        title="(c) Distribution of D",
        outpath=outdir / "figure_c.png",
        bins=args.bins,
    )

    # d — P (kbar)
    panel_hist(
        train_vals=P_tr.to_numpy(),
        gen_vals=P_ge.to_numpy(),
        xlabel="Detonation pressure P (kbar)",
        title="(d) Distribution of P",
        outpath=outdir / "figure_d.png",
        bins=args.bins,
    )

    # e — Q / EG (统一为 km/s)
    panel_hist(
        train_vals=(Q_tr).to_numpy(),  
        gen_vals=EG_ge.to_numpy(),               # 已是 km/s
        xlabel="Energy/Gurney proxy (Q / EG, km/s)",
        title="(e) Distribution of Q / EG",
        outpath=outdir / "figure_e.png",
        bins=args.bins,
    )

    # f — r0 (g/cm³)
    panel_hist(
        train_vals=r0_tr.to_numpy(),
        gen_vals=r0_ge.to_numpy(),
        xlabel="Crystal density r0 (g/cm³)",
        title="(f) Distribution of r0",
        outpath=outdir / "figure_f.png",
        bins=args.bins,
    )

    print("[Done] Panels (c)–(f) exported.")

if __name__ == "__main__":
    main()
