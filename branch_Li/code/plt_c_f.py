#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot panels (c)–(f) with publication-style formatting for Li-electrolyte properties.
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
        kde = gaussian_kde(xvals)  # Scott's Rule 带宽（默认）
        return kde(grid)
    # fallback: sklearn
    kde = KernelDensity(kernel="gaussian", bandwidth=1.06*np.std(xvals)*(xvals.size ** (-1/5)))  # Silverman 近似
    kde.fit(xvals.reshape(-1,1))
    logd = kde.score_samples(grid.reshape(-1,1))
    return np.exp(logd)

# ---------------------- Style ----------------------
DEF_FIGSIZE = (7.2, 5.2)      # inches
DEF_DPI = 300
LW_LINE = 2.0                 # 线条加粗一点
LW_LINE_CONNECT = 2.0         
ALPHA_FILL = 0.45             # 填充稍微透明一点，方便看重叠
ALPHA_LINE = 0.95
NCOLOR_TRAIN = "#1f77b4"      # blue
NCOLOR_GEN = "#ffb000"        # yellow (color-blind friendly)
GRID_ALPHA = 0.15
BINS = 50                     # 默认 bin 数量

mpl.rcParams.update({
    "font.size": 12,          # 字体稍微大一点
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
    # 统一 bin 边界
    if len(train_vals) > 0 and len(gen_vals) > 0:
        # 为了美观，取 1% 和 99% 分位数来确定显示范围，防止极值拉伸图形
        all_vals = np.concatenate([train_vals, gen_vals])
        vmin = np.percentile(all_vals, 0.5)
        vmax = np.percentile(all_vals, 99.5)
        # 稍微放宽一点边界
        margin = (vmax - vmin) * 0.1
        vmin -= margin
        vmax += margin
        
        edges = np.linspace(vmin, vmax, bins)
    else:
        edges = bins

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # 1. 绘制直方图 (Histogram)
    ax.hist(train_vals, bins=edges, density=True, alpha=ALPHA_FILL,
            color=NCOLOR_TRAIN, label="Training set", edgecolor="none")
    ax.hist(gen_vals, bins=edges, density=True, alpha=ALPHA_FILL,
            color=NCOLOR_GEN, label="Generated", edgecolor="none")

    # 2. 绘制 KDE 曲线 (平滑分布)
    if isinstance(edges, (list, np.ndarray)):
        lo, hi = edges[0], edges[-1]
    else:
        lo, hi = np.min(np.r_[train_vals, gen_vals]), np.max(np.r_[train_vals, gen_vals])
    
    grid = np.linspace(lo, hi, 512)
    
    # Training KDE
    if len(train_vals) > 1:
        density_tr = _kde_curve(train_vals, grid)
        ax.plot(grid, density_tr, color=NCOLOR_TRAIN, linewidth=LW_LINE, alpha=ALPHA_LINE, label="Training KDE")
    
    # Generated KDE
    if len(gen_vals) > 1:
        density_ge = _kde_curve(gen_vals, grid)
        ax.plot(grid, density_ge, color=NCOLOR_GEN, linewidth=LW_LINE, alpha=ALPHA_LINE, label="Generated KDE")

    # 3. 绘制中位数虚线
    if len(train_vals) > 0:
        ax.axvline(np.median(train_vals), color=NCOLOR_TRAIN, linestyle="--",
                   linewidth=LW_LINE, alpha=0.8, label="Training median")
    if len(gen_vals) > 0:
        ax.axvline(np.median(gen_vals), color=NCOLOR_GEN, linestyle="-.",
                   linewidth=LW_LINE, alpha=0.8, label="Generated median")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=0.8)

    # 整理图例 (去重)
    handles, labels = ax.get_legend_handles_labels()
    unique = []
    seen = set()
    for h, l in zip(handles, labels):
        if l == "_nolegend_": continue
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    
    if unique:
        ax.legend([h for h, _ in unique], [l for _, l in unique], frameon=False, loc='best')

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    print(f"[Saved] {outpath.resolve()}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot publication-style panels for Li-electrolyte.")
    ap.add_argument("--train", required=True, help="Training set CSV")
    ap.add_argument("--gen", required=True, help="Generated set CSV")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--bins", type=int, default=BINS, help="Number of bins")
    args = ap.parse_args()

    print(f"Loading Training data: {args.train}")
    train_df = read_table(args.train)
    print(f"Loading Generated data: {args.gen}")
    gen_df = read_table(args.gen)

    # ================= 修改区域 Start =================
    # 1. 定义需要的列名 (根据你的 Li_iron 数据)
    #    Training CSV 列名: mu, gap, cv, TPSA
    #    Generated CSV 列名: mu, gap, cv, TPSA (假设一致，或者是 mu_phys_... 如果前面有 unscale 脚本)
    
    # 这里我们做一个智能判断，兼容 'mu' 和 'mu_phys_Debye' 这种带单位的列名
    def get_col_data(df, keywords):
        # 尝试完全匹配
        for k in keywords:
            if k in df.columns:
                return pick_numeric(df[k])
        # 尝试模糊匹配 (比如找包含 'mu' 和 'Debye' 的列)
        for col in df.columns:
            if keywords[0] in col: # 只要包含关键词的前缀
                return pick_numeric(df[col])
        print(f"⚠️ Warning: Could not find column for {keywords} in {df.columns}")
        return pd.Series([], dtype=float)

    # 2. 读取数据 (Training vs Generated)
    # Mu (偶极矩)
    mu_tr = get_col_data(train_df, ["mu", "dipole"])
    mu_ge = get_col_data(gen_df,   ["pred_mu", "mu_phys_Debye", "dipole"])

    # Gap (能隙)
    gap_tr = get_col_data(train_df, ["gap", "gap_eV"])
    gap_ge = get_col_data(gen_df,   ["pred_gap", "gap_phys_Hartree", "gap_phys_eV"])

    # CV (热容)
    cv_tr = get_col_data(train_df, ["cv", "heat_capacity"])
    cv_ge = get_col_data(gen_df,   ["pred_cv", "cv_phys_cal_molK"])

    # TPSA (极性表面积)
    tpsa_tr = get_col_data(train_df, ["TPSA", "tpsa"])
    tpsa_ge = get_col_data(gen_df,   ["pred_TPSA", "TPSA_phys_A2", "tpsa"])

    outdir = Path(args.outdir)

    # ================= 绘图调用 =================
    
    # Panel (c): Mu
    if len(mu_tr) > 0 and len(mu_ge) > 0:
        panel_hist(
            train_vals=mu_tr.to_numpy(),
            gen_vals=mu_ge.to_numpy(),
            xlabel="Dipole Moment $\mu$ (Debye)",
            title="(c) Distribution of Dipole Moment",
            outpath=outdir / "figure_c_mu.png",
            bins=args.bins,
        )

    # Panel (d): Gap
    if len(gap_tr) > 0 and len(gap_ge) > 0:
        panel_hist(
            train_vals=gap_tr.to_numpy(),
            gen_vals=gap_ge.to_numpy(),
            xlabel="HOMO-LUMO Gap (Hartree)",
            title="(d) Distribution of Energy Gap",
            outpath=outdir / "figure_d_gap.png",
            bins=args.bins,
        )

    # Panel (e): CV
    if len(cv_tr) > 0 and len(cv_ge) > 0:
        panel_hist(
            train_vals=cv_tr.to_numpy(),
            gen_vals=cv_ge.to_numpy(),
            xlabel="Heat Capacity $C_v$ (cal/mol·K)",
            title="(e) Distribution of Heat Capacity",
            outpath=outdir / "figure_e_cv.png",
            bins=args.bins,
        )

    # Panel (f): TPSA
    if len(tpsa_tr) > 0 and len(tpsa_ge) > 0:
        panel_hist(
            train_vals=tpsa_tr.to_numpy(),
            gen_vals=tpsa_ge.to_numpy(),
            xlabel="TPSA ($\AA^2$)",
            title="(f) Distribution of TPSA",
            outpath=outdir / "figure_f_tpsa.png",
            bins=args.bins,
        )

    print("[Done] Panels (c)–(f) for Li-electrolyte exported.")

if __name__ == "__main__":
    main()