#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_nature_figs_Li_TPSA.py
Customized for Li-ion Electrolyte Research:
- Fig 6 Change: Plot Molecular Weight vs TPSA (instead of LogP).
- Keeps all previous fixes (V8 margins, strict filtering, NPG colors).
"""

import os, re, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# ==========================================
# 1. Global Style & NPG Palette
# ==========================================

NPG_RED    = "#E64B35"
NPG_BLUE   = "#4DBBD5"
NPG_GREEN  = "#00A087"
NPG_DARK   = "#3C5488"
NPG_LIGHT  = "#B0C4DE" 

def set_nature_style():
    """Sets a strict Nature-style plot theme."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6
    plt.rcParams["grid.linewidth"] = 0.4
    plt.rcParams["figure.dpi"] = 300

def setup_axis(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)

# ==========================================
# 2. Data Helpers (TPSA Update Here)
# ==========================================
def canon_smiles(s: str):
    if not isinstance(s, str) or not s.strip(): return None
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) if m else None
    except: return None

# --- UPDATE: Calculate MW and TPSA instead of LogP ---
def get_mol_weight_tpsa(smi):
    """Calculates Molecular Weight and TPSA."""
    m = Chem.MolFromSmiles(smi)
    if m:
        return (Descriptors.MolWt(m), Descriptors.TPSA(m))
    return (np.nan, np.nan)

def morgan_fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

def murcko_scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            if scaf and scaf.GetNumAtoms() > 0:
                return Chem.MolToSmiles(scaf, canonical=True)
    except Exception: return None
    return None

def parse_gate_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return pd.DataFrame(), pd.DataFrame()
    event_re = re.compile(r"\bs=([-\d\.eE]+)\s+vs\s+thr=([-\d\.eE]+)\s+->\s+(PASS|FAIL)")
    events = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = event_re.search(line)
            if m:
                src_raw = "Teacher" if "teacher" in line.lower() else "Generated"
                events.append({"s": float(m.group(1)), "thr": float(m.group(2)), 
                               "decision": m.group(3), "src": src_raw})
    return pd.DataFrame(events), pd.DataFrame()

# ==========================================
# 3. Plotting Functions
# ==========================================

def plot_fig1_similarity_kde(gen_sims, train_sims, out_dir):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    sns.kdeplot(train_sims, fill=True, color="#808080", alpha=0.3, linewidth=0, ax=ax, label="Train Internal")
    sns.kdeplot(train_sims, color="#606060", linewidth=1, ax=ax)
    sns.kdeplot(gen_sims, fill=True, color=NPG_RED, alpha=0.4, linewidth=0, ax=ax, label="Generated (Novel)")
    sns.kdeplot(gen_sims, color=NPG_RED, linewidth=1.2, ax=ax)
    ax.set_xlabel("Max Tanimoto Similarity")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, loc='upper left')
    setup_axis(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "NMI_Fig1_Similarity_Dist.png"))
    plt.close()

def plot_fig2_reward_vs_novelty_hex(df, out_dir):
    if "reward" not in df.columns: return
    g = sns.JointGrid(data=df, x="max_sim_train", y="reward", height=3.5, ratio=5)
    hb = g.ax_joint.hexbin(df["max_sim_train"], df["reward"], 
                           gridsize=30, cmap="Spectral_r", mincnt=1, 
                           linewidths=0.2, edgecolors='none')
    sns.kdeplot(x=df["max_sim_train"], ax=g.ax_marg_x, fill=True, color=NPG_DARK, alpha=0.2, linewidth=0.8)
    sns.kdeplot(y=df["reward"], ax=g.ax_marg_y, fill=True, color=NPG_DARK, alpha=0.2, linewidth=0.8)
    sns.regplot(data=df, x="max_sim_train", y="reward", scatter=False, ax=g.ax_joint, 
                color="#333333", line_kws={'linestyle':'--', 'linewidth':1.0, 'alpha':0.8})
    g.set_axis_labels("Max Similarity to Train", "Reward / Property Score")
    
    # V8 Fix Layout
    cax = g.fig.add_axes([0.83, 0.15, 0.02, 0.35]) 
    cb = plt.colorbar(hb, cax=cax)
    cb.set_label('Molecule Count', rotation=270, labelpad=12, fontsize=6)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6, width=0.5)

    setup_axis(g.ax_joint)
    g.ax_marg_x.spines["left"].set_visible(False)
    g.ax_marg_y.spines["bottom"].set_visible(False)
    g.ax_marg_x.spines["bottom"].set_visible(False)
    g.ax_marg_y.spines["left"].set_visible(False)
    
    plt.subplots_adjust(left=0.18, right=0.80, top=0.92, bottom=0.15)
    plt.savefig(os.path.join(out_dir, "NMI_Fig2_Reward_Landscape.png"))
    plt.close()

def plot_fig3_tradeoff_dual_axis(thresholds, acceptance, mean_rewards, out_dir):
    fig, ax1 = plt.subplots(figsize=(3.4, 2.6))
    color1 = NPG_DARK
    ax1.set_xlabel("Similarity Threshold (t)")
    ax1.set_ylabel("Acceptance Rate", color=color1)
    ax1.plot(thresholds, acceptance, color=color1, marker='o', markersize=3, 
             linewidth=1, label="Acceptance")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = NPG_RED
    ax2.set_ylabel("Mean Reward (Kept)", color=color2)
    ax2.plot(thresholds, mean_rewards, color=color2, marker='s', markersize=3, 
             linestyle='--', linewidth=1, label="Reward")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(True, axis='x', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "NMI_Fig3_Tradeoff_Combined.png"))
    plt.close()

def plot_fig4_metrics_horizontal(metrics_dict, out_dir):
    labels = ["Validity", "Uniqueness", "Novelty", "Scaffold Nov."]
    keys = ["validity", "uniqueness", "novelty_exact", "scaffold_novelty"]
    values = [metrics_dict[k] for k in keys]
    colors = [NPG_DARK, "#5A75A8", "#7997C8", NPG_BLUE]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='none', height=0.6, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Ratio")
    setup_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='y', length=0)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center', fontsize=7, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "NMI_Fig4_Metrics_Bar.png"))
    plt.close()

def plot_fig5_gate_violin(events_df, out_dir):
    if events_df.empty: return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    plot_data = events_df[events_df['s'] > events_df['s'].quantile(0.005)].copy()
    my_pal = {"Generated": "#A0D8EF", "Teacher": "#F4B5B5"}
    sns.violinplot(data=plot_data, x="src", y="s", palette=my_pal, 
                   inner="quartile", linewidth=0.8, ax=ax, saturation=0.7, width=0.6)
    median_thr = events_df['thr'].median()
    ax.axhline(median_thr, color="#555555", linestyle='--', linewidth=0.8, alpha=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Gate Score s(z)")
    setup_axis(ax, grid=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "NMI_Fig5_Gate_Violin.png"))
    plt.close()

# --- UPDATE: Fig 6 adapted for MW vs TPSA ---
def plot_fig6_chem_space(gen_df, train_df, out_dir, x_col, y_col):
    """
    修改版：支持指定 x_col 和 y_col 进行绘图
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    
    # 1. 绘制训练集背景 (灰色)
    # 检查训练集是否有这两列，如果没有则跳过不画背景，防止报错
    if x_col in train_df.columns and y_col in train_df.columns:
        # 为了不让图太卡，只随机采样 3000 个点
        if len(train_df) > 3000:
            train_sample = train_df.sample(3000)
        else:
            train_sample = train_df
        
        # 处理一下可能的 NaN 值
        train_sample = train_sample.dropna(subset=[x_col, y_col])
        
        ax.scatter(train_sample[x_col], train_sample[y_col], 
                   c='#E8E8E8', s=6, alpha=0.5, label='Train', edgecolors='none', zorder=1)
    
    # 2. 绘制生成集前景 (彩色)
    if x_col in gen_df.columns and y_col in gen_df.columns:
        # 去除空值
        gen_clean = gen_df.dropna(subset=[x_col, y_col])
        
        if "reward" in gen_clean.columns:
            sc = ax.scatter(gen_clean[x_col], gen_clean[y_col], c=gen_clean["reward"], cmap="Spectral_r", 
                            s=8, alpha=0.8, label='Generated (Novel)', edgecolors='none', zorder=2)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Reward", rotation=270, labelpad=10)
            cbar.outline.set_linewidth(0.5)
            cbar.ax.tick_params(width=0.5)
        else:
            ax.scatter(gen_clean[x_col], gen_clean[y_col], c=NPG_BLUE, s=8, alpha=0.6, label='Generated (Novel)', zorder=2)
    else:
        print(f"Error: Columns {x_col} or {y_col} not found in Generated DataFrame!")

    # 设置轴标签
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    ax.legend(loc='upper left', frameon=False, markerscale=1.5)
    setup_axis(ax)
    
    plt.tight_layout()
    # 文件名加上列名，防止覆盖
    save_name = f"NMI_Fig6_ChemSpace_{x_col}_vs_{y_col}.png"
    plt.savefig(os.path.join(out_dir, save_name))
    plt.close()

# ==========================================
# Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_csv", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--out_dir", default="nature_figs_Li_TPSA")
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_nature_style()
    
    print("Loading data...")
    gen = pd.read_csv(args.gen_csv)
    train = pd.read_csv(args.train_csv)
    gen["smiles_canon"] = gen["smiles"].map(canon_smiles)
    train["smiles_canon"] = train["smiles"].map(canon_smiles)
    gen = gen.dropna(subset=["smiles_canon"])
    train = train.dropna(subset=["smiles_canon"])
    
    gen_unique = gen.drop_duplicates(subset=["smiles_canon"]).copy()
    validity = len(gen) / max(len(pd.read_csv(args.gen_csv)), 1)
    uniqueness = len(gen_unique) / max(len(gen), 1)
    train_set = set(train["smiles_canon"])
    gen_unique["is_in_train"] = gen_unique["smiles_canon"].isin(train_set)
    novelty = 1.0 - gen_unique["is_in_train"].mean()
    gen_unique["scaffold"] = gen_unique["smiles_canon"].map(murcko_scaffold)
    train_scafs = set([murcko_scaffold(s) for s in train["smiles_canon"] if s])
    scaf_novelty = 1.0 - gen_unique["scaffold"].isin(train_scafs).mean()
    metrics = {"validity": validity, "uniqueness": uniqueness, 
               "novelty_exact": novelty, "scaffold_novelty": scaf_novelty}
    
    print("Computing fingerprints...")
    train_fps = [morgan_fp(s) for s in train["smiles_canon"]]
    train_fps = [x for x in train_fps if x]
    
    def get_max_sim(smi):
        fp = morgan_fp(smi)
        if not fp: return np.nan
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        return max(sims) if sims else 0.0
    
    gen_unique["max_sim_train"] = gen_unique["smiles_canon"].apply(get_max_sim)
    gen_novel_strict = gen_unique[gen_unique["max_sim_train"] < 0.999].copy()
    
    import random
    train_sample_fps = random.sample(train_fps, min(2000, len(train_fps)))
    train_internal_sims = []
    for i, fp in enumerate(train_sample_fps):
        others = train_sample_fps[:i] + train_sample_fps[i+1:]
        if others: train_internal_sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, others)))
            
    thresholds = np.linspace(0.0, 0.6, 20)
    acc_rates = []
    mean_rews = []
    for t in thresholds:
        kept = gen_novel_strict[gen_novel_strict["max_sim_train"] >= t]
        acc_rates.append(len(kept) / len(gen_novel_strict) if len(gen_novel_strict) else 0)
        mean_rews.append(kept["reward"].mean() if "reward" in kept.columns and len(kept) else np.nan)

    print("Generating figures (Li Electrolyte Version - TPSA)...")
    plot_fig1_similarity_kde(gen_novel_strict["max_sim_train"], train_internal_sims, args.out_dir)
    plot_fig2_reward_vs_novelty_hex(gen_novel_strict, args.out_dir)
    plot_fig3_tradeoff_dual_axis(thresholds, acc_rates, mean_rews, args.out_dir)
    plot_fig4_metrics_horizontal(metrics, args.out_dir)
    
    ev, st = parse_gate_log(args.log)
    plot_fig5_gate_violin(ev, args.out_dir)
    
    # ================= 修改开始 =================
    # 这里指定你想画的两个性质！
    # 如果是 CSV 里已有的列（比如 'phys_homo'），直接写列名。
    # 如果是 RDKit 计算的性质（比如 MW, LogP），需要先算出来加到 DataFrame 里。
    
    # === 场景 1：你想画 CSV 里已有的物理性质 (例如 phys_homo vs phys_gap) ===
    # 只要你的 gen_csv 和 train_csv 里有这几列，直接把下面两行取消注释并修改名字即可：
    
    target_x = "pred_mu"  # 你的CSV列名
    target_y = "pred_gap"   # 你的CSV列名
    plot_fig6_chem_space(gen_novel_strict, train, args.out_dir, x_col=target_x, y_col=target_y)

if __name__ == "__main__":
    main()