#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Li 电解液方向：给 CSV 增加 RDKit LogP，并画 LogP parity 图 (论文发表级样式)。

用法示例：
python add_rdkit_logp_styled.py \
  --csv  ./data/test_full.csv \
  --out  ./data/test_full_with_logp.csv \
  --plot ./plots/logp_parity.png \
  --smiles-col smiles \
  --phys-col "pred_LogP" \
  --quiet-rdkit
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

# ==========================================
# 1. 全局绘图风格设置
# ==========================================
# 字体路径候选列表
FONT_CANDIDATES = [
    "/public/home/users/haoxw/generate_AI/branch_OLED/code/arial.ttf",
    "./arial.ttf",
    "arial.ttf"
]

def setup_style():
    """配置 matplotlib 以符合科学出版标准"""
    # 1. 尝试加载 Arial 字体
    font_loaded = False
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                prop = fm.FontProperties(fname=path)
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [prop.get_name(), 'Arial', 'DejaVu Sans']
                print(f"[Plot] Loaded font from: {path}")
                font_loaded = True
                break
            except Exception as e:
                print(f"[Plot] Warning: Failed to load font {path}: {e}")
    
    if not font_loaded:
        print("[Plot] Warning: Arial font not found. Using default sans-serif.")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

    # 2. 论文级参数设置 (3.5 inch, 7pt/8pt)
    plt.rcParams['mathtext.fontset'] = 'stixsans' # 数学公式无衬线
    plt.rcParams['axes.unicode_minus'] = False 
    plt.rcParams['font.size'] = 7           
    plt.rcParams['axes.labelsize'] = 8      
    plt.rcParams['axes.titlesize'] = 8      
    plt.rcParams['xtick.labelsize'] = 7     
    plt.rcParams['ytick.labelsize'] = 7     
    plt.rcParams['legend.fontsize'] = 7     

# ==========================================
# 2. RDKit 计算功能 (LogP)
# ==========================================
def compute_rdkit_logp(smiles: str) -> float:
    """Return RDKit Crippen MolLogP; NaN if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return float("nan")
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return float("nan")
        return float(Crippen.MolLogP(m))
    except Exception:
        return float("nan")

def insert_column_after(df: pd.DataFrame, after_col: str, new_col: str, values) -> pd.DataFrame:
    """Insert new_col right after after_col."""
    if new_col in df.columns:
        df = df.drop(columns=[new_col])
    
    if after_col in df.columns:
        idx = df.columns.get_loc(after_col) + 1
        df.insert(idx, new_col, values)
    else:
        df[new_col] = values
    return df

# ==========================================
# 3. 绘图功能 (Seaborn JointGrid)
# ==========================================
def plot_parity_styled(x, y, xlabel, ylabel, out_path):
    """
    绘制带边缘分布、统计信息框的高级对偶图。
    x: RDKit values
    y: Phys values
    """
    # 剔除无效值
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        print("[Plot] Error: No valid data points to plot.")
        return

    # 计算统计指标
    n_count = len(x)
    mae = mean_absolute_error(y, x)
    rmse = np.sqrt(mean_squared_error(y, x))
    r2 = r2_score(y, x)

    # 确定范围 (留 5% 边距)
    vmin, vmax = min(x.min(), y.min()), max(x.max(), y.max())
    span = vmax - vmin
    lo, hi = vmin - span * 0.05, vmax + span * 0.05

    # --- 核心绘图 ---
    # 创建 3.5 英寸画布 (双栏单图标准)
    g = sns.JointGrid(x=x, y=y, height=3.5, ratio=5)

    # 1. 散点图 (蓝色带白边)
    g.plot_joint(plt.scatter, color='#4c72b0', s=15, alpha=0.7, 
                 edgecolor='white', linewidth=0.4)

    # 2. 边缘密度图
    g.plot_marginals(sns.kdeplot, fill=True, color='#4c72b0', alpha=0.3, linewidth=0)

    # 3. 对角线
    g.ax_joint.plot([lo, hi], [lo, hi], linestyle='--', color='gray', linewidth=1.0, alpha=0.8)

    # 4. 标签设置
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    g.ax_joint.set_xlim(lo, hi)
    g.ax_joint.set_ylim(lo, hi)

    # 5. 统计框 (包含 N, MAE, RMSE, R2)
    stats_text = (f"N = {n_count}\n"
                  f"MAE: {mae:.3f}\n"
                  f"RMSE: {rmse:.3f}\n"
                  f"R$^2$: {r2:.3f}")
    
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.5, alpha=1.0)
    
    # 放在右下角
    g.ax_joint.text(0.95, 0.05, stats_text, transform=g.ax_joint.transAxes,
                    ha='right', va='bottom', bbox=props, fontsize=7)

    # 保存图片
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    # 保存 PNG (300 DPI)
    g.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved PNG: {out_path}")
    
    # 保存 PDF
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    g.savefig(pdf_path, bbox_inches='tight')
    print(f"[Plot] Saved PDF: {pdf_path}")
    
    plt.close()

# ==========================================
# 4. 主流程
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入CSV（包含 smiles 与 物理值）")
    ap.add_argument("--out", required=True, help="输出CSV（会新增 rdkit 计算列）")
    ap.add_argument("--plot", required=True, help="输出图路径 (PNG)")

    ap.add_argument("--smiles-col", default="smiles", help="SMILES 列名")
    ap.add_argument("--phys-col", default="pred_LogP", help="物理尺度值列名 (y轴)")
    ap.add_argument("--rdkit-col", default="rdkit_LogP", help="RDKit 计算值列名 (x轴)")

    ap.add_argument("--quiet-rdkit", action="store_true", help="静默RDKit错误日志")
    args = ap.parse_args()

    # 初始化绘图风格
    setup_style()

    if args.quiet_rdkit:
        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")

    # 读取数据
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"输入文件不存在: {args.csv}")
        
    df = pd.read_csv(args.csv)

    if args.smiles_col not in df.columns:
        raise KeyError(f"CSV中找不到SMILES列: '{args.smiles_col}'")
    
    # 如果物理列不存在，可能是用户想算 LogP 但没有对比值，我们先算算看
    has_phys = args.phys_col in df.columns
    if not has_phys:
        print(f"Warning: 物理列 '{args.phys_col}' 不存在，将只计算 RDKit 值，不画图。")

    print(f"正在计算 RDKit LogP (Total: {len(df)})...")
    # 计算 rdkit_LogP
    rdkit_vals = [compute_rdkit_logp(str(s)) for s in df[args.smiles_col]]
    
    # 插入新列
    # 如果有物理列，插在物理列后面；否则插在 SMILES 后面
    insert_pos = args.phys_col if has_phys else args.smiles_col
    df = insert_column_after(df, insert_pos, args.rdkit_col, rdkit_vals)

    # 保存 CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote CSV: {args.out}")

    # 画图 (只有当物理值存在时)
    if has_phys:
        x_vals = pd.to_numeric(df[args.rdkit_col], errors="coerce").values
        y_vals = pd.to_numeric(df[args.phys_col], errors="coerce").values
        
        # 定义轴标签
        xlabel = "RDKit LogP"
        ylabel = args.phys_col  # 通常 LogP 无量纲，不加单位
        
        plot_parity_styled(x_vals, y_vals, xlabel, ylabel, args.plot)

if __name__ == "__main__":
    main()