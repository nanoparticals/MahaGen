import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# 1. 配置：字体、路径与单位
# ==========================================
FONT_PATH = "/public/home/users/haoxw/generate_AI/branch_OLED/code/arial.ttf"
DATA_ROOT = "/public/home/users/haoxw/generate_AI/branch_Li/ckpts/chemprop_DPER0_cv5_new_7"
OUT_DIR = "plots_final_paper_units"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 【新增】单位字典：请在这里修改你的实际单位 ---
# 如果某项没有单位（如 LogP），留空字符串 "" 即可
TARGET_UNITS = {
    "mu": "D",                  # 偶极矩 (Dipole Moment)
    "gap": "eV",                # 带隙
    "cv": "cal/(mol$\\cdot$K)", # 热容 (使用 LaTeX 的 \\cdot 显示中间的点)
    "TPSA": "$\\mathring{A}^2$" # 拓扑极性表面积 (使用 \\mathring{A} 显示埃，^2 显示平方)
}

# 注册 Arial 字体
try:
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        prop = fm.FontProperties(fname=FONT_PATH)
        custom_font_name = prop.get_name()
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [custom_font_name, 'Arial', 'DejaVu Sans']
        print(f"✅ 成功加载字体: {custom_font_name}")
    else:
        print(f"⚠️ 未找到字体文件: {FONT_PATH}，使用默认字体。")
        plt.rcParams['font.family'] = 'sans-serif'
except Exception as e:
    print(f"⚠️ 字体加载出错: {e}，使用默认字体。")

# 论文绘图标准
plt.rcParams['mathtext.fontset'] = 'stixsans'
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7     

targets = ["mu", "gap", "cv", "TPSA"]

# ==========================================
# 2. 数据读取 (带诊断功能的修改版)
# ==========================================
print(f"--- 正在检查数据路径: {DATA_ROOT} ---")

if not os.path.exists(DATA_ROOT):
    print(f"❌ 严重错误: 根目录不存在！\n   -> {DATA_ROOT}")
else:
    print(f"✅ 根目录存在。")

dfs = []
# 如果你的数据不在 fold_0 里，请修改这里的 range
for i in range(2): 
    folder = os.path.join(DATA_ROOT, f"fold_{i}")
    file_true = os.path.join(folder, "test_full.csv")
    file_pred = os.path.join(folder, "test_preds.csv")
    
    print(f"\n正在检查 Fold {i}...")
    print(f"  寻找真值文件: {file_true}")
    
    if os.path.exists(file_true):
        print("  -> ✅ 真值文件存在")
    else:
        print("  -> ❌ 真值文件缺失！(请检查文件名是否为 test_full.csv)")
        
    if os.path.exists(file_pred):
        print("  -> ✅ 预测文件存在")
    else:
        print("  -> ❌ 预测文件缺失！(请检查文件名是否为 test_preds.csv)")

    if os.path.exists(file_true) and os.path.exists(file_pred):
        y = pd.read_csv(file_true)
        p = pd.read_csv(file_pred)
        pred_cols = [c for c in p.columns if c.lower() not in ("smiles", "smiles_0")]
        p = p.rename(columns={c: f"{c}_pred" for c in pred_cols})
        df = pd.concat([y.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
        dfs.append(df)
        print(f"  -> ✅ 成功加载 {len(df)} 条数据")

if not dfs:
    print("\n" + "="*40)
    print("❌ 错误总结: 未能加载任何数据！")
    print("请根据上面的红色提示，修改 DATA_ROOT 路径或 CSV 文件名。")
    print("="*40 + "\n")
    # 这里不再 raise error，而是让程序优雅退出，方便你看打印信息
    exit()

val = pd.concat(dfs, ignore_index=True)
print(f"\n数据合并完成，总计 {len(val)} 条样本。")
# ==========================================
# 3. 绘图逻辑
# ==========================================

def get_stats(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

def get_limits(y_true, y_pred):
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    span = vmax - vmin
    # 留 5% 边距
    return vmin - span * 0.05, vmax + span * 0.05

def plot_individual_targets(val_df, target_list):
    for t in target_list:
        if f"{t}_pred" not in val_df.columns:
            continue
            
        y_true = val_df[t].astype(float).values
        y_pred = val_df[f"{t}_pred"].astype(float).values
        lo, hi = get_limits(y_true, y_pred)
        r2, mae, rmse = get_stats(y_true, y_pred)
        
        # 获取单位，如果非空则加个空格
        unit = TARGET_UNITS.get(t, "")
        unit_str = f" {unit}" if unit else ""

        # 1. 绘图
        g = sns.JointGrid(x=y_true, y=y_pred, height=3.5, ratio=5)
        g.plot_joint(plt.scatter, color='#4c72b0', s=15, alpha=0.7, 
                     edgecolor='white', linewidth=0.4)
        g.plot_marginals(sns.kdeplot, fill=True, color='#4c72b0', alpha=0.3, linewidth=0)
        g.ax_joint.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1.0, alpha=0.8)

        # 2. 轴标签 (带单位)
        label_suffix = f" ({unit})" if unit else ""
        g.ax_joint.set_xlabel(f"True {t}{label_suffix}")
        g.ax_joint.set_ylabel(f"Predicted {t}{label_suffix}")
        g.ax_joint.set_xlim(lo, hi)
        g.ax_joint.set_ylim(lo, hi)

        # 3. 统计框 (数值带单位)
        stats_text = (f"MAE: {mae:.4f}{unit_str}\n"
                      f"RMSE: {rmse:.4f}{unit_str}\n"
                      f"R$^2$: {r2:.4f}")
        
        props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.5, alpha=1.0)
        g.ax_joint.text(0.95, 0.05, stats_text, transform=g.ax_joint.transAxes,
                        ha='right', va='bottom', bbox=props)

        fname = f"{OUT_DIR}/parity_{t}.png"
        g.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

def plot_summary_grid(val_df, target_list):
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.ravel()
    
    for i, t in enumerate(target_list):
        ax = axes[i]
        if f"{t}_pred" not in val_df.columns:
            ax.axis('off')
            continue
            
        y_true = val_df[t].astype(float).values
        y_pred = val_df[f"{t}_pred"].astype(float).values
        lo, hi = get_limits(y_true, y_pred)
        r2, mae, rmse = get_stats(y_true, y_pred)
        unit = TARGET_UNITS.get(t, "")
        unit_str = f" {unit}" if unit else ""
        
        ax.scatter(y_true, y_pred, color='#4c72b0', s=15, alpha=0.7, 
                   edgecolor='white', linewidth=0.4)
        ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1.0, alpha=0.8)
        
        # 轴标签和标题带单位
        label_suffix = f" ({unit})" if unit else ""
        ax.set_title(t, fontweight='bold')
        ax.set_xlabel(f"True {t}{label_suffix}")
        ax.set_ylabel(f"Predicted {t}{label_suffix}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        
        # 统计框带单位
        stats_text = (f"MAE: {mae:.4f}{unit_str}\nRMSE: {rmse:.4f}{unit_str}\nR$^2$: {r2:.4f}")
        props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.5, alpha=1.0)
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
                ha='right', va='bottom', bbox=props, fontsize=6)

    plt.tight_layout()
    fname = f"{OUT_DIR}/parity_summary_grid.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Summary: {fname}")

if __name__ == "__main__":
    plot_individual_targets(val, targets)
    plot_summary_grid(val, targets)
    print("\n✅ 已生成带单位的图表！")