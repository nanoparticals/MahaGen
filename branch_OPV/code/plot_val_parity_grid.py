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
DATA_ROOT = "/public/home/users/haoxw/generate_AI/branch_OLED/ckpts/chemprop_DPER0_cv5_new_6"
OUT_DIR = "plots_final_paper_units"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 【新增】单位字典：请在这里修改你的实际单位 ---
# 如果某项没有单位（如 LogP），留空字符串 "" 即可
TARGET_UNITS = {
    "homo": "eV",
    "gap": "eV",
    "MolMR": "cm$^3$/mol",    # 摩尔折射率通常没单位或写起来太长，视情况而定
    "LogP": ""      # 对数值通常无单位
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

targets = ["homo", "gap", "MolMR", "LogP"]

# ==========================================
# 2. 数据读取
# ==========================================
dfs = []
for i in range(1): # 如需合并更多 fold，请改为 range(5)
    folder = f"{DATA_ROOT}/fold_{i}"
    file_true = f"{folder}/test_full.csv"
    file_pred = f"{folder}/test_preds.csv"
    
    if os.path.exists(file_true) and os.path.exists(file_pred):
        y = pd.read_csv(file_true)
        p = pd.read_csv(file_pred)
        pred_cols = [c for c in p.columns if c.lower() not in ("smiles", "smiles_0")]
        p = p.rename(columns={c: f"{c}_pred" for c in pred_cols})
        df = pd.concat([y.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
        dfs.append(df)

if not dfs:
    raise FileNotFoundError("未读取到数据，请检查路径。")

val = pd.concat(dfs, ignore_index=True)

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