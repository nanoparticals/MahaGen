import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import os

def plot_hybrid_regression(x, y, xlabel, ylabel, out_path):
    """
    绘制混合图：包含对角线(Reference)和拟合线(Trend)。
    """
    # 0. 数据清洗
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    # 1. 计算线性拟合 (Slope, Intercept)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # 2. 计算其他指标
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    r2 = r2_score(x, y) # 注意：这是针对 y=x 的 R2，还是针对拟合线的 R2？
    # 通常 R2_score 函数是计算预测值相对于真实值的解释度。
    # 这里我们直接用 stats.linregress 的 r_value**2 作为线性拟合的 R2
    fit_r2 = r_value**2

    # 3. 确定范围
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    span = vmax - vmin
    if span == 0: span = 1.0
    lo = vmin - span * 0.15
    hi = vmax + span * 0.15

    # --- 绘图 ---
    # 使用 JointGrid
    g = sns.JointGrid(x=x, y=y, height=4, ratio=5)

    # A. 绘制散点
    g.plot_joint(plt.scatter, color='#4c72b0', s=30, alpha=0.8, 
                 edgecolor='white', linewidth=0.5, label='Data Points')

    # B. 绘制拟合线 (红色实线)
    # 生成拟合线的 x 点
    x_fit = np.linspace(lo, hi, 100)
    y_fit = slope * x_fit + intercept
    g.ax_joint.plot(x_fit, y_fit, color='#d62728', linewidth=1.5, alpha=0.9, 
                    label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    
    # C. 绘制置信区间 (95% CI) - 可选，增加高级感
    sns.regplot(x=x, y=y, ax=g.ax_joint, scatter=False, color='#d62728', ci=95, truncate=False)

    # D. 绘制对角线 (灰色虚线 - Reference)
    g.ax_joint.plot([lo, hi], [lo, hi], linestyle='--', color='gray', linewidth=1.0, 
                    alpha=0.7, label='Ideal (y=x)')

    # E. 边缘密度图
    try:
        g.plot_marginals(sns.kdeplot, fill=True, color='#4c72b0', alpha=0.2)
    except:
        g.plot_marginals(sns.histplot, color='#4c72b0', alpha=0.3)

    # F. 标签与范围
    g.ax_joint.set_xlabel(xlabel, fontsize=10)
    g.ax_joint.set_ylabel(ylabel, fontsize=10)
    g.ax_joint.set_xlim(lo, hi)
    g.ax_joint.set_ylim(lo, hi)
    
    # G. 统计信息框 (增强版)
    stats_text = (f"Count = {len(x)}\n"
                  f"MAE = {mae:.3f} Ha\n"
                  f"RMSE = {rmse:.3f} Ha\n"
                  f"Fit $R^2$ = {fit_r2:.3f}\n"
                  f"Slope = {slope:.2f}\n"
                  f"Bias = {intercept:.2f}")
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9)
    g.ax_joint.text(0.05, 0.95, stats_text, transform=g.ax_joint.transAxes,
                    ha='left', va='top', bbox=props, fontsize=8, linespacing=1.5)

    # H. 图例 (放在右下角)
    g.ax_joint.legend(loc='lower right', fontsize=8, frameon=True)

    # 保存
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

# --- 运行 ---
if __name__ == "__main__":
    df = pd.read_excel("/public/home/users/haoxw/generate_AI/branch_Li/DFT/DFT.xlsx")
    x_data = df.iloc[:, 1] # DFT
    y_data = df.iloc[:, 2] # Model
    
    # 自动获取单位
    xlabel = "DFT Calculated (Ha)"
    ylabel = "Model Predicted (Ha)"
    
    plot_hybrid_regression(x_data, y_data, xlabel, ylabel, "regression_plot.png")