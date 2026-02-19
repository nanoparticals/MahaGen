#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
画“图 a”式的强化学习训练曲线：
- 自动识别 step 列与 score 列（也支持手动指定）
- 画原始曲线 + 滚动均值
- 如存在标准差列（或你手动指定），会绘制 ±1σ 灰色置信带
python plt_a.py --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_5/rl_log.csv --step-col step --score-col avg_reward --out /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_5/plt/figure_a_training_curve.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def auto_pick_column(df, candidates):
    """按候选关键词在列名中匹配，优先精确等于，其次包含关系。"""
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}

    # 精确等于
    for key in candidates:
        for c in cols:
            if low[c] == key:
                return c

    # 包含关系
    for key in candidates:
        for c in cols:
            if key in low[c]:
                return c
    return None


def main(args):
    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV 为空。")

    # 自动/手动选择列
    step_col = args.step_col
    score_col = args.score_col
    std_col = args.std_col

    if step_col is None:
        step_col = auto_pick_column(
            df,
            ["step", "steps", "rl_step", "global_step", "iter", "iteration", "epoch"],
        )
    if step_col is None:
        df["_auto_step"] = np.arange(len(df))
        step_col = "_auto_step"

    if score_col is None:
        score_col = auto_pick_column(
            df,
            ["score", "reward", "avg_reward", "mean_reward", "objective", "total_reward", "mean_score"],
        )
    if score_col is None:
        # 兜底：任取一个数值列（非 step）
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != step_col]
        if not numeric_cols:
            raise SystemExit("未找到可用的分数列（score/reward 等）且数据集中没有其它数值列。")
        score_col = numeric_cols[0]

    if std_col is None:
        std_col = auto_pick_column(
            df,
            ["std", "stdev", "score_std", "reward_std", "stddev"]
        )

    # 排序并取值
    df = df.sort_values(step_col).reset_index(drop=True)
    x = df[step_col].to_numpy()
    y = df[score_col].to_numpy()

    # 平滑
    if args.window is None:
        window = max(5, len(y) // 50)  # 自适应小窗口
    else:
        window = max(1, int(args.window))

    y_smooth = pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()

    # 绘图
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(x, y, linewidth=1, alpha=0.45, label="raw")
    plt.plot(x, y_smooth, linewidth=2, label=f"rolling mean (w={window})")

    # 置信带（若有）
    if std_col is not None and std_col in df.columns:
        y_std = df[std_col].to_numpy()
        y_upper = y_smooth + y_std
        y_lower = y_smooth - y_std
        plt.fill_between(x, y_lower, y_upper, alpha=0.2, color="gray", label="±1σ (on smoothed)")

    plt.xlabel("Reinforcement learning step")
    plt.ylabel("Score")
    plt.title("Training curve (panel a)")
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved figure to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot panel (a)-style RL training curve.")
    parser.add_argument("--csv", type=str, default="rl_log.csv", help="日志 CSV 路径")
    parser.add_argument("--step-col", type=str, default=None, help="步数列名（可选）")
    parser.add_argument("--score-col", type=str, default=None, help="分数/奖励列名（可选）")
    parser.add_argument("--std-col", type=str, default=None, help="标准差列名（可选，用于置信带）")
    parser.add_argument("--window", type=int, default=None, help="滚动均值窗口大小（默认自适应）")
    parser.add_argument("--out", type=str, default="figure_a_training_curve.png", help="输出图片路径")
    args = parser.parse_args()
    main(args)
