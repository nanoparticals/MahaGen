#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot generator training loss from a CSV log.

Expect columns:
  sample_step, avg_loss, time, lr, cond

Usage:
  python plt_gen_loss.py \
    --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gen_selfies_token_h256_acc64_cond_origin/train_gen_log.csv \
    --out gen_loss.png \
    --rolling 25
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def smooth_ema(x, alpha=0.1):
    """Exponential moving average (optional auxiliary smoother)."""
    y = np.empty_like(x, dtype=float)
    acc = 0.0
    for i, v in enumerate(x):
        acc = v if i == 0 else alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="日志CSV路径")
    ap.add_argument("--out", default="gen_loss.png", help="输出图文件 (png/pdf/svg)")
    ap.add_argument("--rolling", type=int, default=0, help="滑动平均窗口大小(>0启用)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA平滑系数(0~1，0关闭；如0.1)")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="y轴范围，例如 --ylim 2.5 4.5")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=None, engine="python", encoding="utf-8-sig")
    for c in ["sample_step", "avg_loss"]:
        if c not in df.columns:
            raise ValueError(f"缺少必要列 {c}，现有列：{list(df.columns)}")

    # x轴：训练步；y轴：平均loss
    x = df["sample_step"].to_numpy()
    y = df["avg_loss"].to_numpy()

    # 画图
    fig, ax1 = plt.subplots(figsize=(9, 5.4), dpi=150)

    # 原始曲线
    ax1.plot(x, y, lw=1.2, alpha=0.6, label="avg_loss (raw)")

    # 滑动平均
    if args.rolling and args.rolling > 1:
        y_roll = pd.Series(y).rolling(window=args.rolling, min_periods=1).mean().to_numpy()
        ax1.plot(x, y_roll, lw=2.2, label=f"avg_loss (rolling={args.rolling})")

    # EMA 平滑（可选，与 rolling 可共存）
    if 0.0 < args.ema < 1.0:
        y_ema = smooth_ema(y, alpha=args.ema)
        ax1.plot(x, y_ema, lw=2.0, label=f"avg_loss (EMA α={args.ema})")

    ax1.set_xlabel("Training step (sample_step)")
    ax1.set_ylabel("Average loss")
    if args.ylim:
        ax1.set_ylim(args.ylim)
    ax1.grid(alpha=0.25, linestyle="--")

    

    plt.title("Generator training loss")
    plt.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"Saved to {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
