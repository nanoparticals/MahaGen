#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot VGAE training losses from a CSV log.

Expected columns (常见于你的日志):
  epoch, loss, recon, kl, atom_ce, prop_mse, kl_w

Usage:
  python plot_vgae_loss.py \
    --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/vgae_struct_only_z16_log.csv \
    --out vgae_loss.png \
    --rolling 16
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="训练日志 CSV 路径")
    ap.add_argument("--out", default="vgae_loss.png", help="输出图片路径 (png/pdf/svg)")
    ap.add_argument("--rolling", type=int, default=0, help="滑动平均窗口(>0启用)")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="y 轴范围, 例如 --ylim 0 10")
    args = ap.parse_args()

    # 读取
    df = pd.read_csv(args.csv, sep=None, engine="python", encoding="utf-8-sig")

    # 必要列检查
    need_cols = ["epoch", "loss"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}，现有列: {list(df.columns)}")

    # 可选列存在则绘制
    has_recon = "recon" in df.columns
    has_kl    = "kl"    in df.columns
    has_klw   = "kl_w"  in df.columns

    # 滑动平均
    def roll(x):
        w = args.rolling
        if w and w > 1:
            return pd.Series(x).rolling(window=w, min_periods=1).mean().values
        return x

    epochs = df["epoch"].values
    loss   = df["loss"].values
    recon  = df["recon"].values if has_recon else None
    kl     = df["kl"].values if has_kl else None

    # 画图
    plt.figure(figsize=(9, 5.4), dpi=150)

    # 原始曲线
    plt.plot(epochs, loss,  color="#74a9cf", lw=1.2, alpha=0.7, label="loss (raw)")
    if has_recon:
        plt.plot(epochs, recon, color="#fd8d3c", lw=1.2, alpha=0.7, label="recon (raw)")
    if has_kl:
        plt.plot(epochs, kl,    color="#31a354", lw=1.2, alpha=0.7, label="KL (raw)")

    # 平滑曲线
    if args.rolling and args.rolling > 1:
        plt.plot(epochs, roll(loss),  color="#2b8cbe", lw=2.2, label=f"loss (rolling {args.rolling})")
        if has_recon:
            plt.plot(epochs, roll(recon), color="#e6550d", lw=2.0, label=f"recon (rolling {args.rolling})")
        if has_kl:
            plt.plot(epochs, roll(kl),    color="#238b45", lw=2.0, label=f"KL (rolling {args.rolling})")

    # KL 权重二轴（可选）
    if has_klw:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(epochs, df["kl_w"].values, color="#756bb1", lw=1.5, alpha=0.9, label="KL weight")
        ax2.set_ylabel("KL weight", color="#756bb1")
        ax2.tick_params(axis="y", labelcolor="#756bb1")
        # 合并图例
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    else:
        plt.legend(loc="upper right")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if args.ylim:
        plt.ylim(args.ylim)
    plt.title("VGAE training loss")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    # 保存
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    print(f"Saved to {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
