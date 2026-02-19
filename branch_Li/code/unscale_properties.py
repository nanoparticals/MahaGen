#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例：
python /public/home/users/haoxw/generate_AI/branch_Li/code/unscale_properties.py \
  --csv  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_2/outputs/all_molecules.csv \
  --stats /public/home/users/haoxw/generate_AI/branch_Li/ckpts/li_predictor_stats.json \
  --out  /public/home/users/haoxw/generate_AI/branch_Li/ckpts/gpu_rl_vgae_soft_congen_2/outputs/all_molecules_physical.csv

默认假设 CSV 里列名为：
  pred_mu, pred_gap, pred_cv, pred_TPSA
默认假设 stats JSON 的键名为：
  mu, gap, cv, TPSA
"""

import json
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="来自分析脚本的输出 CSV（包含 z-score 预测列）")
    ap.add_argument("--stats", required=True, help="stats.json 路径（包含 means/stds）")
    ap.add_argument("--out", required=True, help="写出新的 CSV")

    # CSV 里的预测列名（z-score）
    ap.add_argument("--col-mu",   default="pred_mu")
    ap.add_argument("--col-gap",  default="pred_gap")
    ap.add_argument("--col-cv",   default="pred_cv")
    ap.add_argument("--col-TPSA", default="pred_TPSA")

    # stats JSON 里的键名（means/stds 的 key）
    ap.add_argument("--key-mu",   default="mu")
    ap.add_argument("--key-gap",  default="gap")
    ap.add_argument("--key-cv",   default="cv")
    ap.add_argument("--key-TPSA", default="TPSA")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    with open(args.stats, "r") as f:
        stats = json.load(f)

    means = stats["means"]
    stds  = stats["stds"]

    def unz(colname: str, key: str) -> pd.Series:
        if colname not in df.columns:
            raise KeyError(f"CSV 中找不到列: {colname}")
        if key not in means or key not in stds:
            raise KeyError(f"stats JSON 中找不到 key: {key}（需要同时存在于 means/stds）")
        z = pd.to_numeric(df[colname], errors="coerce")
        return z * float(stds[key]) + float(means[key])

    # 输出“物理尺度”列（列名你也可以按习惯改）
    df["mu_phys"]   = unz(args.col_mu,   args.key_mu)
    df["gap_phys"]  = unz(args.col_gap,  args.key_gap)
    df["cv_phys"]   = unz(args.col_cv,   args.key_cv)
    df["TPSA_phys"] = unz(args.col_TPSA, args.key_TPSA)

    df.to_csv(args.out, index=False)
    print("[OK] wrote:", args.out)

if __name__ == "__main__":
    main()
