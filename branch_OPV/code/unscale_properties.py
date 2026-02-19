#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unscale (z-score -> physical) for columns pred_<target> using stats JSON (means/stds).

Example:
python unscale_properties.py \
  --csv /path/to/all_molecules.csv \
  --stats /path/to/OPV_predictor_stats.json \
  --out /path/to/all_molecules_physical.csv \
  --targets homo gap MolMR LogP
python unscale_properties.py   --csv /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_3/outputs/all_molecules.csv   --stats /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/OPV_predictor_stats.json   --out /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_3/outputs/all_molecules_physical.csv
"""
import json
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd


def _load_stats(path: str) -> tuple[Dict[str, float], Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        stats: Any = json.load(f)

    # 支持多种 stats 格式：{"means":..., "stds":...} 或 {t:{mean,std}} 或 {t_mean,t_std}
    if isinstance(stats, dict) and "means" in stats and "stds" in stats:
        means = {k: float(v) for k, v in stats["means"].items()}
        stds = {k: float(v) for k, v in stats["stds"].items()}
        return means, stds

    if isinstance(stats, dict) and all(isinstance(v, dict) for v in stats.values()):
        means = {t: float(stats[t].get("mean", 0.0)) for t in stats.keys()}
        stds = {t: float(stats[t].get("std", 1.0)) for t in stats.keys()}
        return means, stds

    if isinstance(stats, dict) and any(k.endswith("_mean") for k in stats.keys()):
        targets = sorted({k[:-5] for k in stats.keys() if k.endswith("_mean")})
        means = {t: float(stats.get(f"{t}_mean", 0.0)) for t in targets}
        stds = {t: float(stats.get(f"{t}_std", 1.0)) for t in targets}
        return means, stds

    raise ValueError("stats JSON 结构不被识别：需要 (means,stds) 或 per-target mean/std。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="来自 analyze_selfies_sa_reward.py 的输出 CSV")
    ap.add_argument("--stats", required=True, help="stats JSON 路径（包含 means/stds）")
    ap.add_argument("--out", required=True, help="写出新的 CSV")
    ap.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="只反标定这些 targets（例如 homo gap MolMR LogP）。不提供则自动处理 stats 中所有且 CSV 里存在 pred_<t> 的列。",
    )
    ap.add_argument(
        "--pred-prefix",
        default="pred_",
        help="预测列前缀，默认 pred_（即 pred_<target>）",
    )
    ap.add_argument(
        "--out-prefix",
        default="phys_",
        help="输出物理值列前缀，默认 phys_（即 phys_<target>）",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出列已存在，允许覆盖（默认遇到同名列会跳过）。",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    means, stds = _load_stats(args.stats)

    # 选择要处理的 targets
    if args.targets and len(args.targets) > 0:
        targets: List[str] = list(args.targets)
    else:
        targets = list(means.keys())

    done = 0
    skipped = 0
    missing = 0

    for t in targets:
        pred_col = f"{args.pred_prefix}{t}"
        out_col = f"{args.out_prefix}{t}"

        if pred_col not in df.columns:
            missing += 1
            continue
        if (out_col in df.columns) and (not args.overwrite):
            skipped += 1
            continue

        m = float(means.get(t, 0.0))
        s = float(stds.get(t, 1.0)) if float(stds.get(t, 1.0)) != 0.0 else 1.0

        # z -> physical: x = z*std + mean
        df[out_col] = df[pred_col].astype(float) * s + m
        done += 1

    df.to_csv(args.out, index=False)
    print(f"[OK] wrote: {args.out}")
    print(f"[INFO] unscaled columns: {done}, skipped(existing): {skipped}, missing(pred_col): {missing}")


if __name__ == "__main__":
    main()
