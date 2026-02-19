#!/usr/bin/env python3
#python make_stats_json.py --data domain.csv --label-cols Qcal,Density --keep-header --out ckpts/et_stats.json --format dict2 --dropna,
#然后在 train_rl_multiobj.py 里：--predictor-stats ckpts/et_stats.json --zscore

import argparse, json, pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="包含SMILES与标签列的CSV")
ap.add_argument("--label-cols", required=True, help="逗号分隔列名，例如 Qcal,Density")
ap.add_argument("--keep-header", action="store_true", help="CSV首行为列名")
ap.add_argument("--out", required=True, help="输出到这个stats.json")
ap.add_argument("--format", choices=["dict2","dict1","flat","list"], default="dict2",
                help="输出格式：dict2=有means/stds两个子字典；dict1=每目标一个子字典；flat=后缀键；list=列表")
ap.add_argument("--dropna", action="store_true", help="对任一目标缺失的行整体丢弃")
args = ap.parse_args()

# 读取
if args.keep_header:
    df = pd.read_csv(args.data)
else:
    df = pd.read_csv(args.data, header=None)
targets = [t.strip() for t in args.label_cols.split(",") if t.strip()]

# 选择列（允许 SMILES 列名不是 'SMILES'，只要你写对 label-cols 即可）
sub = df[targets].copy()

# 缺失处理
if args.dropna:
    sub = sub.dropna(how="any")
else:
    # 逐列丢弃缺失，仅用该列非缺失样本计算均值/标准差
    pass

means = {t: float(sub[t].dropna().mean()) for t in targets}
stds  = {t: float(sub[t].dropna().std(ddof=0) or 1.0) for t in targets}

# 输出
fmt = args.format
if fmt == "dict2":
    out = {"means": means, "stds": stds}
elif fmt == "dict1":
    out = {t: {"mean": means[t], "std": stds[t]} for t in targets}
elif fmt == "flat":
    out = {}
    for t in targets:
        out[f"{t}_mean"] = means[t]; out[f"{t}_std"] = stds[t]
else:  # list
    out = [{"target": t, "mean": means[t], "std": stds[t]} for t in targets]

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Saved {args.out} with targets={targets}")
