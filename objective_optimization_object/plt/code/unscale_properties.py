#!/usr/bin/env python3
'''python unscale_properties.py   --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_22/outputs/all_molecules.csv   --stats /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/et_stats_DPER0.json   --out /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_22/outputs/all_molecules_physical.csv'''
import json, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="来自 analyze_selfies_sa_reward.py 的输出 CSV")
    ap.add_argument("--stats", required=True, help="et_stats_DPER0.json 路径")
    ap.add_argument("--out", required=True, help="写出新的 CSV")
    # 这些列名按你的脚本默认：pred_D, pred_P, pred_EG, pred_r0
    ap.add_argument("--col-D", default="pred_D")
    ap.add_argument("--col-P", default="pred_P")
    ap.add_argument("--col-EG", default="pred_EG")
    ap.add_argument("--col-r0", default="pred_r0")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    with open(args.stats, "r") as f:
        stats = json.load(f)
    means, stds = stats["means"], stats["stds"]

    def unz(col, mean, std):  # z -> physical
        return df[col].astype(float) * float(stds[mean]) + float(means[mean])

    # 注意：JSON里键名就是 "D","P","EG","r0"
    df["D_phys_km_s"]   = df[args.col_D].astype(float)  * stds["D"]  + means["D"]
    df["P_phys_kbar"]   = df[args.col_P].astype(float)  * stds["P"]  + means["P"]
    df["EG_phys_m_s"]   = df[args.col_EG].astype(float) * stds["EG"] + means["EG"]
    df["r0_phys_g_cm3"] = df[args.col_r0].astype(float) * stds["r0"] + means["r0"]

    df.to_csv(args.out, index=False)
    print("[OK] wrote:", args.out)

if __name__ == "__main__":
    main()
