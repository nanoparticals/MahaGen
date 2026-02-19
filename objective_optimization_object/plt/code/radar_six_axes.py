#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
六轴雷达图（四个性质 + SA + Reward）
- 先筛选 SA < 阈值（默认 5）
- 中心分子图强制透明（白底抠透明）
- 右侧打印原始值（含单位）
- 支持单位 (--units)
- 支持手动量纲范围、log 轴、三种归一化
- 支持 --prop-cols 映射到任意列名（如 D_phys_km_s 等）
- 可选 --unzscore-stats 对所选列做 z-score 反标准化（通常物理列无需）
- === 新增/修改：去重 ===
  * --dedup 打开去重
  * --dedup-by {canonical_smiles,smiles,inchi,inchikey}
  * --dedup-keep {max_reward,min_sa,max_reward_then_min_sa,first,last}
  * --dedup-stage {before,after} 选择去重发生在 SA 筛选前/后
  python radar_six_axes.py   --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin/outputs/all_molecules_physical.csv  --prop-cols D D_phys_km_s P P_phys_kbar EG EG_phys_m_s r0 r0_phys_g_cm3  --manual-range D 7 10 P 200 400 EG 6000 10000 r0 1.0 1.9 SA 1.5 8 Reward 0 3   --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin/outputs --units D g/cm³ P GPa EG eV r0 % SA score Reward a.u. --dedup
  python radar_six_axes.py   --csv /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin/outputs/all_molecules_physical.csv  --prop-cols D D_phys_km_s P P_phys_kbar EG EG_phys_m_s r0 r0_phys_g_cm3  --manual-range D 8 12 P 400 650 EG 10000 15200 r0 1.0 1.6 SA 2.5 8 Reward 0 6   --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_18/outputs --units D km/s P kbar EG J/g r0 g/cm3 SA score Reward a.u. --dedup
"""

from __future__ import annotations
import os
import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from rdkit import Chem
from rdkit.Chem import Draw

# --- RDKit InChI 可选支持 ---
try:
    from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
    _HAS_INCHI = True
except Exception:
    _HAS_INCHI = False


# ---------- helpers ----------

def mol_image_rgba(
    mol: Chem.Mol,
    size=(300, 225),
    white_threshold: int = 250,
) -> "PIL.Image.Image":
    """生成 RDKit 分子图并强制透明。"""
    img = Draw.MolToImage(
        mol,
        size=size,
        kekulize=True,
        wedgeBonds=True,
        bgColor=(255, 255, 255),
    )
    try:
        from PIL import Image
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = np.array(img)
        r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        mask = (r >= white_threshold) & (g >= white_threshold) & (b >= white_threshold)
        arr[mask, 3] = 0
        img = Image.fromarray(arr)
    except Exception:
        pass
    return img


def safe_log(arr: np.ndarray, eps: float = 1e-6):
    """对<=0的轴自动平移后取 log。"""
    a = arr.astype(float)
    finite = np.isfinite(a)
    if not finite.any():
        return a
    amin = np.nanmin(a[finite])
    shift = -amin + eps if amin <= 0 else 0.0
    return np.log(a + shift + eps)


def compute_norm_values(values_raw: Dict[str, float],
                        goals: Dict[str, str],
                        ref_stats: Dict[str, Dict[str, float]],
                        norm: str = "percentile",
                        zclip: float = 2.0,
                        gamma: float = 1.0) -> List[float]:
    """三种归一化并按目标方向翻转。"""
    out = []
    for k in values_raw.keys():
        v = float(values_raw.get(k, np.nan))
        g = goals.get(k, "max").lower()
        s = ref_stats[k]
        if norm == "percentile":
            lo, hi = s["qlo"], s["qhi"]
            if not np.isfinite(v) or hi <= lo:
                val = 0.5
            else:
                val = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
        elif norm == "zscore":
            mu, sd = s["mean"], s["std"]
            if not np.isfinite(v) or sd <= 1e-12:
                val = 0.5
            else:
                z = float(np.clip((v - mu) / sd, -zclip, zclip))
                val = (z + zclip) / (2 * zclip)
        else:  # minmax
            lo, hi = s["lo"], s["hi"]
            if not np.isfinite(v) or hi <= lo:
                val = 0.5
            else:
                val = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

        if g == "min":
            val = 1.0 - val

        if np.isfinite(val) and gamma != 1.0 and val >= 0.0:
            val = float(np.clip(val, 0.0, 1.0)) ** float(gamma)
        elif not np.isfinite(val):
            val = 0.5
        out.append(val)
    return out


# ---------- 去重辅助（新增） ----------

def _make_key_from_mol(mol: Optional[Chem.Mol], method: str, raw_smi: str) -> Optional[str]:
    """根据指定方法生成去重键。失败时返回 None。"""
    if mol is None:
        return None
    method = method.lower()
    try:
        if method in ("canonical_smiles", "smiles"):
            # canonical=True 生成规范 SMILES；isomeric 保留立体信息
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        elif method == "inchi":
            if not _HAS_INCHI:
                return None
            return MolToInchi(mol)
        elif method == "inchikey":
            if not _HAS_INCHI:
                return None
            return MolToInchiKey(mol)
    except Exception:
        return None
    return None


def deduplicate_df(df: pd.DataFrame,
                   smiles_col: str,
                   reward_col: str,
                   sa_col: str,
                   dedup_by: str = "canonical_smiles",
                   keep: str = "max_reward") -> pd.DataFrame:
    """
    对 DataFrame 进行分子去重。
    - dedup_by: 选择去重键类型
    - keep:
        * max_reward: 奖励最高者保留
        * min_sa: SA 最小者保留
        * max_reward_then_min_sa: 先按奖励降序、再按 SA 升序
        * first/last: pandas 语义
    """
    df = df.copy()
    # 生成键
    keys: List[Optional[str]] = []
    for smi in df[smiles_col].astype(str).tolist():
        mol = Chem.MolFromSmiles(smi)
        key = _make_key_from_mol(mol, dedup_by, smi)
        # 如果指定方法失败，回退到 canonical SMILES；再失败就置 None
        if key is None and mol is not None:
            try:
                key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            except Exception:
                key = None
        keys.append(key)

    df["_dedup_key"] = keys
    n_all = len(df)
    n_key_na = df["_dedup_key"].isna().sum()

    # 对于 key 为 None 的行，使用原始 SMILES 兜底，避免把它们全部视作同一个键
    if n_key_na > 0:
        df.loc[df["_dedup_key"].isna(), "_dedup_key"] = (
            "RAWSMI::" + df.loc[df["_dedup_key"].isna(), smiles_col].astype(str)
        )

    # 排序以决定保留策略
    keep = keep.lower()
    if keep == "max_reward":
        df = df.sort_values(reward_col, ascending=False)
        df_dedup = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    elif keep == "min_sa":
        df = df.sort_values(sa_col, ascending=True)
        df_dedup = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    elif keep == "max_reward_then_min_sa":
        df = df.sort_values([reward_col, sa_col], ascending=[False, True])
        df_dedup = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    elif keep in ("first", "last"):
        df_dedup = df.drop_duplicates(subset=["_dedup_key"], keep=keep)
    else:
        raise ValueError(f"未知 keep 策略：{keep}")

    n_after = len(df_dedup)
    print(f"[dedup] 方法={dedup_by}, 策略={keep}: {n_all} -> {n_after}（去除 {n_all - n_after} 个重复）")
    df_dedup = df_dedup.drop(columns=["_dedup_key"])
    return df_dedup


# ---------- radar plot ----------

def radar_chart(values_proc: Dict[str, float],
                values_raw: Dict[str, float],
                goals: Dict[str, str],
                ref_stats: Dict[str, Dict[str, float]],
                title: str,
                center_img_rgba,
                out_png: str,
                units: Dict[str, str] | None = None,
                norm: str = "percentile",
                zclip: float = 2.0,
                gamma: float = 1.0,
                center_alpha: float = 0.55,
                transparent_fig: bool = False) -> None:

    labels = list(values_proc.keys())
    norm_vals = compute_norm_values(values_proc, goals, ref_stats, norm=norm, zclip=zclip, gamma=gamma)

    # 轴标签
    label_texts = []
    for l in labels:
        if units and l in units:
            label_texts.append(f"{l}\n({units[l]})")
        else:
            label_texts.append(l)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals = norm_vals + norm_vals[:1]
    angs = angles + angles[:1]

    fig = plt.figure(figsize=(6.2, 4.6), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    ax = fig.add_subplot(gs[0, 0], polar=True)
    ax.set_facecolor((1, 1, 1, 0) if transparent_fig else "white")
    ax.grid(True, color="#C3C3C3", linewidth=0.6, alpha=0.9, zorder=0.1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(label_texts, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, pad=10)

    # 中心分子图
    if center_img_rgba is not None:
        try:
            imagebox = OffsetImage(center_img_rgba, zoom=0.42)
            imagebox.set_alpha(float(np.clip(center_alpha, 0.0, 1.0)))
            ab = AnnotationBbox(imagebox, (0.0, 0.0), frameon=False, zorder=0.2)
            ax.add_artist(ab)
        except Exception:
            pass

    # 面与边
    ax.fill(angs, vals, alpha=0.25, color="#6e6e6e", zorder=0.85)
    ax.plot(angs, vals, linewidth=1.3, color="black", zorder=0.9)

    # 右侧文字
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.axis("off")
    lines = []
    for k in labels:
        v = values_raw.get(k, float("nan"))
        unit = f" {units[k]}" if units and k in units else ""
        if isinstance(v, float) and np.isfinite(v):
            lines.append(f"{k}: {v:.3g}{unit}")
        else:
            lines.append(f"{k}: n/a")
    ax_txt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", transparent=transparent_fig)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--unzscore-stats", help="JSON 文件（含 means/stds），对所选性质列做 Z-score 反标准化", default=None)

    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--reward-col", default="reward")
    ap.add_argument("--sa-col", default="SA")
    ap.add_argument("--sa-threshold", type=float, default=5.0, help="仅保留 SA 小于该阈值的样本（默认 5.0）")

    # 目标性质名（雷达轴标签会用它们的名字）
    ap.add_argument("--targets", nargs="+", default=["D", "P", "EG", "r0"])
    ap.add_argument("--goals", nargs="+", default=["max", "max", "max", "max"])
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--use-all-range", action="store_true")

    # 归一化与数值处理
    ap.add_argument("--norm", choices=["percentile", "zscore", "minmax"], default="percentile")
    ap.add_argument("--quantiles", nargs=2, type=float, default=[5.0, 95.0])
    ap.add_argument("--zclip", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--log-axes", nargs="*", default=[])
    ap.add_argument("--manual-range", nargs="*", default=[])

    # 指标→列名映射
    ap.add_argument("--prop-cols", nargs="*", default=[],
                    help="把 D/P/EG/r0 映射到 CSV 的实际列名，如: --prop-cols D D_phys_km_s P P_phys_kbar EG EG_phys_m_s r0 r0_phys_g_cm3")

    # 单位
    ap.add_argument("--units", nargs="*", default=[],
                    help="为各指标指定单位，如: --units D km/s P kbar EG m/s r0 g/cm³ SA score Reward a.u.")

    # 中心分子透明抠图阈值
    ap.add_argument("--center-white-threshold", type=int, default=250,
                    help="将接近白色的像素抠透明的阈值(0-255)，默认250")

    ap.add_argument("--center-alpha", type=float, default=0.55)
    ap.add_argument("--transparent-figure", action="store_true")
    ap.add_argument("--outdir", required=True)

    # === 新增/修改：去重 ===
    ap.add_argument("--dedup", action="store_true", help="对分子去重")
    ap.add_argument("--dedup-by", choices=["canonical_smiles", "smiles", "inchi", "inchikey"],
                    default="canonical_smiles", help="去重键类型（默认 canonical_smiles）")
    ap.add_argument("--dedup-keep",
                    choices=["max_reward", "min_sa", "max_reward_then_min_sa", "first", "last"],
                    default="max_reward", help="去重时的保留策略（默认 max_reward）")
    ap.add_argument("--dedup-stage", choices=["before", "after"], default="after",
                    help="去重阶段：before=在 SA 筛选前；after=在 SA<阈值后（默认）")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    radar_dir = os.path.join(args.outdir, "radars")
    os.makedirs(radar_dir, exist_ok=True)

    # --- 单位解析 ---
    units = {}
    if args.units:
        if len(args.units) % 2 != 0:
            raise ValueError("单位参数必须成对出现，例如: D km/s P kbar EG m/s r0 g/cm³ ...")
        for i in range(0, len(args.units), 2):
            units[args.units[i]] = args.units[i + 1]

    # --- 指标→列名映射 ---
    prop_cols: Dict[str, str] = {t: f"pred_{t}" for t in args.targets}
    if args.prop_cols:
        if len(args.prop_cols) % 2 != 0:
            raise ValueError("prop-cols 参数必须成对出现，例如: D D_phys_km_s P P_phys_kbar EG EG_phys_m_s r0 r0_phys_g_cm3")
        for i in range(0, len(args.prop_cols), 2):
            key = args.prop_cols[i]
            col = args.prop_cols[i + 1]
            prop_cols[key] = col

    # --- 数据读取 ---
    df = pd.read_csv(args.csv)
    # 检查必要列
    needed = [args.smiles_col, args.reward_col, args.sa_col] + [prop_cols[t] for t in args.targets]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少列：{miss}\n现有列：{list(df.columns)}")

    # --- 可选：反标准化（如果你传了 z-score 列且想还原为物理量）
    if args.unzscore_stats:
        with open(args.unzscore_stats) as f:
            stats = json.load(f)
        means, stds = stats.get("means", {}), stats.get("stds", {})
        for t in args.targets:
            col = prop_cols[t]
            if col in df.columns and (t in means) and (t in stds):
                df[col] = df[col].astype(float) * float(stds[t]) + float(means[t])
                print(f"[unzscore] {col} = z * {stds[t]:.6g} + {means[t]:.6g}")

    # --- 去重阶段（before） ---
    if args.dedup and args.dedup_stage == "before":
        df = deduplicate_df(df, args.smiles_col, args.reward_col, args.sa_col,
                            dedup_by=args.dedup_by, keep=args.dedup_keep)

    # --- 筛选 SA < 阈值 ---
    before = len(df)
    df = df[df[args.sa_col].astype(float) < args.sa_threshold].copy()
    print(f"[filter] SA < {args.sa_threshold}: {before} -> {len(df)}")

    # --- 去重阶段（after，默认） ---
    if args.dedup and args.dedup_stage == "after":
        df = deduplicate_df(df, args.smiles_col, args.reward_col, args.sa_col,
                            dedup_by=args.dedup_by, keep=args.dedup_keep)

    # --- 排序 & TopN ---
    df_sorted = df.sort_values(args.reward_col, ascending=False).reset_index(drop=True)
    df_top = df_sorted.head(args.topn).copy()
    ref_df = df_sorted if args.use_all_range else df_top

    # --- 参考统计（用于归一化）---
    goals = {t: g.lower() for t, g in zip(args.targets, args.goals)}
    goals["SA"] = "min"
    goals["Reward"] = "max"

    qlo, qhi = float(args.quantiles[0]), float(args.quantiles[1])
    ref_stats: Dict[str, Dict[str, float]] = {}

    # 四个性质
    for t in args.targets:
        col = prop_cols[t]
        vals = ref_df[col].astype(float).values
        if t in args.log_axes:
            vals = safe_log(vals)
        ref_stats[t] = {
            "qlo": float(np.nanpercentile(vals, qlo)),
            "qhi": float(np.nanpercentile(vals, qhi)),
            "lo":  float(np.nanmin(vals)),
            "hi":  float(np.nanmax(vals)),
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals) + 1e-12),
        }

    # SA / Reward
    for special, colname in [("SA", args.sa_col), ("Reward", args.reward_col)]:
        vals = ref_df[colname].astype(float).values
        if special in args.log_axes:
            vals = safe_log(vals)
        ref_stats[special] = {
            "qlo": float(np.nanpercentile(vals, qlo)),
            "qhi": float(np.nanpercentile(vals, qhi)),
            "lo":  float(np.nanmin(vals)),
            "hi":  float(np.nanmax(vals)),
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals) + 1e-12),
        }

    # --- 手动范围覆盖 ---
    if args.manual_range:
        if len(args.manual_range) % 3 != 0:
            raise ValueError("manual-range 参数格式错误：必须成组出现，如 D 6 10 P 200 400 ...")
        for i in range(0, len(args.manual_range), 3):
            key = args.manual_range[i]
            lo = float(args.manual_range[i + 1])
            hi = float(args.manual_range[i + 2])
            if key in ref_stats:
                ref_stats[key]["lo"] = lo
                ref_stats[key]["hi"] = hi
                ref_stats[key]["qlo"] = lo
                ref_stats[key]["qhi"] = hi
                print(f"[manual-range] {key}: ({lo}, {hi})")

    # --- 绘图 ---
    for i, row in df_top.iterrows():
        smi = str(row[args.smiles_col])
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mol_img = mol_image_rgba(m, size=(300, 225), white_threshold=args.center_white_threshold)

        # 原始显示值：来自映射列
        raw_vals: Dict[str, float] = {t: float(row.get(prop_cols[t], np.nan)) for t in args.targets}
        raw_vals["SA"] = float(row[args.sa_col])
        raw_vals["Reward"] = float(row[args.reward_col])

        # 归一化前的处理（可选 log）
        proc_vals: Dict[str, float] = {}
        for k, v in raw_vals.items():
            proc_vals[k] = float(safe_log(np.array([v]))[0]) if k in args.log_axes else v

        title = f"Top{i+1}: Reward={raw_vals['Reward']:.3f}, SA={raw_vals['SA']:.2f}"
        out_png = os.path.join(radar_dir, f"radar_top_{i+1:02d}.png")
        radar_chart(proc_vals, raw_vals, goals, ref_stats, title, mol_img, out_png,
                    units=units, norm=args.norm, zclip=args.zclip, gamma=args.gamma,
                    center_alpha=args.center_alpha, transparent_fig=args.transparent_figure)

    print(f"[OK] 已保存 {len(df_top)} 张雷达图到：{radar_dir}")


if __name__ == "__main__":
    main()
