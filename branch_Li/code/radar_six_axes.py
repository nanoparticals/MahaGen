#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
六轴雷达图（四个性质 + SA + Reward）
- 自动根据 --prop-cols 推断 targets（无需重复指定）
- 支持去重、单位、手动范围、多种归一化
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
        if k not in ref_stats:
             out.append(0.5)
             continue
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
    """对 DataFrame 进行分子去重。"""
    df = df.copy()
    keys: List[Optional[str]] = []
    for smi in df[smiles_col].astype(str).tolist():
        mol = Chem.MolFromSmiles(smi)
        key = _make_key_from_mol(mol, dedup_by, smi)
        if key is None and mol is not None:
            try:
                key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            except Exception:
                key = None
        keys.append(key)

    df["_dedup_key"] = keys
    n_all = len(df)
    n_key_na = df["_dedup_key"].isna().sum()

    if n_key_na > 0:
        df.loc[df["_dedup_key"].isna(), "_dedup_key"] = (
            "RAWSMI::" + df.loc[df["_dedup_key"].isna(), smiles_col].astype(str)
        )

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

    # 目标性质名（默认含能材料，但会根据 prop-cols 自动覆盖）
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
                    help="把 D/P/EG/r0 映射到 CSV 的实际列名，如: --prop-cols D D_phys_km_s ...")

    # 单位
    ap.add_argument("--units", nargs="*", default=[],
                    help="为各指标指定单位，如: --units D km/s ...")

    # 中心分子透明抠图阈值
    ap.add_argument("--center-white-threshold", type=int, default=250)
    ap.add_argument("--center-alpha", type=float, default=0.55)
    ap.add_argument("--transparent-figure", action="store_true")
    ap.add_argument("--outdir", required=True)

    # 去重
    ap.add_argument("--dedup", action="store_true", help="对分子去重")
    ap.add_argument("--dedup-by", choices=["canonical_smiles", "smiles", "inchi", "inchikey"],
                    default="canonical_smiles")
    ap.add_argument("--dedup-keep",
                    choices=["max_reward", "min_sa", "max_reward_then_min_sa", "first", "last"],
                    default="max_reward")
    ap.add_argument("--dedup-stage", choices=["before", "after"], default="after")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    radar_dir = os.path.join(args.outdir, "radars")
    os.makedirs(radar_dir, exist_ok=True)

    # --- 1. 解析用户提供的映射 ---
    user_prop_map = {}
    if args.prop_cols:
        if len(args.prop_cols) % 2 != 0:
            raise ValueError("prop-cols 参数必须成对出现")
        for i in range(0, len(args.prop_cols), 2):
            user_prop_map[args.prop_cols[i]] = args.prop_cols[i + 1]

    # --- 2. 智能推断 targets ---
    # 如果用户没指定 --targets (即使用了默认值)，但提供了 prop-cols
    # 且 prop-cols 的键跟默认 targets (D, P...) 完全对不上，则认为用户想画 prop-cols 里的那些键。
    default_targets = set(["D", "P", "EG", "r0"])
    current_targets = args.targets
    
    if set(current_targets) == default_targets and user_prop_map:
        # 检查 prop_cols 的 keys 是否在默认 targets 里
        # 如果 prop_cols 的 key 都是新的 (如 homo, gap)，则自动切换 targets
        user_keys = set(user_prop_map.keys())
        if not user_keys.intersection(default_targets):
            print(f"[Auto-detect] 检测到 prop-cols 与默认 targets 不符，自动将 targets 切换为: {list(user_prop_map.keys())}")
            current_targets = list(user_prop_map.keys())

    # --- 3. 构建最终的 prop_cols ---
    # 默认逻辑: prop_cols[t] = "pred_" + t
    prop_cols = {}
    for t in current_targets:
        if t in user_prop_map:
            prop_cols[t] = user_prop_map[t]
        else:
            prop_cols[t] = f"pred_{t}"

    # --- 单位解析 ---
    units = {}
    if args.units:
        if len(args.units) % 2 != 0:
            raise ValueError("单位参数必须成对出现")
        for i in range(0, len(args.units), 2):
            units[args.units[i]] = args.units[i + 1]

    # --- 数据读取 ---
    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    
    # 检查必要列
    needed = [args.smiles_col, args.reward_col, args.sa_col] + [prop_cols[t] for t in current_targets]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少列：{miss}\n现有列：{list(df.columns)}\n"
                         f"提示：请检查 --prop-cols 是否正确映射了 CSV 中的列名，或者 --targets 是否指定正确。")

    # --- 可选：反标准化 ---
    if args.unzscore_stats:
        with open(args.unzscore_stats) as f:
            stats = json.load(f)
        means, stds = stats.get("means", {}), stats.get("stds", {})
        for t in current_targets:
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

    # --- 保存去重后的数据 (供后续步骤使用) ---
    if args.dedup:
        dedup_path = os.path.join(args.outdir, "all_molecules_dedup.csv")
        df.to_csv(dedup_path, index=False)
        print(f"[Saved] 去重后数据已保存至: {dedup_path}")

    # --- 排序 & TopN ---
    df_sorted = df.sort_values(args.reward_col, ascending=False).reset_index(drop=True)
    df_top = df_sorted.head(args.topn).copy()
    ref_df = df_sorted if args.use_all_range else df_top

    # --- 参考统计 ---
    # 填充 goals: 如果用户提供的 goals 数量少于 targets，默认补 "max"
    goals_list = args.goals
    if len(goals_list) < len(current_targets):
        goals_list = goals_list + ["max"] * (len(current_targets) - len(goals_list))
    
    goals = {t: g.lower() for t, g in zip(current_targets, goals_list)}
    goals["SA"] = "min"
    goals["Reward"] = "max"

    qlo, qhi = float(args.quantiles[0]), float(args.quantiles[1])
    ref_stats: Dict[str, Dict[str, float]] = {}

    # 目标性质
    for t in current_targets:
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

        # 原始显示值
        raw_vals: Dict[str, float] = {t: float(row.get(prop_cols[t], np.nan)) for t in current_targets}
        raw_vals["SA"] = float(row[args.sa_col])
        raw_vals["Reward"] = float(row[args.reward_col])

        # 归一化前的处理
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