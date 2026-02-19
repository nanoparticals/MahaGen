#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析分子集合多样性的脚本（Circles + Bottleneck + 生成→训练最近邻）

功能：
1. 读取训练集 & 生成集 CSV（SMILES 或 SELFIES）。
2. 计算 Morgan 指纹（radius=2, nBits=2048），使用 Tanimoto/Jaccard 距离。
3. 训练集 / 生成集 内部：
    - Circles（贪心最大排斥圆）
    - Bottleneck（最近邻距离平均值）
4. 生成集 → 训练集 最近邻：
    - 每个生成分子到最近训练分子的 Tanimoto 距离 & 相似度
    - 保存 CSV
5. 等量子采样对比：
    - 从生成集中多次随机抽取与训练集同样数量的分子
    - 计算每个子样本的 Circles & Bottleneck
6. Circles(train ∪ gen_sample) - Circles(train)：
    - 估算生成分子贡献的“新簇”数量

依赖：
    pandas, numpy, scikit-learn, rdkit, 可选 selfies

示例：
    python analyze_circles_bottleneck.py \
        --train /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin_selfies.csv \
        --gen   /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_physical.csv \
        --use-selfies \
        --threshold 0.5 \
        --repeats 30 \
        --out-prefix vgae_gate_16

"""

import argparse
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

RDLogger.DisableLog("rdApp.*")


# ---------- 基础工具 ----------

def detect_column(df: pd.DataFrame,
                  preferred: Optional[str],
                  mode_selfies: bool) -> str:
    """自动检测 SMILES / SELFIES 列名（也可以手动指定）"""
    if preferred and preferred in df.columns:
        return preferred
    candidates = []
    if mode_selfies:
        candidates = ["selfies", "SELFIES"]
    else:
        candidates = ["smiles", "SMILES"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"无法在列 {list(df.columns)} 中找到合适的 "
                     f"{'SELFIES' if mode_selfies else 'SMILES'} 列，请用 --train-col / --gen-col 指定。")


def load_strings(path: str,
                 col_name: Optional[str],
                 use_selfies: bool) -> List[str]:
    """从 CSV 读取分子字符串（SMILES 或 SELFIES）"""
    df = pd.read_csv(path)
    if 'valid' in df.columns:
        df = df[df['valid'] == True]
    col = detect_column(df, col_name, use_selfies)
    strings = df[col].dropna().astype(str).tolist()
    return strings


def decode_selfies_to_smiles(strings: List[str]) -> List[str]:
    """SELFIES -> SMILES"""
    import selfies as sf
    smiles = []
    for s in strings:
        try:
            smi = sf.decoder(s)
            if smi:
                smiles.append(smi)
        except Exception:
            continue
    return smiles


def mols_and_fps_from_smiles(smiles: List[str],
                             radius: int = 2,
                             n_bits: int = 2048):
    """SMILES -> RDKit Mol & Morgan 指纹"""
    mols = []
    fps = []
    valid_smiles = []
    bad = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            continue
        mols.append(m)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        fps.append(fp)
        valid_smiles.append(s)
    if bad > 0:
        print(f"[info] {bad} 条 SMILES 无法解析，已跳过。")
    return mols, fps, valid_smiles


def fps_to_bit_array(fps: List[DataStructs.ExplicitBitVect],
                     n_bits: int = 2048) -> np.ndarray:
    """RDKit 指纹列表 -> bool 矩阵 (N, n_bits)，用于 sklearn"""
    arr = np.zeros((len(fps), n_bits), dtype=bool)
    tmp = np.zeros((n_bits,), dtype=np.int8)
    for i, fp in enumerate(fps):
        tmp[:] = 0
        DataStructs.ConvertToNumpyArray(fp, tmp)
        arr[i] = tmp.astype(bool)
    return arr


def jaccard_distance_matrix(bit_arr: np.ndarray) -> np.ndarray:
    """集合内部 Jaccard 距离矩阵（= Tanimoto 距离）"""
    if bit_arr.shape[0] <= 1:
        return np.zeros((bit_arr.shape[0], bit_arr.shape[0]), dtype=float)
    return pairwise_distances(bit_arr, metric='jaccard')


# ---------- Circles & Bottleneck ----------

def greedy_circles(dist_mat: np.ndarray, threshold: float) -> List[int]:
    """
    贪心最大排斥圆：
      输入：距离矩阵 D，阈值 t
      输出：被选为“圆心”的索引列表 centers，满足任意 i!=j, D[i,j] > t
    """
    n = dist_mat.shape[0]
    remaining = list(range(n))
    centers: List[int] = []
    while remaining:
        i = remaining[0]
        centers.append(i)
        # 保留与当前中心距离 > t 的点
        new_remaining = []
        for j in remaining[1:]:
            if dist_mat[i, j] > threshold:
                new_remaining.append(j)
        remaining = new_remaining
    return centers


def bottleneck_from_dist(dist_mat: np.ndarray) -> Tuple[Optional[float], np.ndarray]:
    """
    Bottleneck: 每个点到最近邻的距离平均值
      返回 (平均值, 每个点的最近邻距离数组)
    """
    n = dist_mat.shape[0]
    if n <= 1:
        return None, np.array([], dtype=float)
    m = dist_mat.copy()
    np.fill_diagonal(m, np.inf)
    nn = m.min(axis=1)
    mean_val = float(nn.mean())
    return mean_val, nn


# ---------- 生成 → 训练 最近邻 ----------

def gen_to_train_nearest(train_fps,
                         gen_fps,
                         train_smiles: List[str],
                         gen_smiles: List[str],
                         out_csv: Optional[str] = None):
    """
    对每个生成分子，计算它到最近训练分子的 Tanimoto 相似度 / 距离。
    返回： (nn_dists, nn_sims)
    """
    nn_dists = []
    nn_sims = []
    rows = []
    for i, (fp_g, smi_g) in enumerate(zip(gen_fps, gen_smiles)):
        sims = DataStructs.BulkTanimotoSimilarity(fp_g, train_fps)
        sims_arr = np.array(sims, dtype=float)
        j = int(sims_arr.argmax())
        s_max = float(sims_arr[j])
        d_min = 1.0 - s_max
        nn_sims.append(s_max)
        nn_dists.append(d_min)
        rows.append({
            "gen_idx": i,
            "gen_smiles": smi_g,
            "train_idx": j,
            "train_smiles": train_smiles[j],
            "nn_similarity": s_max,
            "nn_distance": d_min,
        })
    if out_csv is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[save] 生成→训练 最近邻结果已保存到 {out_csv}")
    return np.array(nn_dists), np.array(nn_sims)


# ---------- 子采样对比 & 联合集 Circles ----------

def subsample_gen_equals_train(gen_bits: np.ndarray,
                               n_train: int,
                               repeats: int,
                               threshold: float,
                               seed: int = 42):
    """
    多次从生成集中抽取 n_train 个分子，计算每个子集的 Circles & Bottleneck
    返回： (circles_counts, bottleneck_means)
    """
    rng = np.random.default_rng(seed)
    n_gen = gen_bits.shape[0]
    if n_gen < n_train:
        raise ValueError(f"生成集 ({n_gen}) 小于训练集 ({n_train})，无法做等量子采样。")

    circles_list = []
    bottleneck_list = []
    for r in range(repeats):
        idx = rng.choice(n_gen, size=n_train, replace=False)
        sub_bits = gen_bits[idx]
        dist_sub = jaccard_distance_matrix(sub_bits)
        centers = greedy_circles(dist_sub, threshold)
        circle_count = len(centers)
        bottleneck_mean, _ = bottleneck_from_dist(dist_sub)
        circles_list.append(circle_count)
        bottleneck_list.append(bottleneck_mean)
        print(f"[subsample {r+1}/{repeats}] Circles={circle_count}, Bottleneck={bottleneck_mean:.3f}")
    return np.array(circles_list), np.array(bottleneck_list)


def union_circles(train_bits: np.ndarray,
                  gen_bits: np.ndarray,
                  threshold: float,
                  gen_sample: Optional[int],
                  seed: int = 42) -> Tuple[int, int]:
    """
    在 train ∪ gen_sample 上计算 Circles：
      - gen_sample: 若不为 None，则从生成集中随机抽取该数量
    返回：(circles_union, total_points_union)
    """
    rng = np.random.default_rng(seed)
    if gen_sample is not None and gen_bits.shape[0] > gen_sample:
        idx = rng.choice(gen_bits.shape[0], size=gen_sample, replace=False)
        gen_use = gen_bits[idx]
    else:
        gen_use = gen_bits
    both = np.vstack([train_bits, gen_use])
    dist_union = jaccard_distance_matrix(both)
    centers_union = greedy_circles(dist_union, threshold)
    return len(centers_union), both.shape[0]


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="Circles & Bottleneck 多样性分析")
    ap.add_argument("--train", required=True, help="训练集 CSV 路径")
    ap.add_argument("--gen", required=True, help="生成集 CSV 路径")
    ap.add_argument("--train-col", default=None,
                    help="训练集 SMILES/SELFIES 列名（默认自动检测）")
    ap.add_argument("--gen-col", default=None,
                    help="生成集 SMILES/SELFIES 列名（默认自动检测）")
    ap.add_argument("--use-selfies", action="store_true",
                    help="将输入视为 SELFIES，否则视为 SMILES")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Circles 的 Tanimoto 距离阈值 t，默认 0.5")
    ap.add_argument("--repeats", type=int, default=30,
                    help="从生成集等量子采样的重复次数")
    ap.add_argument("--union-gen-sample", type=int, default=None,
                    help="联合集 Circles 时，从生成集中抽取的样本数（默认用全部）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="circles_analysis",
                    help="输出文件前缀")
    args = ap.parse_args()

    # 1. 读取字符串
    print("[step] 读取训练集 / 生成集 分子字符串 ...")
    train_strings = load_strings(args.train, args.train_col, args.use_selfies)
    gen_strings = load_strings(args.gen, args.gen_col, args.use_selfies)
    print(f"[info] 训练集原始条数: {len(train_strings)}")
    print(f"[info] 生成集原始条数: {len(gen_strings)}")

    # SELFIES -> SMILES
    if args.use_selfies:
        print("[step] SELFIES 解码为 SMILES ...")
        try:
            train_strings = decode_selfies_to_smiles(train_strings)
            gen_strings = decode_selfies_to_smiles(gen_strings)
        except ImportError:
            print("错误: 需要安装 selfies 库 (pip install selfies)")
            sys.exit(1)

    # 2. SMILES -> Mol & FPS
    print("[step] 计算 Morgan 指纹 ...")
    train_mols, train_fps, train_smiles = mols_and_fps_from_smiles(train_strings)
    gen_mols, gen_fps, gen_smiles = mols_and_fps_from_smiles(gen_strings)
    print(f"[info] 训练集有效分子: {len(train_mols)}")
    print(f"[info] 生成集有效分子: {len(gen_mols)}")

    if len(train_mols) < 2 or len(gen_mols) < 2:
        print("有效分子太少，无法计算距离。")
        sys.exit(1)

    # 3. 指纹 -> bool 矩阵
    train_bits = fps_to_bit_array(train_fps)
    gen_bits = fps_to_bit_array(gen_fps)

    # 4. 训练 / 生成 内部 Circles & Bottleneck
    print("[step] 训练集内部 Circles & Bottleneck ...")
    dist_train = jaccard_distance_matrix(train_bits)
    centers_train = greedy_circles(dist_train, args.threshold)
    circles_train = len(centers_train)
    bott_train, nn_train = bottleneck_from_dist(dist_train)
    print(f"训练集: Circles = {circles_train}, Bottleneck = {bott_train:.3f}")

    print("[step] 生成集内部 Circles & Bottleneck ...")
    dist_gen_small = jaccard_distance_matrix(gen_bits)  # 若生成很大，可改成子采样
    centers_gen = greedy_circles(dist_gen_small, args.threshold)
    circles_gen = len(centers_gen)
    bott_gen, nn_gen = bottleneck_from_dist(dist_gen_small)
    print(f"生成集: Circles = {circles_gen}, Bottleneck = {bott_gen:.3f}")

    # 5. 生成 → 训练 最近邻
    print("[step] 计算 生成 → 训练 最近邻 ...")
    out_nn_csv = f"{args.out_prefix}_gen_to_train_nn.csv"
    nn_dists, nn_sims = gen_to_train_nearest(
        train_fps, gen_fps, train_smiles, gen_smiles,
        out_csv=out_nn_csv
    )
    print(f"[NN] 生成→训练 最近邻 距离: mean={nn_dists.mean():.3f}, "
          f"median={np.median(nn_dists):.3f}, "
          f"95%分位={np.quantile(nn_dists, 0.95):.3f}")
    print(f"[NN] 生成→训练 最近邻 相似度: mean={nn_sims.mean():.3f}, "
          f"median={np.median(nn_sims):.3f}, "
          f"5%分位={np.quantile(nn_sims, 0.05):.3f}")

    # 6. 等量子采样：生成子集 vs 训练集
    print("[step] 从生成集中等量子采样进行 Circles / Bottleneck 对比 ...")
    circles_sub, bott_sub = subsample_gen_equals_train(
        gen_bits, n_train=len(train_mols),
        repeats=args.repeats,
        threshold=args.threshold,
        seed=args.seed
    )
    out_sub_csv = f"{args.out_prefix}_gen_subsample_stats.csv"
    pd.DataFrame({
        "circles": circles_sub,
        "bottleneck": bott_sub
    }).to_csv(out_sub_csv, index=False)
    print(f"[save] 子采样统计已保存到 {out_sub_csv}")
    print(f"[subsample Circles] mean={circles_sub.mean():.1f}, "
          f"std={circles_sub.std(ddof=1):.1f}, "
          f"train={circles_train}")
    print(f"[subsample Bottleneck] mean={bott_sub.mean():.3f}, "
          f"std={bott_sub.std(ddof=1):.3f}, "
          f"train={bott_train:.3f}")

    # 7. 联合集 Circles(train ∪ gen_sample)
    print("[step] 计算 train ∪ gen_sample 的 Circles ...")
    circles_union, n_union = union_circles(
        train_bits, gen_bits,
        threshold=args.threshold,
        gen_sample=args.union_gen_sample,
        seed=args.seed
    )
    print(f"[union] Circles(train ∪ gen_sample) = {circles_union} (共 {n_union} 个分子)")
    print(f"[union] 新簇 ≈ {circles_union - circles_train} (Circles_union - Circles_train)")

    print("[done] 全部分析完成。")


if __name__ == "__main__":
    main()
