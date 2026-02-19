#!/usr/bin/env python3
# tsne_visualize.py: 可视化生成分子Morgan指纹的t-SNE分布（按奖励值着色）

import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def compute_morgan_fingerprints(smiles_list, radius=2, nbits=2048):
    """计算一组SMILES的Morgan指纹, 返回fingerprint矩阵和有效分子索引"""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # 跳过非法SMILES
        # 计算Morgan指纹位向量
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=int)
        ConvertToNumpyArray(bv, arr)  # 将BitVect转换为numpy数组
        fps.append(arr)
        valid_idx.append(i)
    if len(fps) == 0:
        return np.empty((0, nbits)), []
    X = np.stack(fps, axis=0)
    return X, valid_idx

def main():
    parser = argparse.ArgumentParser(description="Visualize molecules via t-SNE on Morgan fingerprints.")
    parser.add_argument('--gen-csv', required=True, help="CSV file containing generated molecules (SMILES, reward, etc.)")
    parser.add_argument('--smiles-col', required=True, help="Name of the SMILES column in the input CSV")
    parser.add_argument('--reward-col', required=True, help="Name of the reward column in the input CSV")
    parser.add_argument('--sa-col', help="(Optional) Name of the Synthetic Accessibility (SA) score column")
    parser.add_argument('--outdir', default='.', help="Output directory to save results")
    parser.add_argument('--radius', type=int, default=2, help="Morgan fingerprint radius (default 2)")
    parser.add_argument('--nbits', type=int, default=2048, help="Morgan fingerprint length in bits (default 2048)")
    parser.add_argument('--perplexity', type=float, default=None, help="t-SNE perplexity (default: auto choose)")
    parser.add_argument('--n-iter', type=int, default=1000, help="t-SNE number of iterations (default 1000)")
    parser.add_argument('--sample', type=int, help="If set, randomly sample this many molecules for visualization")
    parser.add_argument('--random-state', type=int, default=None, help="Random seed for t-SNE and sampling (optional)")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    # 读取CSV数据
    df = pd.read_csv(args.gen_csv)
    # 下采样数据（如有需要）
    if args.sample is not None and len(df) > args.sample:
        df_sample = df.sample(n=args.sample, random_state=args.random_state)
        df_sample = df_sample.reset_index(drop=True)
    else:
        df_sample = df.copy()
    smiles_list = df_sample[args.smiles_col].astype(str).tolist()
    reward_list = df_sample[args.reward_col].tolist()
    sa_list = None
    if args.sa_col:
        # 如果指定了SA列但数据中不存在，则给出警告
        if args.sa_col not in df_sample.columns:
            print(f"Warning: SA column '{args.sa_col}' not found in data. Ignoring SA.")
        else:
            sa_list = df_sample[args.sa_col].tolist()

    # 计算Morgan指纹矩阵
    X, valid_idx = compute_morgan_fingerprints(smiles_list, radius=args.radius, nbits=args.nbits)
    if X.shape[0] == 0:
        raise SystemExit("Error: No valid molecules for fingerprinting (check SMILES validity).")

    # 如有无效分子被跳过，从原始列表中过滤相应的数据以保持对应关系
    if len(valid_idx) < len(smiles_list):
        smiles_list = [smiles_list[i] for i in valid_idx]
        reward_list = [reward_list[i] for i in valid_idx]
        if sa_list is not None:
            sa_list = [sa_list[i] for i in valid_idx]

    # 自动选择perplexity（若未提供）
    perplexity = args.perplexity
    if perplexity is None:
        # 若未指定，选一个默认值：小数据集用较小perplexity，大数据集用30
        if X.shape[0] - 1 < 5:
            perplexity = 5.0  # 最小取5
        elif X.shape[0] - 1 < 30:
            # 对于非常小的样本数，取样本数减1的一定比例
            perplexity = float(max(5, X.shape[0] - 1))
        else:
            perplexity = 30.0
    # 确保perplexity不超过样本数 - 1
    if perplexity >= X.shape[0]:
        perplexity = float(X.shape[0] - 1)
    print(f"Using t-SNE perplexity = {perplexity}, n_iter = {args.n_iter}")

    # 运行t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=args.n_iter, random_state=args.random_state)
    embedding = tsne.fit_transform(X)  # 得到二维坐标数组

    # 准备输出数据表
    output_dict = {
        "smiles": smiles_list,
        "reward": reward_list,
        "x": embedding[:, 0],
        "y": embedding[:, 1]
    }
    if sa_list is not None:
        output_dict["SA"] = sa_list
    df_out = pd.DataFrame(output_dict)
    out_csv_path = os.path.join(args.outdir, "tsne_embedding.csv")
    df_out.to_csv(out_csv_path, index=False)
    print(f"Saved embedding data to {out_csv_path}")

    # 绘制散点图（奖励值着色）
    x_coords = embedding[:, 0]
    y_coords = embedding[:, 1]
    rewards = np.array(reward_list)
    # 计算颜色映射范围（1%和99%分位）
    vmin = np.percentile(rewards, 1) if rewards.size > 0 else None
    vmax = np.percentile(rewards, 99) if rewards.size > 0 else None
    if vmax is not None and vmin is not None and vmax <= vmin:
        # 若分位结果退化（如所有reward相等），则不截断
        vmin, vmax = None, None

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x_coords, y_coords, c=rewards, cmap='viridis', 
                     vmin=vmin, vmax=vmax, s=20, edgecolors='none', alpha=0.8)
    plt.colorbar(sc, label='Reward')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE of molecules (colored by reward)')
    plt.tight_layout()
    out_img_path = os.path.join(args.outdir, "tsne_reward.png")
    plt.savefig(out_img_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {out_img_path}")

if __name__ == "__main__":
    main()
