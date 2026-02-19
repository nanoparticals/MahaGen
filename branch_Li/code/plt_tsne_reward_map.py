#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: 生成式含能分子 — t-SNE map colored by total reward

Purpose
-------
Take generated molecules (CSV from analyze_selfies_sa_reward.py, e.g., sa_lt_5.csv or all_molecules.csv),
compute Morgan fingerprints, embed them to 2D using t-SNE, and plot a scatter map colored by total reward.

Outputs
-------
- tsne_embedding.csv : columns [smiles, reward, x, y] (+ optional SA if present)
- tsne_reward.png    : 2D scatter colored by reward (robust color scaling via 1–99th percentiles)

Optional
--------
- Overlay a small random subset of the training set in light gray for visual reference.

Usage
-----
python tsne_reward_map.py \
  --gen-csv outputs/sa_reward/sa_lt_5.csv --smiles-col smiles --reward-col reward \
  --outdir outputs/tsne_map --radius 2 --nbits 2048 --perplexity auto --n-iter 1500 --sample 20000

# with training overlay
python tsne_reward_map.py \
  --gen-csv outputs/sa_reward/all_molecules.csv --smiles-col smiles --reward-col reward \
  --train data/train_smiles.csv --train-col smiles --train-has-header --train-overlay 2000 \
  --outdir outputs/tsne_map
"""
from __future__ import annotations
import os, argparse, math, random
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE

# --------- helpers ---------

def _read_column(path: str, col: str | int, has_header: bool) -> List[str]:
    df = pd.read_csv(path, header=0 if has_header else None)
    if isinstance(col, str) and col.isdigit():
        col = int(col)
    series = df.iloc[:, col] if isinstance(col, int) else df[col]
    return [str(x).strip() for x in series.tolist() if str(x).strip()]


def canonicalize_smiles_list(vals: List[str]) -> List[str]:
    cans = []
    for s in vals:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            cans.append(Chem.MolToSmiles(m, canonical=True))
    return cans


def morgan_fp_bits(smiles: List[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    X = np.zeros((len(smiles), nbits), dtype=np.uint8)
    keep = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            keep.append(False)
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.int8)
        # RDKit explicit conversion
        from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(bv, arr)
        X[i, :] = (arr > 0).astype(np.uint8)
        keep.append(True)
    return X[keep, :], [s for s,k in zip(smiles, keep) if k]

# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen-csv', required=True, help='CSV from analyze_selfies_sa_reward.py (sa_lt_5.csv or all_molecules.csv)')
    ap.add_argument('--smiles-col', default='smiles')
    ap.add_argument('--reward-col', default='reward')
    ap.add_argument('--sa-col', default='SA', help='optional; if present will be carried through')

    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--nbits', type=int, default=2048)

    ap.add_argument('--perplexity', default='auto', help='float or "auto" (defaults to min(30, (n-1)/3))')
    ap.add_argument('--n-iter', type=int, default=1500)
    ap.add_argument('--learning-rate', default='auto')
    ap.add_argument('--random-state', type=int, default=42)

    ap.add_argument('--sample', type=int, default=0, help='subsample generated points for speed (0 = no downsample)')

    # Optional training overlay
    ap.add_argument('--train', default='', help='training SMILES file (CSV/TXT)')
    ap.add_argument('--train-col', default='smiles')
    ap.add_argument('--train-has-header', action='store_true')
    ap.add_argument('--train-overlay', type=int, default=0, help='overlay up to N random train points (0=off)')

    ap.add_argument('--outdir', required=True)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load generated CSV
    gen_df = pd.read_csv(args.gen_csv)
    if args.smiles_col not in gen_df.columns or args.reward_col not in gen_df.columns:
        raise ValueError(f"Columns '{args.smiles_col}' and '{args.reward_col}' must exist in {args.gen_csv}.")

    smiles = gen_df[args.smiles_col].astype(str).tolist()
    rewards = gen_df[args.reward_col].astype(float).values
    SA_vals = gen_df[args.sa_col].astype(float).values if args.sa_col in gen_df.columns else None

    # Subsample for speed if needed
    idx = np.arange(len(smiles))
    if args.sample and args.sample > 0 and args.sample < len(smiles):
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(idx, size=args.sample, replace=False)
        smiles = [smiles[i] for i in idx]
        rewards = rewards[idx]
        if SA_vals is not None:
            SA_vals = SA_vals[idx]

    # Canonicalize and featurize
    smiles = canonicalize_smiles_list(smiles)
    X, smiles = morgan_fp_bits(smiles, radius=args.radius, nbits=args.nbits)
    # Match rewards/SA length after dropping invalids
    valid_mask = np.ones(X.shape[0], dtype=bool)
    if len(rewards) != len(X):
        # rebuild by mapping smiles… assume gen_csv may contain duplicates/invalids; remap via a simple pass
        # Build map from canonical smiles to first-seen reward; duplicates handled by first occurrence
        smi_to_reward = {}
        smi_to_sa = {}
        for s, r, irow in zip(gen_df[args.smiles_col].astype(str).tolist(), gen_df[args.reward_col].astype(float).tolist(), range(len(gen_df))):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            cs = Chem.MolToSmiles(m, canonical=True)
            if cs not in smi_to_reward:
                smi_to_reward[cs] = r
                if args.sa_col in gen_df.columns:
                    try:
                        smi_to_sa[cs] = float(gen_df.iloc[irow][args.sa_col])
                    except Exception:
                        pass
        rewards = np.array([smi_to_reward.get(s, np.nan) for s in smiles], dtype=float)
        if args.sa_col in gen_df.columns:
            SA_vals = np.array([smi_to_sa.get(s, np.nan) for s in smiles], dtype=float)

    n = X.shape[0]
    if n < 5:
        raise RuntimeError('Too few valid molecules for t-SNE.')

    if str(args.perplexity).lower() == 'auto':
        perplexity = float(min(30, max(5, (n - 1) / 3)))
    else:
        perplexity = float(args.perplexity)
        if perplexity >= n:
            perplexity = max(5.0, min(30.0, 0.25 * n))
    print(f"[info] t-SNE with n={n}, perplexity={perplexity}, n_iter={args.n_iter}")

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=args.n_iter,
                learning_rate=args.learning_rate, init='pca', random_state=args.random_state, verbose=1)
    Y = tsne.fit_transform(X)

    # Robust color scaling by reward (clip 1–99 percentile)
    lo, hi = np.nanpercentile(rewards, [1, 99])
    rewards_clipped = np.clip(rewards, lo, hi)

    # Plot
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Y[:,0], Y[:,1], c=rewards_clipped, s=8, cmap='viridis')
    cb = plt.colorbar(sc)
    cb.set_label('Total reward (clipped 1–99%ile)')
    plt.title('t-SNE of generated molecules (colored by total reward)')
    plt.xticks([]); plt.yticks([])

    # Optional training overlay (light gray, behind)
    if args.train and args.train_overlay and args.train_overlay > 0:
        # load train smiles
        if args.train.lower().endswith('.csv'):
            tr_vals = _read_column(args.train, args.train_col, args.train_has_header)
        else:
            with open(args.train, 'r', encoding='utf-8') as f:
                tr_vals = [ln.strip() for ln in f if ln.strip()]
        tr_smiles = canonicalize_smiles_list(tr_vals)
        if len(tr_smiles) > args.train_overlay:
            rng = np.random.default_rng(args.random_state)
            tr_smiles = list(rng.choice(tr_smiles, size=args.train_overlay, replace=False))
        # project train into the same space by parametric trick? (t-SNE is non-parametric)
        # Here we only visualize them as small gray dots by running a separate t-SNE on combined data
        all_smiles = smiles + tr_smiles
        X_all, all_smiles = morgan_fp_bits(all_smiles, radius=args.radius, nbits=args.nbits)
        n_gen = len(smiles)
        tsne2 = TSNE(n_components=2, perplexity=perplexity, n_iter=max(750, args.n_iter//2),
                     learning_rate=args.learning_rate, init='pca', random_state=args.random_state, verbose=0)
        Y_all = tsne2.fit_transform(X_all)
        # redraw: train in light gray, gen colored
        plt.clf()
        plt.figure(figsize=(7, 6))
        plt.scatter(Y_all[n_gen:,0], Y_all[n_gen:,1], c='lightgray', s=4, alpha=0.6, label='train overlay')
        sc = plt.scatter(Y_all[:n_gen,0], Y_all[:n_gen,1], c=rewards_clipped, s=8, cmap='viridis', label='generated')
        cb = plt.colorbar(sc)
        cb.set_label('Total reward (clipped 1–99%ile)')
        plt.legend(frameon=False)
        plt.title('t-SNE of generated molecules (with train overlay)')
        plt.xticks([]); plt.yticks([])

    png_path = os.path.join(args.outdir, 'tsne_reward.png')
    plt.tight_layout(); plt.savefig(png_path, dpi=180)
    plt.close()

    # Save embedding CSV
    out_df = pd.DataFrame({'smiles': smiles,
                           'reward': rewards,
                           'x': Y[:,0], 'y': Y[:,1]})
    if SA_vals is not None and len(SA_vals) == len(out_df):
        out_df['SA'] = SA_vals
    out_csv = os.path.join(args.outdir, 'tsne_embedding.csv')
    out_df.to_csv(out_csv, index=False)
    print('[OK] Wrote:', out_csv)
    print('[OK] Wrote:', png_path)

if __name__ == '__main__':
    main()
