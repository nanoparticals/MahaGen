#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare property distributions between training and generated molecules.
Draw KDE plots, boxplots, and t-SNE colored by reward.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

PROPS = ['pred_D', 'pred_P', 'pred_EG', 'pred_r0']


def morgan_fp_bits(smiles_list, radius=2, nbits=2048):
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        arr = np.zeros((nbits,), dtype=np.int8)
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        AllChem.DataStructs.ConvertToNumpyArray(bv, arr)
        fps.append(arr)
        valid_smiles.append(smi)
    return np.array(fps), valid_smiles


def plot_kde_box_tsne(train_df, gen_df, outpath):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    sns.set_style("whitegrid")

    for i, prop in enumerate(PROPS):
        ax_kde = axes[0, i]
        ax_box = axes[1, i]
        sns.kdeplot(train_df[prop], label='Train', ax=ax_kde, color='steelblue', lw=2)
        sns.kdeplot(gen_df[prop], label='Generated', ax=ax_kde, color='goldenrod', lw=2)
        ax_kde.set_title(f"KDE of {prop}")
        ax_kde.legend()

        sns.boxplot(data=[train_df[prop], gen_df[prop]], ax=ax_box, palette=['steelblue', 'goldenrod'])
        ax_box.set_xticklabels(['Train', 'Generated'])
        ax_box.set_title(f"Boxplot of {prop}")

    # t-SNE subplot
    smiles = gen_df['smiles'].tolist()
    fps, valid_smiles = morgan_fp_bits(smiles)
    rewards = gen_df.loc[gen_df['smiles'].isin(valid_smiles), 'reward'].values
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    Y = tsne.fit_transform(fps)

    ax_tsne = axes[2, 1]
    sc = ax_tsne.scatter(Y[:, 0], Y[:, 1], c=rewards, cmap='viridis', s=10)
    cb = plt.colorbar(sc, ax=ax_tsne)
    cb.set_label('Reward')
    ax_tsne.set_title('t-SNE of Generated Molecules')
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])

    # Remove unused subplot spaces
    axes[2, 0].axis('off')
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--gen_csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    gen_df = pd.read_csv(args.gen_csv)
    for p in PROPS:
        if p not in train_df.columns:
            train_df[p] = np.nan
    plot_kde_box_tsne(train_df, gen_df, args.output)


if __name__ == '__main__':
    main()