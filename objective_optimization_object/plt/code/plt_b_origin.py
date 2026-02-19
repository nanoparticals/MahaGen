#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot panel (b): 2D projection of chemical space for
- training structures (blue)
- generated structures (yellow/orange)

Now supports either SMILES or SELFIES for each file independently.

Usage example (your case):
  python plt_b.py \
    --train /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin_selfies.csv \
    --gen   /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_18/outputs/all_molecules_dedup.csv \
    --train-col smiles --gen-col selfies \
    --train-mode smiles --gen-mode selfies \
    --out /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_18/outputs/figure_b_chemspace_gen.png

Features:
  - RDKit Morgan fingerprints (radius=2, nBits=2048) if available
  - Fallback to SMILES character n-gram hashed features if RDKit is unavailable
  - Dimensionality reduction: UMAP -> t-SNE -> PCA (auto fallback)
  - Optional subsampling to speed up plotting
  - Robust reading of CSV/TSV with auto separator detection
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

# ----- Optional imports (handled gracefully) -----
_have_rdkit = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    _have_rdkit = False

_have_umap = True
try:
    import umap
except Exception:
    _have_umap = False

_have_selfies = True
try:
    import selfies as sf
except Exception:
    _have_selfies = False


# ---------------- IO helpers ----------------

def read_table(path: str, colname: str) -> list[str]:
    """Robust reader:
    - CSV/TSV with a named column (case-insensitive)
    - CSV/TSV single unnamed column -> use that column
    - Plain text file (one string per line)
    All paths return a list[str] without empty lines.
    """
    # Try CSV/TSV first
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        # case-insensitive match
        if colname not in df.columns:
            for c in df.columns:
                if str(c).lower() == str(colname).lower():
                    colname = c
                    break
        if colname in df.columns:
            s = df[colname].astype(str)
            s = s[s.str.len() > 0]
            return s.tolist()
        # single-column fallback
        if df.shape[1] == 1:
            s = df.iloc[:, 0].astype(str)
            s = s[s.str.len() > 0]
            return s.tolist()
        # otherwise, raise a clear error
        raise ValueError(f"Column '{colname}' not found in {path}. Columns: {list(df.columns)[:12]}…")
    except Exception:
        # Plain text fallback
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f]
            lines = [x for x in lines if x]
            if lines:
                return lines
        except Exception:
            pass
        raise



def selfies_to_smiles_list(seq_list: list[str]) -> list[str]:
    if not _have_selfies:
        raise RuntimeError("SELFIES requested but the 'selfies' package is not installed. pip install selfies")
    out = []
    for s in seq_list:
        try:
            smi = sf.decoder(s)
        except Exception:
            smi = None
        out.append(smi or "")
    return out


def normalize_smiles_list(smiles_list: list[str]) -> list[str]:
    if not _have_rdkit:
        # give back raw strings; feature builder may hash n-grams
        return [x for x in smiles_list if isinstance(x, str) and len(x) > 0]
    out = []
    for s in smiles_list:
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        out.append(Chem.MolToSmiles(mol))
    return out


# ---------------- Feature builders ----------------

def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    fps = []
    keep_idx = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        keep_idx.append(i)
    if not fps:
        raise ValueError("No valid molecules parsed by RDKit from provided SMILES.")
    return np.array(fps, dtype=np.float32), np.array(keep_idx, dtype=int)


def smiles_to_ngram_features(smiles_list, ngram=(1, 3), n_features=2048):
    # build n-gram tokens
    tokens_per_row = []
    n_min, n_max = ngram
    for s in smiles_list:
        s = s if isinstance(s, str) else str(s)
        toks = []
        for n in range(n_min, n_max + 1):
            toks.extend([f"g{n}:{s[i:i+n]}" for i in range(0, max(0, len(s)-n+1))])
        tokens_per_row.append(toks or ["__EMPTY__"])
    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    X = hasher.transform(tokens_per_row).toarray().astype(np.float32)
    return X, np.arange(len(smiles_list), dtype=int)


def build_features(smiles, prefer_rdkit=True, n_bits=2048, radius=2):
    if prefer_rdkit and _have_rdkit:
        try:
            X, keep = smiles_to_morgan(smiles, radius=radius, n_bits=n_bits)
            return X, keep, "Morgan"
        except Exception:
            warnings.warn("RDKit fingerprint failed; falling back to n-gram features.")
    X, keep = smiles_to_ngram_features(smiles, ngram=(1, 3), n_features=n_bits)
    return X, keep, "SMILES n-gram"


# ---------------- DR reducers ----------------

def reduce_dim(X, method="auto", seed=42, n_neighbors=30, min_dist=0.1):
    if method == "umap" or (method == "auto" and _have_umap):
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric="jaccard" if X.dtype.kind in "biu" else "euclidean",
                            random_state=seed)
        Y = reducer.fit_transform(X)
        return Y, "UMAP"
    if method in ("tsne", "auto"):
        try:
            Y = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                     init="pca", random_state=seed).fit_transform(X)
            return Y, "t-SNE"
        except Exception:
            pass
    Y = PCA(n_components=2, random_state=seed).fit_transform(X)
    return Y, "PCA"


# ---------------- misc ----------------

def maybe_subsample(n, max_n, seed=42):
    if max_n is None or n <= max_n:
        return np.arange(n, dtype=int)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return np.sort(idx)


def prepare_smiles(path: str, col: str, mode: str) -> list[str]:
    seqs = read_table(path, col)
    # Normalize common SELFIES wrappers like angle brackets
    seqs = [s.strip().replace(" ", "") for s in seqs]
    mode = mode.lower()
    if mode == "auto":
        n_bracket = sum(('[' in s and ']' in s) for s in seqs[:200])
        mode = "selfies" if n_bracket >= max(5, len(seqs[:200]) // 10) else "smiles"
    if mode == "selfies":
        # Remove possible angle brackets occasionally included in dumps
        seqs = [s.replace("<", "").replace(">", "") for s in seqs]
        smi = selfies_to_smiles_list(seqs)
    elif mode == "smiles":
        smi = seqs
    else:
        raise ValueError("mode must be one of {auto, smiles, selfies}")
    smi = normalize_smiles_list(smi)
    if len(smi) == 0:
        raise RuntimeError(f"No valid molecules after parsing/normalizing from {path} ({mode}).")
    return smi


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Chemical space projection (SMILES/SELFIES aware)")
    ap.add_argument("--train", required=True, help="训练集 CSV/TSV 路径")
    ap.add_argument("--gen", required=True, help="生成集 CSV/TSV 路径")
    ap.add_argument("--train-col", default="smiles", help="训练集分子列名（SMILES或SELFIES）")
    ap.add_argument("--gen-col", default="smiles", help="生成集分子列名（SMILES或SELFIES）")
    ap.add_argument("--train-mode", choices=["auto", "smiles", "selfies"], default="auto",
                    help="训练集列的解释方式")
    ap.add_argument("--gen-mode", choices=["auto", "smiles", "selfies"], default="auto",
                    help="生成集列的解释方式")
    ap.add_argument("--max-train", type=int, default=4000, help="训练集最大抽样数")
    ap.add_argument("--max-gen", type=int, default=20000, help="生成集最大抽样数")
    ap.add_argument("--bits", type=int, default=2048, help="指纹/哈希特征维度")
    ap.add_argument("--radius", type=int, default=4, help="Morgan 指纹半径")
    ap.add_argument("--reducer", choices=["auto", "umap", "tsne", "pca"], default="auto", help="降维方法")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--out", default="figure_b_chemspace_2w.png", help="输出图片文件")
    args = ap.parse_args()

    # Prepare SMILES for both sets (handles SELFIES if requested)
    smiles_train = prepare_smiles(args.train, args.train_col, args.train_mode)
    smiles_gen   = prepare_smiles(args.gen,   args.gen_col,   args.gen_mode)

    # Build features
    X_train, keep_tr, fe_name = build_features(smiles_train, n_bits=args.bits, radius=args.radius)
    X_gen,   keep_ge, _       = build_features(smiles_gen,   n_bits=args.bits, radius=args.radius)

    # Optional subsampling indices
    idx_tr = maybe_subsample(X_train.shape[0], args.max_train, seed=args.seed)
    idx_ge = maybe_subsample(X_gen.shape[0],   args.max_gen,   seed=args.seed)

    X_all = np.vstack([X_train[idx_tr], X_gen[idx_ge]])
    labels = np.array([0] * len(idx_tr) + [1] * len(idx_ge))

    # Reduce to 2D
    Y, red_name = reduce_dim(X_all, method=args.reducer, seed=args.seed)

    # Split back
    Y_tr = Y[labels == 0]
    Y_ge = Y[labels == 1]

    # Plot
    plt.figure(figsize=(7.5, 5.5), dpi=150)
    plt.scatter(Y_ge[:, 0], Y_ge[:, 1], s=14, alpha=0.7, label="Generated", color="#ff7f0e", edgecolors="none")
    plt.scatter(Y_tr[:, 0], Y_tr[:, 1], s=14, alpha=0.8, label="Training set", color="#1f77b4", edgecolors="none")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Chemical space projection — {red_name} on {fe_name} features")
    plt.legend()
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)

    print(f"Saved figure to: {Path(args.out).resolve()}")
    print(f"[Info] Features: {fe_name} | Reducer: {red_name} | RDKit: {_have_rdkit} | UMAP: {_have_umap} | SELFIES: {_have_selfies}")


if __name__ == "__main__":
    main()
