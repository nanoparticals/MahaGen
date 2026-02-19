#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chemical space projection (panel b) with two upgrades:

(A) Fit embedding on TRAINING SET only, then transform generated set.
(C) Plot TRAINING density contours (2D histogram -> contour).

Notes:
- UMAP supports transform(); t-SNE does NOT. So this script prefers UMAP; if not
  available, falls back to PCA (still fit on train, transform gen).
- For Morgan bit vectors, use metric='jaccard' (≈ Tanimoto geometry).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
import warnings

# ----- Optional imports (handled gracefully) -----
_have_rdkit = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
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
    """Robust reader for CSV/TSV or plain text."""
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        if colname not in df.columns:
            for c in df.columns:
                if str(c).lower() == str(colname).lower():
                    colname = c
                    break
        if colname in df.columns:
            s = df[colname].astype(str)
            s = s[s.str.len() > 0]
            return s.tolist()
        if df.shape[1] == 1:
            s = df.iloc[:, 0].astype(str)
            s = s[s.str.len() > 0]
            return s.tolist()
        raise ValueError(f"Column '{colname}' not found in {path}. Columns: {list(df.columns)[:12]}…")
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        lines = [x for x in lines if x]
        if not lines:
            raise
        return lines


def selfies_to_smiles_list(seq_list: list[str]) -> list[str]:
    if not _have_selfies:
        raise RuntimeError("SELFIES requested but 'selfies' package is not installed.")
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


def prepare_smiles(path: str, col: str, mode: str) -> list[str]:
    seqs = read_table(path, col)
    seqs = [s.strip().replace(" ", "") for s in seqs]
    mode = mode.lower()
    if mode == "auto":
        n_bracket = sum(('[' in s and ']' in s) for s in seqs[:200])
        mode = "selfies" if n_bracket >= max(5, len(seqs[:200]) // 10) else "smiles"
    if mode == "selfies":
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


# ---------------- Feature builders ----------------

def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    if not _have_rdkit:
        raise RuntimeError("RDKit not available; cannot compute Morgan fingerprints.")
    fps = []
    keep_idx = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        keep_idx.append(i)
    if not fps:
        raise ValueError("No valid molecules parsed by RDKit from provided SMILES.")
    # Keep float32 for sklearn/umap, but it is still binary 0/1
    X = np.array(fps, dtype=np.float32)
    return X, np.array(keep_idx, dtype=int)


def smiles_to_ngram_features(smiles_list, ngram=(1, 3), n_features=2048):
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


# ---------------- Subsampling ----------------

def maybe_subsample(n, max_n, seed=42):
    if max_n is None or n <= max_n:
        return np.arange(n, dtype=int)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return np.sort(idx)


# ---------------- (C) training density contours ----------------

def training_density_contours(ax, Y_tr, bins=70, levels=8):
    """2D histogram -> contour lines for training density (robust, no SciPy)."""
    x = Y_tr[:, 0]
    y = Y_tr[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
    Z = H.T  # contour expects [y,x]

    z_nonzero = Z[Z > 0]
    if len(z_nonzero) < 10:
        return
    qs = np.linspace(0.30, 0.95, levels)
    lev = np.unique(np.quantile(z_nonzero, qs))
    if len(lev) < 3:
        return
    ax.contour(Xc, Yc, Z, levels=lev, linewidths=1.0, alpha=0.7)


# ---------------- (A) fit train, transform gen ----------------

def fit_on_train_transform_gen(X_train, X_gen, fe_name, seed=42, n_neighbors=30, min_dist=0.1):
    if _have_umap:
        # Key fix: Morgan bits should use Jaccard, not Euclidean
        metric = "jaccard" if fe_name == "Morgan" else "euclidean"
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        Y_tr = reducer.fit_transform(X_train)
        Y_ge = reducer.transform(X_gen)
        return Y_tr, Y_ge, f"UMAP(metric={metric})"

    # Fallback: PCA (still supports fit->transform)
    reducer = PCA(n_components=2, random_state=seed)
    Y_tr = reducer.fit_transform(X_train)
    Y_ge = reducer.transform(X_gen)
    return Y_tr, Y_ge, "PCA"


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Chemical space projection: fit on train + density contours")
    ap.add_argument("--train", required=True, help="训练集 CSV/TSV 路径")
    ap.add_argument("--gen", required=True, help="生成集 CSV/TSV 路径")
    ap.add_argument("--train-col", default="smiles", help="训练集分子列名（SMILES或SELFIES）")
    ap.add_argument("--gen-col", default="smiles", help="生成集分子列名（SMILES或SELFIES）")
    ap.add_argument("--train-mode", choices=["auto", "smiles", "selfies"], default="auto")
    ap.add_argument("--gen-mode", choices=["auto", "smiles", "selfies"], default="auto")
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-gen", type=int, default=10000)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=6, help="Morgan 半径（建议=2）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--density-bins", type=int, default=70)
    ap.add_argument("--density-levels", type=int, default=0)
    ap.add_argument("--out", default="figure_b_chemspace_fit_train_density.png")
    args = ap.parse_args()

    smiles_train = prepare_smiles(args.train, args.train_col, args.train_mode)
    smiles_gen   = prepare_smiles(args.gen,   args.gen_col,   args.gen_mode)

    X_train, _, fe_name = build_features(smiles_train, n_bits=args.bits, radius=args.radius)
    X_gen,   _, _       = build_features(smiles_gen,   n_bits=args.bits, radius=args.radius)

    idx_tr = maybe_subsample(X_train.shape[0], args.max_train, seed=args.seed)
    idx_ge = maybe_subsample(X_gen.shape[0],   args.max_gen,   seed=args.seed)

    Y_tr, Y_ge, red_name = fit_on_train_transform_gen(
        X_train[idx_tr],
        X_gen[idx_ge],
        fe_name=fe_name,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)

    # (C) Training density contours first
    training_density_contours(ax, Y_tr, bins=args.density_bins, levels=args.density_levels)

    # Points
    ax.scatter(Y_ge[:, 0], Y_ge[:, 1], s=12, alpha=0.52, label="Generated", edgecolors="none")
    ax.scatter(Y_tr[:, 0], Y_tr[:, 1], s=16, alpha=0.85, label="Training set", edgecolors="none")

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(
        f"Chemical space projection — {red_name} on {fe_name} features\n"
        f"(fit on train, transform gen; training density contours)"
    )
    ax.legend(frameon=True)
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"Saved figure to: {Path(args.out).resolve()}")
    print(f"[Info] Features={fe_name} | Reducer={red_name} | RDKit={_have_rdkit} | UMAP={_have_umap} | SELFIES={_have_selfies}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
