'''python fingerprint_similarity.py \
  --train /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/aug_brics_1w.csv \
  --gen   /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_7/outputs/all_molecules_physical.csv \
  --radius 4 --nbits 2048 --topk 5 \
  --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/outputs/gpu_rl_vgae_soft_congen_7 --prefix morgan_r2_2048
'''
import argparse, sys, math, os, csv
from pathlib import Path
from typing import List, Tuple
import numpy as np

# RDKit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# Optional: selfies -> smiles
try:
    import selfies as sf
except Exception:
    sf = None

RDLogger.DisableLog("rdApp.*")

def read_molecules(path: Path) -> List[str]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in [".csv", ".tsv"]:
        import pandas as pd
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep)
        cols = [c.lower() for c in df.columns]
        smiles_col = None
        if "smiles" in cols:
            smiles_col = df.columns[cols.index("smiles")]
            smiles = df[smiles_col].astype(str).tolist()
            return smiles
        elif "selfies" in cols:
            if sf is None:
                raise RuntimeError("Found 'selfies' column but the 'selfies' package is not installed. Install selfies or provide SMILES.")
            selfies_list = df[df.columns[cols.index('selfies')]].astype(str).tolist()
            smiles = []
            for s in selfies_list:
                try:
                    smiles.append(sf.decoder(s))
                except Exception:
                    smiles.append("")
            return smiles
        else:
            raise RuntimeError("CSV/TSV must have a 'smiles' or 'selfies' column.")
    elif ext == ".smi" or ext == ".txt":
        smiles = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # allow "SMILES  name" tab-separated
                parts = line.split()
                smiles.append(parts[0])
        return smiles
    else:
        raise RuntimeError(f"Unsupported file type: {ext}")

def to_valid_mols(smiles_list: List[str]) -> List[Chem.Mol]:
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        mols.append(m)
    return mols

def morgan_fp(mol, radius=2, nbits=2048):
    if mol is None: return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    except Exception:
        return None

def bulk_nn_tanimoto(gen_fps, train_fps, topk=1) -> Tuple[np.ndarray, np.ndarray]:
    # For each generated fp, compute similarity to all train and take top-k
    nn = []
    nn_idx = []
    for gf in gen_fps:
        if gf is None:
            nn.append([np.nan]*topk); nn_idx.append([-1]*topk); continue
        sims = DataStructs.BulkTanimotoSimilarity(gf, train_fps)
        # argsort descending
        idx = np.argsort(sims)[::-1][:topk]
        nn.append([float(sims[i]) for i in idx])
        nn_idx.append([int(i) for i in idx])
    return np.array(nn, dtype=float), np.array(nn_idx, dtype=int)

def internal_diversity(fps) -> float:
    # 1 - average Tanimoto over 1% sampled pairs (or up to 20k pairs)
    import random
    ids = [i for i,f in enumerate(fps) if f is not None]
    n = len(ids)
    if n < 2: return float('nan')
    npairs = min(20000, max(1, int(0.01*n*(n-1)/2)))
    s = 0.0
    for _ in range(npairs):
        i, j = random.sample(ids, 2)
        s += DataStructs.TanimotoSimilarity(fps[i], fps[j])
    return 1.0 - s/npairs

def novelty_vs_train(gen_fps, train_fps, threshold=0.7) -> float:
    # fraction of generated molecules whose nearest neighbor in train < threshold
    nn1, _ = bulk_nn_tanimoto(gen_fps, train_fps, topk=1)
    nn1 = nn1[:,0]
    return float(np.mean(np.nan_to_num(nn1, nan=0.0) < threshold))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Training molecules (CSV with smiles/selfies, or .smi)")
    ap.add_argument("--gen", required=True, help="Generated/test molecules (CSV with smiles/selfies, or .smi)")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--outdir", default="/mnt/data")
    ap.add_argument("--prefix", default="fp")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    train_smiles = read_molecules(Path(args.train))
    gen_smiles   = read_molecules(Path(args.gen))

    train_mols = to_valid_mols(train_smiles)
    gen_mols   = to_valid_mols(gen_smiles)

    # fingerprints
    train_fps = [morgan_fp(m, args.radius, args.nbits) for m in train_mols]
    gen_fps   = [morgan_fp(m, args.radius, args.nbits) for m in gen_mols]

    # validity
    train_valid = sum(f is not None for f in train_fps)
    gen_valid   = sum(f is not None for f in gen_fps)

    # nearest-neighbor similarities
    nn, nn_idx = bulk_nn_tanimoto(gen_fps, train_fps, topk=args.topk)
    nn1 = nn[:,0]

    # coverage/novelty at common cutoffs
    cutoffs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    coverage = {c: float(np.mean(np.nan_to_num(nn1, nan=0.0) >= c)) for c in cutoffs}
    novelty  = {c: float(np.mean(np.nan_to_num(nn1, nan=0.0) < c))  for c in cutoffs}

    # internal diversity (generated set)
    idev = internal_diversity(gen_fps)

    # histogram
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    v = nn1[~np.isnan(nn1)]
    ax.hist(v, bins=40, alpha=0.8)
    ax.set_xlabel("Nearest-neighbor Tanimoto to TRAIN (Morgan R=%d, nbits=%d)" % (args.radius, args.nbits))
    ax.set_ylabel("Count")
    ax.set_title("Generated vs TRAIN similarity (valid gen=%d / train=%d)" % (gen_valid, train_valid))
    ax.grid(True)
    fig.tight_layout()
    hist_path = outdir / f"{args.prefix}_nn_tanimoto_hist.png"
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)

    # summary CSV
    import json
    summary = {
        "train_path": str(Path(args.train).resolve()),
        "gen_path": str(Path(args.gen).resolve()),
        "radius": args.radius, "nbits": args.nbits,
        "valid_train": int(train_valid), "valid_gen": int(gen_valid),
        "mean_nn": float(np.nanmean(nn1)), "median_nn": float(np.nanmedian(nn1)),
        "p90_nn": float(np.nanpercentile(nn1, 90)),
        "internal_diversity_gen": idev,
        "coverage_at_cutoffs": coverage,
        "novelty_at_cutoffs": novelty,
    }
    summ_path = outdir / f"{args.prefix}_similarity_summary.csv"
    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in summary.items():
            if isinstance(v, dict):
                w.writerow([k, json.dumps(v)])
            else:
                w.writerow([k, v])

    # optional: write top-k neighbors (indices & sims)
    nn_path = outdir / f"{args.prefix}_gen_top{args.topk}_nn.csv"
    with open(nn_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = ["gen_index", "nn_index_in_train"] + [f"sim_top{i+1}" for i in range(args.topk)]
        w.writerow(hdr)
        for i in range(len(gen_fps)):
            row = [i, int(nn_idx[i,0]) if not np.isnan(nn1[i]) else -1] + [nn[i,j] if j < nn.shape[1] else "" for j in range(args.topk)]
            w.writerow(row)

    print("Wrote:", hist_path)
    print("Wrote:", summ_path)
    print("Wrote:", nn_path)

if __name__ == "__main__":
    main()
