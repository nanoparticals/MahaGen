#!/usr/bin/env python3
# make_z_bank.py
'''python make_z_bank.py \
     --smiles-csv /public/home/users/haoxw/generate_AI/branch_Li/data/Li_iron_selfies.csv \
     --smiles-col 0 \
     --selfies-col 1 \
     --keep-header \
     --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_Li/ckpts/vgae_Li_iron_4props.pt \
     --outdir /public/home/users/haoxw/generate_AI/branch_Li/ckpts/z_stats_gen \
     --out-valid-csv /public/home/users/haoxw/generate_AI/branch_Li/data/Li_selfies_valid.csv \
     --device cuda \
     --max-nodes 72
'''
import argparse, csv, os, random, sys
import numpy as np
import torch
from rdkit import Chem

# adapt import to your repo: VGAEPrior implementation used earlier
from vgae_prior_GAT import VGAEPrior   # ensure this import works

def read_smiles(csv_path, col=0, keep_header=False):
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        if keep_header:
            next(r)
        for row in r:
            if not row: continue
            s = (row[col] or "").strip()
            if not s: continue
            yield s

def smiles_to_graph_batch(smiles_list, max_nodes, atom_feat_fn):
    # Minimal packaging consistent with vgae.encode_graph_level(X,A,mask)
    # This depends on your VGAE implementation input format: adjust if needed.
    import torch
    batch = len(smiles_list)
    X = torch.zeros((batch, max_nodes, 19), dtype=torch.float32)
    A = torch.zeros((batch, max_nodes, max_nodes), dtype=torch.bool)
    mask = torch.zeros((batch, max_nodes), dtype=torch.bool)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None: 
            continue
        N = m.GetNumAtoms()
        for a_idx, a in enumerate(m.GetAtoms()):
            X[i, a_idx] = torch.tensor(atom_feat_fn(a), dtype=torch.float32)
        for b in m.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            A[i, u, v] = True; A[i, v, u] = True
        for j in range(N):
            A[i, j, j] = True
        mask[i, :N] = True
    return X, A, mask

# Reuse small atom feature function with 19-dim as vgae expects in your repo
_ATOMS = [1,5,6,7,8,9,14,15,16,17,35,53]
_IDX = {z:i for i,z in enumerate(_ATOMS)}
def atom_feat(a):
    z = a.GetAtomicNum()
    onehot = np.zeros(len(_ATOMS), dtype=np.float32)
    if z in _IDX:
        onehot[_IDX[z]] = 1.0
    arom = float(a.GetIsAromatic())
    deg = float(a.GetDegree()) / 4.0
    val = float(a.GetTotalValence()) / 6.0
    charge = float(a.GetFormalCharge()) / 3.0
    hyb = float(int(a.GetHybridization())) / 6.0
    ring = float(a.IsInRing())
    return np.concatenate([onehot, np.array([arom, deg, val, charge, hyb, ring, 0.0], dtype=np.float32)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-csv", required=True)
    ap.add_argument("--smiles-col", type=int, default=0)
    ap.add_argument("--keep-header", action="store_true")
    ap.add_argument("--vgae-ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-samples", type=int, default=500000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-nodes", type=int, default=72)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    print("Loading VGAE ckpt:", args.vgae_ckpt)
    # instantiate model with placeholders if needed - rely on ckpt metadata
    vgae = VGAEPrior(z_dim=16, hidden=128, in_dim=19)  # adjust z_dim if you know
    vgae.load_from_ckpt(args.vgae_ckpt, map_location=device, strict='auto')
    vgae.to(device).eval()

    smiles_gen = read_smiles(args.smiles_csv, args.smiles_col, args.keep_header)
    z_list = []
    batch_smis = []
    cnt = 0
    for smi in smiles_gen:
        batch_smis.append(smi)
        if len(batch_smis) >= args.batch:
            X, A, mask = smiles_to_graph_batch(batch_smis, args.max_nodes, atom_feat)
            X = X.to(device); A = A.to(device); mask = mask.to(device)
            with torch.no_grad():
                enc = vgae.encode_graph_level(X, A, mask)  # adapt to your API
                # enc may be (mu, logvar) or mu; ensure shape [B,z]
                if isinstance(enc, tuple):
                    mu = enc[0]
                else:
                    mu = enc
                mu = mu.detach().cpu().numpy()
            for z in mu:
                z_list.append(z.astype(np.float64))
                cnt += 1
                if cnt >= args.max_samples:
                    break
            batch_smis = []
            if cnt >= args.max_samples:
                break

    # leftover
    if batch_smis and cnt < args.max_samples:
        X, A, mask = smiles_to_graph_batch(batch_smis, args.max_nodes, atom_feat)
        X = X.to(device); A = A.to(device); mask = mask.to(device)
        with torch.no_grad():
            enc = vgae.encode_graph_level(X, A, mask)
            mu = enc[0] if isinstance(enc, tuple) else enc
            mu = mu.detach().cpu().numpy()
        for z in mu:
            if cnt >= args.max_samples: break
            z_list.append(z.astype(np.float64)); cnt += 1

    Z = np.vstack(z_list)
    print("Collected Z shape:", Z.shape)
    # compute stats
    mean = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False, bias=False)  # shape (D,D)
    np.save(os.path.join(args.outdir, "z_bank.npy"), Z)
    np.save(os.path.join(args.outdir, "z_mean.npy"), mean)
    np.save(os.path.join(args.outdir, "z_cov.npy"), cov)
    print("Saved z_bank,z_mean,z_cov to", args.outdir)

if __name__ == "__main__":
    main()
