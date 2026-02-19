#!/usr/bin/env python3
# make_z_bank.py
"""
生成 VGAE 的 z_bank，同时输出与 z_bank 严格行对齐的 valid 两列 CSV（smiles,selfies）。

示例：
python make_z_bank.py \
  --smiles-csv /public/home/users/haoxw/generate_AI/branch_OLED/data/opv_selfies.csv \
  --smiles-col 0 \
  --selfies-col 1 \
  --keep-header \
  --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/vgae_opv_4props.pt \
  --outdir /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/z_stats_gen \
  --out-valid-csv /public/home/users/haoxw/generate_AI/branch_OLED/data/opv_selfies_valid.csv \
  --device cuda \
  --max-nodes 72

"""
import argparse, csv, os, random
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdchem

from vgae_prior_GAT import VGAEPrior


def read_pairs(csv_path, smiles_col=0, selfies_col=1, keep_header=False):
    """从CSV读取 (smiles, selfies) 对。keep_header=True 表示文件有表头，需要跳过首行。"""
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        if keep_header:
            next(r, None)
        for row in r:
            if not row:
                continue
            if smiles_col >= len(row):
                continue
            smi = (row[smiles_col] or "").strip()
            if not smi:
                continue
            sf_str = ""
            if selfies_col is not None and selfies_col < len(row):
                sf_str = (row[selfies_col] or "").strip()
            yield smi, sf_str


# =========================
# 35-dim atom features (match train_vgae_ref2-style)
# =========================
ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H,B,C,N,O,F,Si,P,S,Cl,Br,I
DEG_LIST = list(range(6))                                # 0..5 (+ extra "deg>=6")
CHG_LIST = [-2, -1, 0, 1, 2]                             # (+ unknown)
HYB_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]


def _one_hot(val, choices):
    """one-hot over choices + 1 unknown bit"""
    v = np.zeros(len(choices) + 1, dtype=np.float32)
    try:
        j = choices.index(val)
        v[j] = 1.0
    except ValueError:
        v[-1] = 1.0
    return v


def atom_feat_35(a: rdchem.Atom) -> np.ndarray:
    # element (12 + unk) = 13
    f_elem = _one_hot(a.GetAtomicNum(), ATOM_LIST)  # 13

    # total degree (0..5 + unk) + (deg>=6) = 7 + 1 = 8
    deg = int(a.GetTotalDegree())
    f_deg = _one_hot(min(deg, 5) if deg <= 5 else 999, DEG_LIST)  # 7
    f_deg_ge6 = np.array([1.0 if deg >= 6 else 0.0], dtype=np.float32)  # 1

    # formal charge (-2..2 + unk) = 6
    f_chg = _one_hot(int(a.GetFormalCharge()), CHG_LIST)  # 6

    # aromatic = 1
    f_arom = np.array([1.0 if a.GetIsAromatic() else 0.0], dtype=np.float32)  # 1

    # hybridization (5 + unk) = 6
    f_hyb = _one_hot(a.GetHybridization(), HYB_LIST)  # 6

    # total H = 1
    f_h = np.array([float(a.GetTotalNumHs(includeNeighbors=True))], dtype=np.float32)  # 1

    feat = np.concatenate([f_elem, f_deg, f_deg_ge6, f_chg, f_arom, f_hyb, f_h], axis=0).astype(np.float32)
    if feat.shape[0] != 35:
        raise RuntimeError(f"atom_feat_35 dim mismatch: {feat.shape[0]}")
    return feat


def pairs_to_graph_batch(pairs, max_nodes):
    """
    输入: pairs = [(smiles, selfies), ...]
    输出: X, A, mask, kept_smiles, kept_selfies
    自动跳过：RDKit 失败、N==0、N>max_nodes、原子特征异常等
    """
    kept_smiles, kept_selfies = [], []
    Xs, As, Ms = [], [], []

    for smi, sf_str in pairs:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue

        # canonicalize smiles（用于构图/编码）
        smi_can = Chem.MolToSmiles(m, canonical=True)
        m = Chem.MolFromSmiles(smi_can)
        if m is None:
            continue

        N = m.GetNumAtoms()
        if N == 0 or N > max_nodes:
            continue

        X = torch.zeros((max_nodes, 35), dtype=torch.float32)
        A = torch.zeros((max_nodes, max_nodes), dtype=torch.bool)
        mask = torch.zeros((max_nodes,), dtype=torch.bool)

        try:
            for a_idx, a in enumerate(m.GetAtoms()):
                X[a_idx] = torch.from_numpy(atom_feat_35(a))
        except Exception:
            continue

        for b in m.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u < max_nodes and v < max_nodes:
                A[u, v] = True
                A[v, u] = True

        for j in range(N):
            A[j, j] = True

        mask[:N] = True

        kept_smiles.append(smi_can)
        kept_selfies.append(sf_str)  # ✅ 保留原 SELFIES（保证与原训练序列一致）
        Xs.append(X); As.append(A); Ms.append(mask)

    if not kept_smiles:
        return None, None, None, [], []

    Xb = torch.stack(Xs, dim=0)  # (B,N,35)
    Ab = torch.stack(As, dim=0)  # (B,N,N)
    Mb = torch.stack(Ms, dim=0)  # (B,N)
    return Xb, Ab, Mb, kept_smiles, kept_selfies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-csv", required=True)
    ap.add_argument("--smiles-col", type=int, default=0)
    ap.add_argument("--selfies-col", type=int, default=1, help="SELFIES列索引（默认1）。若输入没有SELFIES列，可设为-1")
    ap.add_argument("--keep-header", action="store_true")
    ap.add_argument("--vgae-ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-valid-csv", default="", help="输出valid两列CSV（smiles,selfies），与z_bank严格对齐")
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

    vgae = VGAEPrior(z_dim=16, hidden=128, in_dim=35)
    vgae.load_from_ckpt(args.vgae_ckpt, map_location=device, strict="auto")
    vgae.to(device).eval()

    selfies_col = None if args.selfies_col < 0 else args.selfies_col
    pair_gen = read_pairs(args.smiles_csv, args.smiles_col, selfies_col, args.keep_header)

    z_list = []
    kept_smiles_all = []
    kept_selfies_all = []

    cnt = 0
    n_read = 0
    n_kept_graph = 0

    batch_pairs = []
    for smi, sf_str in pair_gen:
        n_read += 1
        batch_pairs.append((smi, sf_str))

        if len(batch_pairs) >= args.batch:
            X, A, mask, kept_smiles, kept_selfies = pairs_to_graph_batch(batch_pairs, args.max_nodes)
            batch_pairs = []

            if not kept_smiles:
                continue

            n_kept_graph += len(kept_smiles)

            X = X.to(device)
            A = A.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                enc = vgae.encode_graph_level(X, A, mask)  # returns (z_graph, mu, logvar, ...)
                z_graph = enc[0] if isinstance(enc, tuple) else enc
                z_np = z_graph.detach().cpu().numpy()  # (B, z_dim)

            # ✅ 逐条写入，保证与 kept_smiles/kept_selfies 完全对齐
            for j, z in enumerate(z_np):
                if cnt >= args.max_samples:
                    break
                z_list.append(z.astype(np.float64))
                kept_smiles_all.append(kept_smiles[j])
                kept_selfies_all.append(kept_selfies[j])
                cnt += 1

            if cnt >= args.max_samples:
                break

    # leftover
    if batch_pairs and cnt < args.max_samples:
        X, A, mask, kept_smiles, kept_selfies = pairs_to_graph_batch(batch_pairs, args.max_nodes)
        if kept_smiles:
            n_kept_graph += len(kept_smiles)
            X = X.to(device); A = A.to(device); mask = mask.to(device)
            with torch.no_grad():
                enc = vgae.encode_graph_level(X, A, mask)
                z_graph = enc[0] if isinstance(enc, tuple) else enc
                z_np = z_graph.detach().cpu().numpy()
            for j, z in enumerate(z_np):
                if cnt >= args.max_samples:
                    break
                z_list.append(z.astype(np.float64))
                kept_smiles_all.append(kept_smiles[j])
                kept_selfies_all.append(kept_selfies[j])
                cnt += 1

    if not z_list:
        raise RuntimeError("No valid molecules were encoded. Check input smiles and max_nodes.")

    Z = np.vstack(z_list)
    print(f"[stats] read_smiles={n_read} kept_graph={n_kept_graph} encoded={Z.shape[0]}")
    print("Collected Z shape:", Z.shape)

    mean = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False, bias=False)

    np.save(os.path.join(args.outdir, "z_bank.npy"), Z)
    np.save(os.path.join(args.outdir, "z_mean.npy"), mean)
    np.save(os.path.join(args.outdir, "z_cov.npy"), cov)
    print("Saved z_bank,z_mean,z_cov to", args.outdir)

    # ✅ 输出严格对齐的 valid CSV（smiles,selfies）
    out_valid = args.out_valid_csv.strip()
    if not out_valid:
        out_valid = os.path.join(args.outdir, "valid_smiles_selfies.csv")
    os.makedirs(os.path.dirname(out_valid) or ".", exist_ok=True)
    with open(out_valid, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "selfies"])
        for smi, sf_str in zip(kept_smiles_all, kept_selfies_all):
            w.writerow([smi, sf_str])
    print("[OK] wrote valid aligned CSV:", out_valid, f"(rows={len(kept_smiles_all)})")


if __name__ == "__main__":
    main()
