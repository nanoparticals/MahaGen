
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_vgae_z_stats.py  ——  独立版（不依赖 vgae_density/z_generator_stub）
从一批 SMILES 计算 VGAE 潜变量 z 的经验分布，并导出：
- z_bank.npy : [N, z_dim]
- z_mean.npy : [z_dim]
- z_cov.npy  : [z_dim, z_dim]

关键修复：
- 从 ckpt 自动推断期望的 in_dim（第一层 GAT 线性层 weight 的第二维）
- 在编码前把节点特征 X 裁剪/零填充到该 in_dim，避免 35 vs 19 的维度不匹配

用法：
python make_vgae_z_stats.py \
  --smiles-csv /public/home/users/haoxw/generate_AI/branch_OLED/data/opv_selfies_valid.csv --smiles-col smiles \
  --vgae-ckpt /public/home/users/haoxw/generate_AI/branch_OLED/ckpts/vgae_opv_4props.pt \
  --outdir ckpts/z_stats --batch 512
"""
from __future__ import annotations
import os, argparse, csv
from typing import List, Optional

import numpy as np
from rdkit import Chem

import torch
from vgae_prior_GAT import VGAEPrior
from train_vgae_ref2 import build_dataset, mol_to_padded_graph   # 确保同目录

# ---------------- I/O ----------------
def read_smiles_from_csv(path: str, col: str, has_header: bool=True, delimiter: str=',') -> List[str]:
    smiles = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader)
            idx = None
            if isinstance(col, str):
                if col in header:
                    idx = header.index(col)
                else:
                    norm = [h.strip() for h in header]
                    if col.strip() in norm:
                        idx = norm.index(col.strip())
                    else:
                        lower = [h.strip().lower() for h in header]
                        if col.strip().lower() in lower:
                            idx = lower.index(col.strip().lower())
            else:
                idx = int(col)
            if idx is None:
                raise SystemExit(f"[ERR] CSV header 不包含列名: {col}；可用列: {header}")
        else:
            idx = int(col)
        for row in reader:
            if not row or idx >= len(row): continue
            s = row[idx].strip()
            if s: smiles.append(s)
    return smiles

def canon_smiles_list(smiles: List[str]) -> List[str]:
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try: out.append(Chem.MolToSmiles(m, canonical=True))
        except Exception: out.append(Chem.MolToSmiles(m))
    return out

# ------------- 维度推断 & 对齐 -------------
def ckpt_expected_in_dim(ckpt_path: str, fallback: int = 35) -> int:
    """
    从 ckpt state_dict 推断第一层 GAT 线性层输入维度（in_dim）。
    优先匹配包含 'gat1' 且以 'lin.weight' 结尾的键；找不到则回退 fallback。
    """
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if isinstance(sd, dict):
            cands = []
            for k, v in sd.items():
                if not hasattr(v, "shape"): continue
                kl = str(k).lower()
                if "gat1" in kl and kl.endswith("lin.weight"):
                    cands.append(v)
            if not cands:
                for k, v in sd.items():
                    if hasattr(v, "shape") and str(k).lower().endswith("lin.weight"):
                        cands.append(v)
            if cands:
                w = cands[0]
                return int(w.shape[1])  # [out_features, in_features]
    except Exception:
        pass
    return int(fallback)

def align_features(X: torch.Tensor, expected_in: int) -> torch.Tensor:
    """
    将节点特征 X 的最后一维裁剪/零填充到 expected_in。
    X: [B,N,F] 或 [1,N,F]
    """
    F = X.size(-1)
    if F == expected_in:
        return X
    if F > expected_in:
        return X[..., :expected_in]
    # F < expected_in: 右侧零填充
    pad = torch.zeros(*X.shape[:-1], expected_in - F, dtype=X.dtype, device=X.device)
    return torch.cat([X, pad], dim=-1)

# ------------- 加载模型 -------------
@torch.no_grad()
def load_vgae(ckpt: str, device: torch.device, max_nodes: int = 72) -> VGAEPrior:
    """
    加载 VGAE，并记录 ckpt 期望的 in_dim 到 model.expected_in_dim。
    其它结构超参按你训练命令：z_dim=64, hidden=128；heads_hidden/attn_dropout 与默认一致。
    """
    expected_in = ckpt_expected_in_dim(ckpt, fallback=35)  # 你的 ckpt 是 19
    model = VGAEPrior(z_dim=16, hidden=128, in_dim=expected_in, heads_hidden=4, attn_dropout=0.0)
    model.max_nodes = max_nodes
    model.expected_in_dim = expected_in
    model.load_from_ckpt(ckpt, map_location=device, strict='auto')
    model.to(device).eval()
    return model

# ------------- 编码 -------------
@torch.no_grad()
def encode_smiles_list(model: VGAEPrior, smiles: List[str], device: torch.device,
                       batch: int = 256, max_nodes: int = 72) -> np.ndarray:
    """
    优先用 build_dataset 批量构图；失败则逐条 mol_to_padded_graph。
    在前向前对 X 做 feature 对齐（裁剪/零填充到 ckpt 的 in_dim）。
    返回 [N, z_dim]
    """
    expected_in = getattr(model, "expected_in_dim", None)

    # 批量（更快）
    try:
        X, A, M, C, kept = build_dataset(smiles, max_nodes=max_nodes)
        if X is not None and len(kept) > 0:
            X = X.to(device); A = A.to(device); M = M.to(device)
            if expected_in is None: expected_in = X.size(-1)
            X = align_features(X, expected_in)
            z_graph, mu, logvar = model.encode_graph_level(X, A, node_mask=M, return_alpha=False)
            return z_graph.detach().cpu().numpy()
    except Exception:
        pass

    # 逐条（兜底）
    Z = []
    for s in smiles:
        g = mol_to_padded_graph(Chem.MolFromSmiles(s), max_nodes)
        if g is None: continue
        X, A, M, _ = g
        X = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
        A = torch.tensor(A, dtype=torch.float32, device=device).unsqueeze(0)
        M = torch.tensor(M, dtype=torch.float32, device=device).unsqueeze(0)
        if expected_in is None: expected_in = X.size(-1)
        X = align_features(X, expected_in)
        z_graph, mu, logvar = model.encode_graph_level(X, A, node_mask=M, return_alpha=False)
        Z.append(z_graph.squeeze(0).detach().cpu().numpy())
    if not Z:
        raise SystemExit("[ERR] 所有分子都无法构图；请检查 max_nodes 与特征维度设置是否与训练一致。")
    return np.vstack(Z)

# ------------- 主函数 -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smiles-csv', required=True)
    ap.add_argument('--smiles-col', default='smiles')
    ap.add_argument('--no-header', action='store_true')
    ap.add_argument('--delimiter', default=',')
    ap.add_argument('--vgae-ckpt', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--eps', type=float, default=1e-6)
    ap.add_argument('--max-nodes', type=int, default=72)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    smiles = read_smiles_from_csv(args.smiles_csv, args.smiles_col,
                                  has_header=(not args.no_header),
                                  delimiter=args.delimiter)
    smiles = canon_smiles_list(smiles)
    print(f"[info] loaded {len(smiles)} canonical SMILES")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_vgae(args.vgae_ckpt, device, max_nodes=args.max_nodes)
    Z = encode_smiles_list(model, smiles, device, batch=args.batch, max_nodes=args.max_nodes)

    np.save(os.path.join(args.outdir, 'z_bank.npy'), Z)
    z_mean = Z.mean(axis=0)
    z_cov  = np.cov(Z, rowvar=False) + np.eye(Z.shape[1]) * float(args.eps)

    np.save(os.path.join(args.outdir, 'z_mean.npy'), z_mean)
    np.save(os.path.join(args.outdir, 'z_cov.npy'),  z_cov)

    print(f"[done] z_bank.npy shape={Z.shape}")
    print(f"[done] z_mean.npy shape={z_mean.shape}")
    print(f"[done] z_cov.npy  shape={z_cov.shape}")

if __name__ == '__main__':
    main()
