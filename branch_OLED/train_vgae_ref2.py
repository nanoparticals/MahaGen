#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_vgae_ref2_pbar.py - Multi-objective VGAE trainer with tqdm progress bars

import os, csv, json, math, argparse, random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# tqdm (progress bar) - graceful fallback if not installed
try:
    from tqdm.auto import tqdm
except Exception:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, dynamic_ncols=True):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def update(self, n=1): pass
        def set_postfix(self, **kwargs): pass
        def close(self): pass

from rdkit import Chem
from rdkit.Chem import rdchem

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_csv_rows(path: str):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))

def parse_smiles_column(rows, smiles_col, keep_header: bool):
    # 支持列名或索引
    if isinstance(smiles_col, str):
        assert keep_header, "--keep-header 必须配合列名使用"
        header = rows[0]
        name2idx = {n:i for i,n in enumerate(header)}
        assert smiles_col in name2idx, f"未找到列名：{smiles_col}"
        col = name2idx[smiles_col]
        data_rows = rows[1:]
    else:
        col = int(smiles_col)
        data_rows = rows[1:] if keep_header else rows
    smiles = [r[col] for r in data_rows if len(r)>col]
    return smiles

def read_labels_matrix(path: str, label_cols: str, keep_header: bool, by_name: bool):
    values = []; header = []
    rows = read_csv_rows(path)
    if not rows:
        return [], []
    if keep_header:
        header_row = rows[0]; data_rows = rows[1:]
    else:
        header_row = None; data_rows = rows
    if by_name:
        assert header_row is not None, "按列名选择时需要 --keep-header"
        name2idx = {name: i for i, name in enumerate(header_row)}
        cols = [name2idx[c.strip()] for c in label_cols.split(",") if c.strip() in name2idx]
        header = [header_row[i] for i in cols]
    else:
        cols = [int(x.strip()) for x in label_cols.split(",") if x.strip()!=""]
        header = [f"col{i}" for i in cols]
    for row in data_rows:
        row_vals = []
        for i in cols:
            try: row_vals.append(float(row[i]))
            except Exception: row_vals.append(float("nan"))
        values.append(row_vals)
    return values, header

# ---------------------------
# Graph construction (padded)
# ---------------------------
ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H,B,C,N,O,F,Si,P,S,Cl,Br,I
HYB_LIST = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2]

def one_hot(idx, choices):
    v = [0]*len(choices); unknown = 0
    try:
        j = choices.index(idx); v[j] = 1
    except ValueError:
        unknown = 1
    return v+[unknown]

def atom_features(atom: rdchem.Atom):
    z = atom.GetAtomicNum()
    feats = []
    feats += one_hot(z, ATOM_LIST)
    deg = atom.GetTotalDegree(); feats += one_hot(int(deg), list(range(6))) + [1 if deg>=6 else 0]
    chg = atom.GetFormalCharge(); feats += one_hot(int(chg), [-2,-1,0,1,2])
    feats += [1 if atom.GetIsAromatic() else 0]
    hyb = atom.GetHybridization(); feats += one_hot(hyb, HYB_LIST)
    feats += [atom.GetTotalNumHs(includeNeighbors=True)]
    return np.array(feats, dtype=np.float32)

NODE_FEATS_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))

def mol_to_padded_graph(mol: Chem.Mol, max_nodes: int):
    n = mol.GetNumAtoms()
    if n > max_nodes: return None
    X = np.zeros((max_nodes, NODE_FEATS_DIM), dtype=np.float32)
    A = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    M = np.zeros((max_nodes,), dtype=np.float32)
    C = np.full((max_nodes,), -1, dtype=np.int64)
    for i, atom in enumerate(mol.GetAtoms()):
        X[i] = atom_features(atom); M[i] = 1.0
        z = atom.GetAtomicNum()
        C[i] = ATOM_LIST.index(z) if z in ATOM_LIST else len(ATOM_LIST)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        A[i, j] = 1.0; A[j, i] = 1.0
    np.fill_diagonal(A, 0.0)
    return X, A, M, C

def build_dataset(smiles: List[str], max_nodes: int, labels: Optional[List[List[float]]] = None):
    Xs, As, Ms, Cs, kept, Y_kept = [], [], [], [], [], []
    it = zip(smiles, labels) if labels is not None else ((s, None) for s in smiles)
    for s, yrow in it:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        g = mol_to_padded_graph(mol, max_nodes)
        if g is None: continue
        if (labels is not None) and (yrow is not None) and any(v!=v for v in yrow):  # NaN
            continue
        X,A,M,C = g
        Xs.append(X); As.append(A); Ms.append(M); Cs.append(C); kept.append(s)
        if yrow is not None: Y_kept.append([float(v) for v in yrow])
    if not Xs:
        raise RuntimeError("No valid molecules after parsing/padding. Increase --max-nodes?")
    X = torch.tensor(np.stack(Xs), dtype=torch.float32)
    A = torch.tensor(np.stack(As), dtype=torch.float32)
    M = torch.tensor(np.stack(Ms), dtype=torch.float32)
    C = torch.tensor(np.stack(Cs), dtype=torch.long)
    if labels is None:
        return X, A, M, C, kept
    else:
        Y = torch.tensor(np.array(Y_kept, dtype=np.float32)) if len(Y_kept)>0 else None
        return X, A, M, C, kept, Y

# ---------------------------
# VGAE Model
# ---------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)
    def forward(self, x, a, mask):
        B, V, _ = x.shape
        I = torch.eye(V, device=x.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = a + I
        D = A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        D_inv_sqrt = D.pow(-0.5)
        A_norm = D_inv_sqrt * A_hat * D_inv_sqrt.transpose(1,2)
        h = torch.bmm(A_norm, x)
        h = self.lin(h)
        h = self.bn(h.view(-1, h.size(-1))).view(B, V, -1)
        h = F.relu(h)
        h = self.dropout(h)
        return h * mask.unsqueeze(-1)

class VGAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, layers=2, dropout=0.1):
        super().__init__()
        gcn = []; in_dim = x_dim
        for _ in range(layers):
            gcn.append(GCNLayer(in_dim, h_dim, dropout)); in_dim = h_dim
        self.gcn = nn.ModuleList(gcn)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
    def encode(self, x, a, m):
        h = x
        for layer in self.gcn:
            h = layer(h, a, m)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std)
        return mu + eps * std

class InnerProductDecoder(nn.Module):
    def __init__(self, tau=1.0): super().__init__(); self.tau = tau
    def forward(self, z): return torch.bmm(z, z.transpose(1, 2)) / self.tau

class AtomClassifier(nn.Module):
    def __init__(self, z_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, n_classes))
    def forward(self, z): return self.net(z)

# ---------------------------
# Training helpers
# ---------------------------
def bce_with_logits_masked(pred_logits, target, mask, posw_cap=10.0):
    B,V,_ = pred_logits.shape
    diag = torch.eye(V, device=pred_logits.device).unsqueeze(0)
    edge_mask = (1.0 - diag)
    node_mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
    eff_mask = edge_mask * node_mask
    logits = pred_logits[eff_mask.bool()]; tgt = target[eff_mask.bool()]
    pos = tgt.sum().item(); neg = tgt.numel() - pos
    if pos <= 0:
        pos_weight = torch.tensor(1.0, device=pred_logits.device)
    else:
        pw = min(posw_cap, max(1.0, float(neg)/float(pos)))
        pos_weight = torch.tensor(pw, device=pred_logits.device)
    return F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight)

def kl_anneal_weight(step, burnin, warmup, base_w):
    if step < burnin: return 0.0
    t = min(1.0, float(step - burnin) / max(1, warmup))
    return base_w * t

def masked_ce(logits, target_idx, mask):
    B,V,C = logits.shape
    logits = logits.view(B*V, C); target = target_idx.view(B*V)
    m = mask.view(B*V) > 0.5
    keep = (target>=0) & m
    if keep.sum()==0: return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits[keep], target[keep])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV 路径")
    ap.add_argument("--smiles-col", required=True, help="SMILES 列（索引或列名）")
    ap.add_argument("--keep-header", action="store_true", help="CSV 是否包含表头")
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--z-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-nodes", type=int, default=48)
    ap.add_argument("--recon-weight", type=float, default=3.0)
    ap.add_argument("--kl-weight", type=float, default=0.8)
    ap.add_argument("--kl-burnin", type=int, default=50)
    ap.add_argument("--kl-warmup", type=int, default=150)
    ap.add_argument("--atomcls-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save", type=str, default="ckpts/vgae_multi.pt")
    ap.add_argument("--log-csv", type=str, default="ckpts/vgae_train_log.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    # multi-target supervision
    ap.add_argument("--label-cols", type=str, default="",
                    help="多列性质（逗号分隔）。列名或索引；列名需 --keep-header")
    ap.add_argument("--target-weights", type=str, default="",
                    help="与 label-cols 对应的权重，逗号分隔；留空则均等权")
    ap.add_argument("--prop-weight", type=float, default=1.0,
                    help="多目标监督损失的权重（相对 recon/kl/atomcls）")
    ap.add_argument("--zscore", action="store_true",
                    help="对标签做 Z-score 标准化（用训练集 mean/std）")
    # progress bar
    ap.add_argument("--pbar", choices=["none","epoch","batch"], default="epoch",
                    help="进度条级别：none 不显示；epoch 每轮；batch 每小批")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    rows = read_csv_rows(args.input)
    try: smiles_col = int(args.smiles_col)
    except Exception: smiles_col = args.smiles_col
    smiles_raw = parse_smiles_column(rows, smiles_col, keep_header=args.keep_header)

    Y_list, label_names = None, []
    if args.label_cols:
        by_name = bool(args.keep_header)
        Y_list, label_names = read_labels_matrix(
            args.input, args.label_cols, keep_header=args.keep_header, by_name=by_name
        )

    if Y_list is None:
        X, A, M, C, SMILES = build_dataset(smiles_raw, args.max_nodes); Y = None
    else:
        X, A, M, C, SMILES, Y = build_dataset(smiles_raw, args.max_nodes, Y_list)

    device = torch.device(args.device)
    X = X.to(device); A = A.to(device); M = M.to(device); C = C.to(device)
    if Y is not None:
        Y = Y.to(device); T_dim = Y.size(1)
        if args.zscore:
            y_mean = Y.mean(dim=0, keepdim=True)
            y_std  = Y.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-8)
            Y_norm = (Y - y_mean) / y_std
        else:
            y_mean = torch.zeros(1, Y.size(1), device=device)
            y_std  = torch.ones(1,  Y.size(1), device=device)
            Y_norm = Y
        if args.target_weights:
            tw = [float(x.strip()) for x in args.target_weights.split(",") if x.strip()!=""]
            assert len(tw) == T_dim, "--target-weights 的数量需与 label-cols 一致"
            W = torch.tensor(tw, device=device).view(1, -1); W = W / W.sum()
        else:
            W = torch.ones(1, T_dim, device=device) / float(T_dim)
    else:
        T_dim = 0
        y_mean = torch.zeros(1, 0, device=device); y_std = torch.ones(1,  0, device=device)
        Y_norm = None; W = None

    vgae = VGAE(x_dim=NODE_FEATS_DIM, h_dim=args.hidden, z_dim=args.z_dim,
                layers=args.layers, dropout=args.dropout).to(device)
    decoder = InnerProductDecoder(tau=1.0).to(device)
    atom_head = AtomClassifier(z_dim=args.z_dim, n_classes=len(ATOM_LIST)+1).to(device)

    prop_heads = None
    if T_dim > 0:
        prop_heads = nn.ModuleList([nn.Linear(args.z_dim, 1).to(device) for _ in range(T_dim)])

    params = list(vgae.parameters()) + list(decoder.parameters()) + list(atom_head.parameters())
    if prop_heads is not None: params += list(prop_heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    N = X.size(0); idxs = np.arange(N)

    if "/" in args.log_csv: os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    with open(args.log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,loss,recon,kl,atom_ce,prop_mse,kl_w\n")

    # ------- progress bar loops -------
    epoch_iterable = range(1, args.epochs+1)
    if args.pbar == "epoch":
        epoch_iterable = tqdm(epoch_iterable, total=args.epochs, desc="Epoch", dynamic_ncols=True)
    for epoch in epoch_iterable:
        np.random.shuffle(idxs)
        vgae.train(); atom_head.train(); decoder.train()
        if prop_heads is not None:
            for h in prop_heads: h.train()

        total_loss = total_recon = total_kl = total_atom = total_prop = 0.0
        n_batches = 0

        batch_range = range(0, N, args.batch_size)
        if args.pbar == "batch":
            batch_range = tqdm(batch_range, total=math.ceil(N/args.batch_size),
                               desc=f"Epoch {epoch}", dynamic_ncols=True)

        for start in batch_range:
            end = min(N, start + args.batch_size)
            batch = idxs[start:end]
            x = X[batch]; a = A[batch]; m = M[batch]; c = C[batch]

            mu, logvar = vgae.encode(x, a, m)
            z = vgae.reparameterize(mu, logvar)

            m_sum = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            z_graph = (z * m.unsqueeze(-1)).sum(dim=1) / m_sum

            logits = decoder(z)
            recon = bce_with_logits_masked(logits, a, m)

            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl = (kl * m.unsqueeze(-1)).sum(dim=(1,2)) / m.sum(dim=1).clamp_min(1.0)
            kl = kl.mean()
            kl_w = kl_anneal_weight(epoch, args.kl_burnin, args.kl_warmup, args.kl_weight)

            atom_logits = atom_head(z)
            atom_ce = masked_ce(atom_logits, c, m)

            loss = args.recon_weight * recon + kl_w * kl + args.atomcls_weight * atom_ce

            prop_mse = torch.tensor(0.0, device=device)
            if (prop_heads is not None) and (Y_norm is not None):
                y_true = Y_norm[batch]
                y_pred = torch.cat([head(z_graph).view(-1,1) for head in prop_heads], dim=1)
                se = (y_pred - y_true).pow(2)
                prop_mse = (se * W).mean()
                loss = loss + args.prop_weight * prop_mse

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=2.0)
            opt.step()

            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kl.item())
            total_atom += float(atom_ce.item())
            total_prop += float(prop_mse.item())
            n_batches += 1

            if args.pbar == "batch":
                # show latest mini-batch stats
                try:
                    batch_range.set_postfix(recon=f"{recon.item():.3f}", kl=f"{kl.item():.3f}",
                                            atom=f"{atom_ce.item():.3f}", prop=f"{float(prop_mse.item()):.3f}")
                except Exception:
                    pass

        L = total_loss / n_batches
        R = total_recon / n_batches
        K = total_kl / n_batches
        A_ce = total_atom / n_batches
        P = total_prop / n_batches

        if args.pbar == "epoch":
            try:
                epoch_iterable.set_postfix(loss=f"{L:.3f}", recon=f"{R:.3f}",
                                           kl=f"{K:.3f}", prop=f"{P:.3f}")
            except Exception:
                pass

        with open(args.log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{L:.6f},{R:.6f},{K:.6f},{A_ce:.6f},{P:.6f},{kl_w:.4f}\n")

    # save
    if "/" in args.save: os.makedirs(os.path.dirname(args.save), exist_ok=True)
    meta = {
        "x_dim": NODE_FEATS_DIM, "hidden": args.hidden, "z_dim": args.z_dim,
        "layers": args.layers, "dropout": args.dropout, "max_nodes": args.max_nodes,
        "label_cols": label_names,
        "label_mean": (y_mean.detach().cpu().view(-1).tolist() if Y is not None else []),
        "label_std":  (y_std.detach().cpu().view(-1).tolist()  if Y is not None else []),
        "target_weights": (W.detach().cpu().view(-1).tolist() if W is not None else []),
        "zscore": bool(args.zscore),
        "recon_weight": args.recon_weight, "kl_weight": args.kl_weight,
        "kl_burnin": args.kl_burnin, "kl_warmup": args.kl_warmup,
        "atomcls_weight": args.atomcls_weight
    }
    torch.save({
        "state_dict": {
            "vgae": vgae.state_dict(),
            "decoder": decoder.state_dict(),
            "atom_head": atom_head.state_dict(),
            "prop_heads": (None if prop_heads is None else [h.state_dict() for h in prop_heads])
        },
        "meta": meta
    }, args.save)

    print(json.dumps({
        "n_samples": int(X.size(0)), "z_dim": int(args.z_dim),
        "label_cols": label_names, "save": args.save, "log_csv": args.log_csv
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
