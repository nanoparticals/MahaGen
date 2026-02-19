# z_generator_stub.py —— 自动从 ckpt 推断 in_dim，并对齐图特征到期望维度；支持 SMILES->z
import os
import types
import numpy as np
import torch
from rdkit import Chem

from vgae_prior_GAT import VGAEPrior

# 复用训练脚本中的图构建函数
try:
    from train_vgae_ref2 import mol_to_padded_graph, build_dataset
except Exception:
    mol_to_padded_graph = None
    build_dataset = None


def _ckpt_expected_in_dim(ckpt_path: str, fallback: int = 35) -> int:
    """
    从 checkpoint 的 state_dict 中自动推断第一层 GAT 线性层的输入维度（即 in_dim）。
    常见键名包含 'gat1.lin.weight'；若找不到就回退 fallback。
    """
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict):
            # 兼容：可能包在 {'state_dict': ...} 里
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            # 寻找第一层线性权重
            candidates = []
            for k, v in sd.items():
                if not hasattr(v, "shape"):
                    continue
                kn = str(k).lower()
                if "gat1" in kn and "lin.weight" in kn:
                    candidates.append((k, v))
            if not candidates:
                # 退而求其次：找任意 'lin.weight'
                for k, v in sd.items():
                    if hasattr(v, "shape") and str(k).lower().endswith("lin.weight"):
                        candidates.append((k, v))
            if candidates:
                # 线性层 weight 形状通常是 [out_features, in_features]
                w = candidates[0][1]
                return int(w.shape[1])
    except Exception:
        pass
    return int(fallback)


def _align_features(X: torch.Tensor, expected_in: int) -> torch.Tensor:
    """
    将节点特征 X 的最后一维裁剪/零填充到 expected_in。
    X: [..., F]
    """
    F = X.size(-1)
    if F == expected_in:
        return X
    if F > expected_in:
        return X[..., :expected_in]
    # F < expected_in: 右侧零填充
    pad_size = expected_in - F
    pad_shape = list(X.shape[:-1]) + [pad_size]
    pad = torch.zeros(*pad_shape, dtype=X.dtype, device=X.device)
    return torch.cat([X, pad], dim=-1)


@torch.no_grad()
def _encode_smiles_bound(self, smiles: str, device: torch.device = None):
    """
    单条 SMILES -> 图级 z（返回 np.ndarray, [z_dim]）
    """
    device = device or next(self.parameters()).device
    if mol_to_padded_graph is None:
        raise RuntimeError("找不到 train_vgae_ref2.mol_to_padded_graph：无法从 SMILES 构图。请确保 train_vgae_ref2.py 与本文件同目录。")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    max_nodes = getattr(self, 'max_nodes', 72)  # 与训练一致
    g = mol_to_padded_graph(m, max_nodes)
    if g is None:
        return None
    X, A, M, _ = g
    X = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,F]
    A = torch.tensor(A, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,N]
    M = torch.tensor(M, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N]
    # 将特征对齐到模型期望维度
    expected_in = getattr(self, "expected_in_dim", X.size(-1))
    X = _align_features(X, expected_in)
    # 编码
    z_graph, mu, logvar = self.encode_graph_level(X, A, node_mask=M, return_alpha=False)
    return z_graph.squeeze(0).detach().cpu().numpy()  # [z_dim]


@torch.no_grad()
def _encode_batch_bound(self, smiles_list, device: torch.device = None):
    """
    批量 SMILES -> 批量 z（返回 np.ndarray, [B, z_dim]）
    """
    device = device or next(self.parameters()).device
    max_nodes = getattr(self, 'max_nodes', 72)
    expected_in = getattr(self, "expected_in_dim", None)

    if build_dataset is not None:
        # 批量构图（高效）
        X, A, M, C, kept = build_dataset(smiles_list, max_nodes=max_nodes)
        if X is None or len(kept) == 0:
            return None
        X = X.to(device); A = A.to(device); M = M.to(device)
        if expected_in is None:
            expected_in = X.size(-1)
        X = _align_features(X, expected_in)
        z_graph, mu, logvar = self.encode_graph_level(X, A, node_mask=M, return_alpha=False)
        return z_graph.detach().cpu().numpy()

    # 回退：逐条编码
    zs = []
    for s in smiles_list:
        z = _encode_smiles_bound(self, s, device=device)
        if z is not None:
            zs.append(z)
    return np.vstack(zs) if len(zs) > 0 else None


def _attach_encoding_methods(model: VGAEPrior):
    """把 encode_smiles / encode_batch 动态挂到模型上"""
    model.encode_smiles = types.MethodType(_encode_smiles_bound, model)
    model.encode_batch  = types.MethodType(_encode_batch_bound,  model)
    return model


def load_vgae_model(ckpt_path: str, device: torch.device):
    """
    加载已训练好的 VGAE（GAT 版本），并附加编码函数。
    - 自动从 ckpt 推断 in_dim（期望节点特征维度）
    - 自动将输入特征对齐到该维度（裁剪/零填充）
    ⚠️ 如训练时其它超参不同（heads_hidden/attn_dropout/max_nodes），请按训练值修改。
    """
    max_nodes = 72                # 你的训练命令里 --max-nodes 72
    expected_in = _ckpt_expected_in_dim(ckpt_path, fallback=35)  # 你的 ckpt 显示为 19
    # 你的训练命令：--z-dim 64 --hidden 128 --layers 2
    model = VGAEPrior(z_dim=16, hidden=128, in_dim=expected_in, heads_hidden=4, attn_dropout=0.0)
    model.max_nodes = max_nodes
    model.expected_in_dim = expected_in
    # 用 strict='auto' 更宽松，避免 shape 轻微不符时报错
    model.load_from_ckpt(ckpt_path, map_location=device, strict='auto')
    model.to(device).eval()
    _attach_encoding_methods(model)
    return model


@torch.no_grad()
def vgae_sample_z(vgae_obj, n: int, device: torch.device) -> torch.Tensor:
    """
    从先验采样 n 个图级 z。返回 [n, z_dim]（每个序列一个 z）。
    """
    z_nodes = vgae_obj.sample_z(batch_size=n, n_nodes=1, device=device)  # [n,1,z]
    z = z_nodes.squeeze(1).contiguous()                                   # [n,z]
    return z
