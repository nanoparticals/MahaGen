# vgae_prior_GAT.py  ——  GCN 版本（接口对齐原 GAT 的 VGAEPrior）
from __future__ import annotations
from typing import Optional, Dict, Union, Tuple
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# =============== Utils ===============
def _as_float_adj(A: Tensor) -> Tensor:
    """A: (B,N,N) bool/float -> float(0/1)"""
    if A.dtype == torch.bool:
        return A.float()
    return (A > 0).float() if A.is_floating_point() else A.float()


def _build_mask_mat(node_mask: Optional[Tensor], N: int, device, dtype) -> Optional[Tensor]:
    """node_mask: (B,N) -> mask_mat: (B,N,N)"""
    if node_mask is None:
        return None
    nm = (node_mask > 0.5).to(dtype=dtype)
    return nm.unsqueeze(2) * nm.unsqueeze(1)  # (B,N,N)


def normalize_adj(A: Tensor, node_mask: Optional[Tensor] = None) -> Tensor:
    """
    归一化邻接：A_norm = D^{-1/2} (A + I) D^{-1/2}
    - 支持 A 已经带自环 / 不带自环（都会 clamp 到 0/1）
    - 对 padding 节点做裁剪：无连接、无自环
    """
    B, N, _ = A.shape
    device = A.device
    dtype = torch.float32 if not A.is_floating_point() else A.dtype

    Af = _as_float_adj(A).to(dtype=dtype)

    # padding 节点裁剪
    mask_mat = _build_mask_mat(node_mask, N, device, dtype)
    if mask_mat is not None:
        Af = Af * mask_mat

    # 只给真实节点加自环
    if node_mask is None:
        I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    else:
        nm = (node_mask > 0.5).to(dtype=dtype)
        I = torch.diag_embed(nm).to(dtype=dtype)  # (B,N,N)

    A_hat = (Af + I).clamp(0.0, 1.0)

    D = A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-6)     # (B,N,1)
    D_inv_sqrt = D.pow(-0.5)
    A_norm = D_inv_sqrt * A_hat * D_inv_sqrt.transpose(1, 2)  # (B,N,N)

    # 再次把 padding 行列清空（更稳）
    if mask_mat is not None:
        A_norm = A_norm * mask_mat
    return A_norm


# =============== GCN Encoder (same as train_vgae_ref2.py) ===============
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

        # cache for "alpha-like" output (we use A_norm as placeholder)
        self.last_A_norm: Optional[Tensor] = None  # (B,N,N)

    def forward(self, x: Tensor, A: Tensor, mask: Optional[Tensor]) -> Tensor:
        B, N, _ = x.shape
        A_norm = normalize_adj(A, mask)  # (B,N,N)
        self.last_A_norm = A_norm.detach()

        h = torch.bmm(A_norm, x)         # (B,N,F)
        h = self.lin(h)                  # (B,N,out)

        # BN over (B*N, out)
        h = self.bn(h.view(-1, h.size(-1))).view(B, N, -1)
        h = F.relu(h)
        h = self.dropout(h)

        if mask is not None:
            h = h * (mask > 0.5).float().unsqueeze(-1)
        return h


class VGAE(nn.Module):
    """
    Node-wise VGAE with GCN encoder (matching train_vgae_ref2.py):
      - encode(X,A,node_mask, return_alpha=False) -> (mu, logvar) or (mu, logvar, alphas_dict)
      - reparameterize(mu, logvar) -> Z
      - decode_adj(Z) -> Z Z^T (inner product)
    """
    def __init__(self, in_dim: int = 35, hidden_dim: int = 128, z_dim: int = 16,
                 layers: int = 2, dropout: float = 0.1,
                 logvar_min: float = -10.0, logvar_max: float = 10.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.z_dim = int(z_dim)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        gcn = []
        d = self.in_dim
        for _ in range(self.layers):
            gcn.append(GCNLayer(d, self.hidden_dim, dropout=self.dropout))
            d = self.hidden_dim
        self.gcn = nn.ModuleList(gcn)

        self.mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.logvar = nn.Linear(self.hidden_dim, self.z_dim)

    def encode(self, X: Tensor, A: Tensor, node_mask: Optional[Tensor] = None,
               return_alpha: bool = False):
        h = X
        last_A_norm = None
        for layer in self.gcn:
            h = layer(h, A, node_mask)
            last_A_norm = layer.last_A_norm

        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=self.logvar_min, max=self.logvar_max)

        if return_alpha:
            # 与 GAT 版本保持结构：返回 dict，并保证形状 (B,N,N)
            if last_A_norm is None:
                last_A_norm = normalize_adj(A, node_mask).detach()
            alphas: Dict[str, Tensor] = {"h": last_A_norm, "mu": last_A_norm, "logvar": last_A_norm}
            return mu, logvar, alphas
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    @staticmethod
    def decode_adj(Z: Tensor) -> Tensor:
        return torch.bmm(Z, Z.transpose(1, 2))


# =============== Optional pooling (same API as GAT wrapper) ===============
class AttnPool(nn.Module):
    """
    可学习图级读出：alpha_i = softmax(gate(z_i))，z_graph = Σ_i alpha_i z_i
    （保留接口，与 GAT 版本一致；默认不启用）
    """
    def __init__(self, z_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.Tanh(),
            nn.Linear(z_dim, 1)
        )

    def forward(self, Z: Tensor, node_mask: Optional[Tensor] = None):
        e = self.gate(Z).squeeze(-1)  # (B,N)
        if node_mask is not None:
            neg_inf = (~(node_mask > 0.5)).float() * 1e9
            e = e - neg_inf
        alpha = torch.softmax(e, dim=1)  # (B,N)
        if node_mask is not None:
            alpha = alpha * (node_mask > 0.5).float()
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-9)
        z_graph = (Z * alpha.unsqueeze(-1)).sum(dim=1)  # (B,D)
        return z_graph, alpha


# =============== Wrapper: VGAEPrior (API-compatible) ===============
class VGAEPrior(nn.Module):
    """
    与你项目原 GAT 版 VGAEPrior 对齐：
      - sample_z(batch_size, n_nodes, device) -> (B,n_nodes,z_dim)
      - pool_graph_z(mu, mask) -> (B,z_dim)
      - encode_graph_level(X,A,node_mask,return_alpha)
      - load_from_ckpt(ckpt_path, strict='auto') 兼容 train_vgae_ref2.py ckpt
    额外增强：
      - load_from_ckpt 会根据 ckpt 的 meta/权重形状自动重建 encoder，以避免 RL 端占位参数导致 shape mismatch
    """
    def __init__(self,
                 z_dim: int = 16,
                 max_nodes: int = 72,
                 hidden: int = 128,
                 in_dim: int = 35,
                 layers: int = 2,
                 dropout: float = 0.1,
                 use_attn_pool: bool = False,
                 **kwargs):
        super().__init__()
        self.vgae = VGAE(in_dim=in_dim, hidden_dim=hidden, z_dim=z_dim, layers=layers, dropout=dropout)
        self.z_dim = int(z_dim)
        self.max_nodes = int(max_nodes)
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)

        self.use_attn_pool = bool(use_attn_pool)
        self.attn_pool = AttnPool(self.z_dim) if self.use_attn_pool else None

        # 给 stub/其它脚本对齐特征维度用（可选）
        self.expected_in_dim = self.in_dim

    # ------- 抽样与读出 -------
    def sample_z(self, batch_size: int = 1, n_nodes: int = 1, device=None) -> Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(batch_size, n_nodes, self.z_dim, device=device)

    @staticmethod
    def pool_graph_z(mu: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            return mu.mean(1)
        denom = mask.sum(1, keepdim=True).clamp_min(1.0)
        return (mu * mask.unsqueeze(-1)).sum(1) / denom

    def calculate_euclidean_distance(self, mu_1: Tensor, mu_2: Tensor) -> Tensor:
        diff = mu_1 - mu_2
        return torch.norm(diff, p=2, dim=-1)

    def encode_graph_level(self, X: Tensor, A: Tensor,
                           node_mask: Optional[Tensor] = None,
                           return_alpha: bool = False):
        """
        与原 GAT 版一致：
          return_alpha=False:  (z_graph, mu, logvar)
          return_alpha=True :  (z_graph, mu, logvar, alphas_dict)
        """
        if return_alpha:
            mu, logvar, alphas = self.vgae.encode(X, A, node_mask=node_mask, return_alpha=True)
        else:
            mu, logvar = self.vgae.encode(X, A, node_mask=node_mask, return_alpha=False)
            alphas = None

        if self.attn_pool is None:
            z_graph = self.pool_graph_z(mu, node_mask)
            alpha_pool = None
        else:
            z_graph, alpha_pool = self.attn_pool(mu, node_mask)

        if return_alpha:
            if alpha_pool is not None and isinstance(alphas, dict):
                alphas = {**alphas, "pool": alpha_pool}
            return z_graph, mu, logvar, alphas
        return z_graph, mu, logvar

    # ------- ckpt 解析与自适应重建 -------
    @staticmethod
    def _infer_arch_from_vgae_sd(vgae_sd: Dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        从 vgae state_dict 推断 (in_dim, hidden, z_dim, layers)
        """
        in_dim = hidden = z_dim = layers = None

        # gcn.0.lin.weight: [hidden, in_dim]
        for k, v in vgae_sd.items():
            if not hasattr(v, "shape"):
                continue
            if re.match(r"^gcn\.\d+\.lin\.weight$", str(k)):
                shp = tuple(v.shape)
                if len(shp) == 2:
                    hidden = int(shp[0])
                    in_dim = int(shp[1])
                break

        # mu.weight: [z_dim, hidden]
        for k, v in vgae_sd.items():
            if not hasattr(v, "shape"):
                continue
            if str(k) == "mu.weight":
                shp = tuple(v.shape)
                if len(shp) == 2:
                    z_dim = int(shp[0])
                    # hidden 也可从这里校验
                break

        # layers: max index in gcn.<i>.*
        idxs = []
        for k in vgae_sd.keys():
            m = re.match(r"^gcn\.(\d+)\.", str(k))
            if m:
                idxs.append(int(m.group(1)))
        if idxs:
            layers = int(max(idxs) + 1)

        return in_dim, hidden, z_dim, layers

    def _rebuild(self, in_dim: int, hidden: int, z_dim: int, layers: int,
                 dropout: Optional[float] = None):
        """
        重建 encoder，解决 RL 端占位参数初始化导致的 shape mismatch。
        """
        if dropout is None:
            dropout = self.dropout
        self.vgae = VGAE(in_dim=in_dim, hidden_dim=hidden, z_dim=z_dim,
                         layers=layers, dropout=float(dropout))
        self.z_dim = int(z_dim)
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.expected_in_dim = self.in_dim
        if self.use_attn_pool:
            self.attn_pool = AttnPool(self.z_dim)
        else:
            self.attn_pool = None

    def load_from_ckpt(self, ckpt: Union[str, Dict], map_location: Union[str, torch.device] = "cpu",
                       strict: Union[bool, str] = "auto"):
        """
        兼容 train_vgae_ref2.py 的保存格式：
          {
            "state_dict": {"vgae": <dict>, ...},
            "meta": {"x_dim":..., "hidden":..., "z_dim":..., "layers":..., "dropout":..., "max_nodes":...}
          }
        也兼容平铺键名：{"vgae.gcn.0.lin.weight":..., ...}
        strict: True/False/'auto'（'auto' 会先 strict=True，失败后自动重建并 strict=False）
        """
        blob = torch.load(ckpt, map_location=map_location) if isinstance(ckpt, str) else ckpt
        meta = blob.get("meta", {}) if isinstance(blob, dict) else {}

        state_like = blob.get("state_dict", blob) if isinstance(blob, dict) else blob

        # 1) 取出 vgae 的 state_dict
        vgae_sd: Dict = {}
        if isinstance(state_like, dict) and "vgae" in state_like and isinstance(state_like["vgae"], dict):
            vgae_sd = state_like["vgae"]
        elif isinstance(state_like, dict):
            # 平铺键名 vgae.* -> 去前缀
            for k, v in state_like.items():
                if isinstance(k, str) and k.startswith("vgae."):
                    vgae_sd[k[len("vgae."):]] = v

        if not vgae_sd and isinstance(state_like, dict):
            # 最后兜底：可能直接就是 vgae 的 sd
            vgae_sd = state_like

        if not isinstance(vgae_sd, dict) or len(vgae_sd) == 0:
            raise RuntimeError("无法从 ckpt 中提取 vgae state_dict（请检查 ckpt 格式）")

        # 2) 推断/读取结构超参
        in_dim_m = meta.get("x_dim", None)
        hidden_m = meta.get("hidden", None)
        z_dim_m = meta.get("z_dim", None)
        layers_m = meta.get("layers", None)
        dropout_m = meta.get("dropout", None)
        max_nodes_m = meta.get("max_nodes", None)

        in_dim_i, hidden_i, z_dim_i, layers_i = self._infer_arch_from_vgae_sd(vgae_sd)

        in_dim = int(in_dim_m) if in_dim_m is not None else int(in_dim_i) if in_dim_i is not None else self.in_dim
        hidden = int(hidden_m) if hidden_m is not None else int(hidden_i) if hidden_i is not None else self.hidden
        z_dim = int(z_dim_m) if z_dim_m is not None else int(z_dim_i) if z_dim_i is not None else self.z_dim
        layers = int(layers_m) if layers_m is not None else int(layers_i) if layers_i is not None else self.layers
        dropout = float(dropout_m) if dropout_m is not None else self.dropout

        if max_nodes_m is not None:
            self.max_nodes = int(max_nodes_m)

        # 3) 如当前实例结构不匹配，则重建
        need_rebuild = False
        try:
            # 通过关键权重形状判断（最可靠）
            w0 = vgae_sd.get("gcn.0.lin.weight", None)
            wmu = vgae_sd.get("mu.weight", None)
            if hasattr(w0, "shape") and tuple(w0.shape) != tuple(self.vgae.gcn[0].lin.weight.shape):
                need_rebuild = True
            if hasattr(wmu, "shape") and tuple(wmu.shape) != tuple(self.vgae.mu.weight.shape):
                need_rebuild = True
        except Exception:
            need_rebuild = True

        if need_rebuild:
            self._rebuild(in_dim=in_dim, hidden=hidden, z_dim=z_dim, layers=layers, dropout=dropout)

        # 4) 加载权重
        if strict == "auto":
            try:
                self.vgae.load_state_dict(vgae_sd, strict=True)
            except Exception:
                # 重试：非严格（此时结构已对齐，通常不会再有 size mismatch）
                self.vgae.load_state_dict(vgae_sd, strict=False)
        else:
            self.vgae.load_state_dict(vgae_sd, strict=bool(strict))

        return self
