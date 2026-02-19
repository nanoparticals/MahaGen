#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-objective RL trainer (SELFIES)
— unified gate metric (euclid/maha) + teacher coverage calibration
— teacher-quantile tighten on top of coverage
— writes gate_stats.npz for reward-time unification

采样 / 过滤逻辑扩展为三种方案，通过 --sample-mode 切换：
  1) last_batch  —— 方案一：采样阶段完全复用最近一次训练 batch（含 pulse 教师）
  2) buffer      —— 方案二：从训练过程缓存的序列池中采样
  3) new         —— 方案三：保持原始逻辑，每次采样重新 rollout 若干条（默认）
"""
from __future__ import annotations

import os, csv, math, argparse, time, random, re
from typing import Optional, Tuple, List, Sequence, Dict, Any, Union
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
import pandas as pd

# ----- Optional imports (handled gracefully) -----
_have_rdkit = True
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    _have_rdkit = False

_have_selfies = True
try:
    import selfies as sf
except Exception:
    _have_selfies = False

    class sf:  # fallback shim
        @staticmethod
        def decoder(x: str) -> str:
            return x


# ====== project modules ======
from data import GeneratorData
from stackRNN import StackAugmentedRNN
from reward_multiobjective import MultiObjectiveReward, MultiObjConfig


# ---------------- helpers ----------------
def _normalize_transform_targets(val: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, str):
        items = [tok.strip() for tok in re.split(r"[,;]", val) if tok.strip()]
        return items or None
    result: List[str] = []
    for item in val:
        if item is None:
            continue
        if isinstance(item, str):
            parts = [tok.strip() for tok in re.split(r"[,;]", item) if tok.strip()]
            if parts:
                result.extend(parts)
        else:
            result.append(str(item))
    return result or None


# ====== optional VGAE stubs for z sampling ======
try:
    from z_generator_stub import load_vgae_model as _stub_load_vgae_model, vgae_sample_z as _stub_vgae_sample_z

    _HAVE_STUB = True
except Exception:
    _HAVE_STUB = False
    _stub_load_vgae_model = None
    _stub_vgae_sample_z = None

try:
    from vgae_prior_GAT import VGAEPrior as _VGAEPrior

    _HAVE_VGAE_NATIVE = True
except Exception:
    _HAVE_VGAE_NATIVE = False
    _VGAEPrior = None

try:
    from z_generator_stub import load_vgae_model as _gate_load_vgae_model
except Exception:
    _gate_load_vgae_model = None


# ---------------- artifact logging & z_dim probe ----------------
def _infer_vgae_zdim_from_ckpt(ckpt_path: str):
    import torch as _t

    sd = _t.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    candidates = []
    for k, v in sd.items():
        try:
            shape = tuple(v.shape)
        except Exception:
            continue
        if len(shape) == 2 and re.search(r"(mu).*?(lin|proj|fc|head).*?weight|enc.*mu.*weight", k, re.I):
            candidates.append((k, shape))
    if candidates:
        k, shape = sorted(candidates, key=lambda kv: kv[1][0])[-1]
        return shape[0], k
    for k in ["z_dim", "latent_dim", "latent_size"]:
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], (int, np.integer)):
            return int(sd[k]), f"<meta:{k}>"
    return None, None


def log_vgae_artifacts_info(args, log_fn=print):
    ckpt = getattr(args, "vgae_ckpt", None)
    p_mean = getattr(args, "vgae_mean", None)
    p_cov = getattr(args, "vgae_cov", None)
    p_bank = getattr(args, "vgae_zbank", None)

    zdim_ckpt, src_key = None, None
    if ckpt and os.path.exists(ckpt):
        try:
            zdim_ckpt, src_key = _infer_vgae_zdim_from_ckpt(ckpt)
            if zdim_ckpt is not None:
                log_fn(f"[VGAE ckpt] {ckpt} -> z_dim={zdim_ckpt}  (src={src_key})")
            else:
                log_fn(f"[VGAE ckpt] {ckpt} -> 无法自动推断 z_dim（未找到 μ 头权重）")
        except Exception as e:
            log_fn(f"[VGAE ckpt] {ckpt} -> 读取失败: {e}")
    else:
        if ckpt:
            log_fn(f"[VGAE ckpt] {ckpt} -> 文件不存在")
        else:
            log_fn("[VGAE ckpt] 未提供 --vgae-ckpt")

    dim_mean = dim_cov = dim_bank = None
    try:
        if p_mean and os.path.exists(p_mean):
            m = np.load(p_mean, mmap_mode="r")
            dim_mean = m.shape
            log_fn(f"[VGAE mean] {p_mean} -> shape={dim_mean}")
        else:
            log_fn(f"[VGAE mean] {p_mean or '<未提供>'} -> 文件不存在或未提供")
    except Exception as e:
        log_fn(f"[VGAE mean] 读取失败: {e}")

    try:
        if p_cov and os.path.exists(p_cov):
            C = np.load(p_cov, mmap_mode="r")
            dim_cov = C.shape
            log_fn(f"[VGAE cov ] {p_cov} -> shape={dim_cov}")
        else:
            log_fn(f"[VGAE cov ] {p_cov or '<未提供>'} -> 文件不存在或未提供")
    except Exception as e:
        log_fn(f"[VGAE cov ] 读取失败: {e}")

    try:
        if p_bank and os.path.exists(p_bank):
            Z = np.load(p_bank, mmap_mode="r")
            dim_bank = Z.shape
            log_fn(f"[VGAE bank] {p_bank} -> shape={dim_bank}")
        else:
            log_fn(f"[VGAE bank] {p_bank or '<未提供>'} -> 文件不存在或未提供")
    except Exception as e:
        log_fn(f"[VGAE bank] 读取失败: {e}")

    D_mean = dim_mean[0] if (isinstance(dim_mean, tuple) and len(dim_mean) == 1) else None
    D_cov = dim_cov[0] if (isinstance(dim_cov, tuple) and len(dim_cov) == 2 and dim_cov[0] == dim_cov[1]) else None
    D_bank = dim_bank[1] if (isinstance(dim_bank, tuple) and len(dim_bank) == 2) else None

    Ds = [d for d in [D_mean, D_cov, D_bank] if isinstance(d, (int, np.integer))]
    D_stats = None if not Ds else (Ds[0] if all(d == Ds[0] for d in Ds) else None)

    if D_mean is not None and D_cov is not None and D_bank is not None:
        if D_stats is None:
            log_fn("[VGAE check] ❌ mean/cov/z_bank 的维度不一致，请重新生成统计量。")
        else:
            log_fn(f"[VGAE check] 统计量维度一致：D={D_stats}")

    if zdim_ckpt is None:
        log_fn("[hint] 若无法推断 z_dim，可临时以 mean 的长度作为 D，但强烈建议统一以同一 ckpt 重新生成统计量。")


# ---------------- utilities ----------------
def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, path: str):
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def _probe_ckpt_z_dim(ckpt_path: str, mu_npy_path: str | None = None) -> Optional[int]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        if mu_npy_path and os.path.exists(mu_npy_path):
            m = np.load(mu_npy_path)
            return int(m.shape[0]) if m.ndim == 1 else int(m.shape[-1])
        return None
    try:
        blob = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        if mu_npy_path and os.path.exists(mu_npy_path):
            m = np.load(mu_npy_path)
            return int(m.shape[0]) if m.ndim == 1 else int(m.shape[-1])
        return None
    sd_candidates = []
    if isinstance(blob, dict):
        for k in ("state_dict", "model_state", "model", "module", "net", "vgae", "encoder"):
            if k in blob and isinstance(blob[k], dict):
                sd_candidates.append(blob[k])
        sd_candidates.append(blob)
        for v in list(blob.values()):
            if isinstance(v, dict) and any(hasattr(t, "shape") for t in v.values()):
                sd_candidates.append(v)
    elif hasattr(blob, "state_dict"):
        try:
            sd_candidates.append(blob.state_dict())
        except Exception:
            pass

    def extract_from_sd(sd) -> Optional[int]:
        zdim_bias = None
        zdim_weight = None
        for name, tensor in sd.items():
            if not hasattr(tensor, "shape"):
                continue
            shp = tuple(tensor.shape)
            nml = name.lower()
            if len(shp) == 2 and ("mu" in nml or "mean" in nml) and ("weight" in nml or "lin" in nml or "proj" in nml or "fc" in nml):
                zdim_weight = int(shp[0])
            if len(shp) == 1 and ("mu" in nml or "mean" in nml) and ("bias" in nml or nml.endswith("lin.bias")):
                zdim_bias = int(shp[0])
        if zdim_weight and zdim_bias and zdim_weight == zdim_bias:
            return zdim_weight
        if zdim_weight:
            return zdim_weight
        if zdim_bias:
            return zdim_bias
        return None

    for sd in sd_candidates:
        if isinstance(sd, dict):
            zdim = extract_from_sd(sd)
            if zdim:
                return int(zdim)

    if mu_npy_path and os.path.exists(mu_npy_path):
        m = np.load(mu_npy_path)
        return int(m.shape[0]) if m.ndim == 1 else int(m.shape[-1])
    return None


# ---------------- token & sampling tools ----------------
def _split_tokens_like_generator(seq: str) -> List[str]:
    toks, i, n = [], 0, len(seq)
    while i < n:
        c = seq[i]
        if c == "[":
            j = seq.find("]", i + 1)
            if j == -1:
                toks.append(c)
                i += 1
            else:
                toks.append(seq[i : j + 1])
                i = j + 1
        else:
            toks.append(c)
            i += 1
    return toks


def forward_logits_along(model, data, seq: str, device):
    hidden = model.init_hidden().to(device)
    if model.has_cell:
        cell = model.init_cell().to(device)
        hidden = (hidden, cell)
    stack = model.init_stack().to(device) if model.has_stack else None
    tokens = _split_tokens_like_generator(seq) if ("[" in seq and "]" in seq) else list(seq)
    logits_list = []
    for i in range(1, len(tokens)):
        prev_tok = tokens[i - 1]
        if hasattr(data, "all_characters") and prev_tok not in data.all_characters:
            continue
        try:
            inp = data.char_tensor(prev_tok).to(device)
        except KeyError:
            continue
        out, hidden, stack = model.forward(inp, hidden, stack)
        logits_list.append(out.view(-1))
    return logits_list


def rollout_with_logprobs(
    G: StackAugmentedRNN,
    data: GeneratorData,
    max_len: int,
    device: torch.device,
    top_p: float,
    temperature: float,
    prime: str = "<",
    end_token: str = ">",
    eos_bias: float = 0.0,
    min_len: int = 0,
    no_repeat_ngram: int = 3,
    repeat_penalty: float = 1.2,
):
    hidden = G.init_hidden().to(device)
    if G.has_cell:
        cell = G.init_cell().to(device)
        hidden = (hidden, cell)
    stack = G.init_stack().to(device) if G.has_stack else None

    prime_input = data.char_tensor(prime).to(device)
    for p in range(len(prime) - 1):
        _, hidden, stack = G.forward(prime_input[p], hidden, stack)
    inp = prime_input[-1]

    seq = prime
    logprob_sum = torch.tensor(0.0, device=device)
    entropy_sum = torch.tensor(0.0, device=device)
    L = 0

    try:
        eos_idx = data.all_characters.index(end_token) if end_token in data.all_characters else None
        bos_idx = data.all_characters.index("<") if "<" in data.all_characters else None
    except Exception:
        eos_idx = bos_idx = None

    prev_tokens: List[int] = []
    seen_ngrams = set()
    logits_list: List[torch.Tensor] = []

    for _ in range(max_len):
        output, hidden, stack = G.forward(inp, hidden, stack)
        logits = output.view(-1)
        mod_logits = logits.clone()

        if bos_idx is not None:
            mod_logits[bos_idx] = -1e9
        if eos_idx is not None and L < int(min_len or 0):
            mod_logits[eos_idx] = -1e9

        n = int(no_repeat_ngram or 0)
        if n >= 2 and len(prev_tokens) >= (n - 1):
            prefix = tuple(prev_tokens[-(n - 1) :])
            for c in range(mod_logits.numel()):
                if (*prefix, c) in seen_ngrams:
                    mod_logits[c] = -1e9

        if repeat_penalty and repeat_penalty > 1.0 and len(prev_tokens) >= 1:
            last_tok = prev_tokens[-1]
            mod_logits[last_tok] = mod_logits[last_tok] - math.log(float(repeat_penalty))

        if eos_bias != 0.0 and eos_idx is not None and L >= int(min_len or 0):
            mod_logits[eos_idx] = mod_logits[eos_idx] + float(eos_bias)

        if temperature <= 0:
            tok = int(torch.argmax(mod_logits).item())
            probs = F.softmax(mod_logits, dim=-1)
        else:
            probs = F.softmax(mod_logits / float(temperature), dim=-1)
            if (not top_p) or top_p >= 1.0:
                tok = int(torch.multinomial(probs, 1).item())
            else:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                mask = cum <= top_p
                mask[..., 0] = True
                cut_idx = si[mask]
                cut_probs = probs[cut_idx]
                cut_probs = cut_probs / cut_probs.sum()
                tok = int(cut_idx[torch.multinomial(cut_probs, 1).item()].item())

        logprob_sum = logprob_sum + F.log_softmax(mod_logits, dim=-1)[tok]
        entropy_sum = entropy_sum + (-torch.sum(probs * torch.log(probs + 1e-12)))

        logits_list.append(mod_logits.detach())

        ch = data.all_characters[tok]
        seq += ch
        inp = data.char_tensor(ch).to(device)
        L += 1

        prev_tokens.append(tok)
        if n >= 2 and len(prev_tokens) >= n:
            seen_ngrams.add(tuple(prev_tokens[-n:]))

        if ch == end_token:
            break

    return seq, logprob_sum, L, logits_list, entropy_sum


# ---------------- SMILES/SELFIES utilities ----------------
def ensure_wrapped_angles(s: str) -> str:
    s = s.strip()
    if not s.startswith("<"):
        s = "<" + s
    if not s.endswith(">"):
        s = s + ">"
    return s


class OfflinePool:
    def __init__(self, seqs: Sequence[str]):
        self.pool = [ensure_wrapped_angles(s) for s in seqs if isinstance(s, str) and s.strip()]

    def sample(self, k: int) -> List[str]:
        if k <= 0:
            return []
        if k >= len(self.pool):
            return random.sample(self.pool, len(self.pool))
        return random.sample(self.pool, k)


class SampleBuffer:
    """用于方案二：缓存训练过程中见过的序列（含 pulse），采样时从这里抽样。"""

    def __init__(self, max_size: int = 5000):
        self.max_size = int(max_size)
        self.pool: List[str] = []

    def add_many(self, seqs: Sequence[str]):
        if not seqs:
            return
        for s in seqs:
            if not isinstance(s, str):
                continue
            s_norm = ensure_wrapped_angles(s)
            if not s_norm:
                continue
            self.pool.append(s_norm)
        if self.max_size > 0 and len(self.pool) > self.max_size:
            # 只保留最近 max_size 条
            self.pool = self.pool[-self.max_size :]

    def sample(self, k: int) -> List[str]:
        if k <= 0 or not self.pool:
            return []
        if k >= len(self.pool):
            return random.sample(self.pool, len(self.pool))
        return random.sample(self.pool, k)


def _read_offline_sequences(path: str, col: str | int) -> List[str]:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if isinstance(col, str) and col.isdigit():
        col = int(col)
    if isinstance(col, int):
        colname = df.columns[col]
    else:
        colname = None
        for c in df.columns:
            if str(c).lower() == str(col).lower():
                colname = c
                break
        if colname is None:
            raise ValueError(f"列 '{col}' 不在 {path} 中。可选列：{list(df.columns)}")
    seqs = df[colname].astype(str).tolist()
    return [ensure_wrapped_angles(s) for s in seqs if isinstance(s, str) and s.strip()]


def _seq_core_local(seq: str) -> str:
    return seq.strip().strip("<>").replace(" ", "")


def _seq_to_smiles_local(seq: str, gen_mode: str) -> str:
    core = _seq_core_local(seq)
    if gen_mode.lower() == "selfies":
        try:
            return sf.decoder(core)
        except Exception:
            return ""
    else:
        return core


# ---------------- Z sources ----------------
def load_z_bank(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        z = np.load(path)
    else:
        z = torch.load(path, map_location="cpu")
        z = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    z = np.asarray(z, dtype=np.float32)
    return z[None, :] if z.ndim == 1 else z


class ZSource:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.bank: Optional[np.ndarray] = None
        self.vgae_prior = None
        self._stub = None
        if args.z_dim > 0:
            if args.z_source == "bank":
                assert args.z_bank, "--z-source bank 需要 --z-bank"
                self.bank = load_z_bank(args.z_bank)
                assert self.bank.shape[1] == args.z_dim, f"z_bank dim={self.bank.shape[1]} != --z-dim {args.z_dim}"
            elif args.z_source == "vgae":
                assert args.vgae_ckpt, "--z-source vgae 需要 --vgae-ckpt"
                if _HAVE_VGAE_NATIVE:
                    zdim = _probe_ckpt_z_dim(args.vgae_ckpt) or args.z_dim
                    self.vgae_prior = _VGAEPrior(z_dim=int(zdim), hidden=128, in_dim=19)
                    try:
                        self.vgae_prior.load_from_ckpt(args.vgae_ckpt, map_location=self.device, strict="auto")
                        self.vgae_prior.to(self.device).eval()
                    except Exception:
                        self.vgae_prior = None
                if self.vgae_prior is None and _HAVE_STUB:
                    self._stub = _stub_load_vgae_model(args.vgae_ckpt, self.device)
            elif args.z_source == "fixed":
                assert args.z_fixed is not None, "--z-source fixed 需要 --z-fixed"
                assert len(args.z_fixed) == args.z_dim

    def sample(self) -> Optional[torch.Tensor]:
        a = self.args
        if a.z_dim <= 0:
            return None
        if a.z_source == "gauss":
            z = torch.randn(1, a.z_dim, device=self.device)
        elif a.z_source == "fixed":
            z = torch.tensor(a.z_fixed, dtype=torch.float32, device=self.device).view(1, -1)
        elif a.z_source == "bank":
            idx = np.random.randint(0, self.bank.shape[0])
            z = torch.tensor(self.bank[idx : idx + 1], dtype=torch.float32, device=self.device)
        else:  # vgae
            if self.vgae_prior is not None:
                z_nodes = self.vgae_prior.sample_z(batch_size=1, n_nodes=1, device=self.device)  # [1,1,D]
                z = z_nodes.squeeze(1).contiguous()
            else:
                z = _stub_vgae_sample_z(self._stub, n=1, device=self.device)  # [1, D]
        if z.dim() == 1:
            z = z.view(1, -1)
        if z.size(1) != a.z_dim:
            print(f"[warn] z_dim mismatch: sampled {z.size(1)} vs arg {a.z_dim} — auto-fixing")
            if z.size(1) > a.z_dim:
                z = z[:, : a.z_dim]
            else:
                pad = torch.zeros(z.size(0), a.z_dim - z.size(1), device=z.device, dtype=z.dtype)
                z = torch.cat([z, pad], dim=1)
        return z


class GateForSampling:
    def __init__(
            self,
            args,
            rer: MultiObjectiveReward | None,
            gen_mode: str = "selfies",
            teacher_smiles: Optional[List[str]] = None,
        ):
            self.args = args
            self.rer = rer
            self.gen_mode = gen_mode.lower()
            self.enabled = bool(args.sample_hard_gate)
            self.mu: Optional[np.ndarray] = None
            self.cov: Optional[np.ndarray] = None
            self._Si: Optional[np.ndarray] = None
            self.d2_thr: Optional[float] = None
            self.s_thr: Optional[float] = None
            self._enc = None
            self._teacher_smiles = teacher_smiles or []
            # diagnostics / export
            self._teacher_s_vals: List[float] = []
            self._last_s_thr_teacher: Optional[float] = None
            self._last_emp_pass: Optional[float] = None
            # ⭐ NEW: 采样端“reward-tighten”之后真正使用的 s 阈值（不写回 gate_stats）
            self._s_thr_reward_eff: Optional[float] = None
    
            if not self.enabled:
                return
    
            # 1) try gate_stats (mu/cov and maybe s_thr)
            if getattr(args, "vgae_gate_stats", None) and args.vgae_gate_stats and os.path.exists(args.vgae_gate_stats):
                try:
                    G = np.load(args.vgae_gate_stats, allow_pickle=True)
                    if "mu" in G and G["mu"] is not None:
                        self.mu = G["mu"].astype(np.float32)
                    if "cov" in G and G["cov"] is not None:
                        self.cov = G["cov"].astype(np.float32)
                    if "s_thr" in G and G["s_thr"] is not None:
                        self.s_thr = float(G["s_thr"])
                    elif "d2_thr" in G and G["d2_thr"] is not None:
                        self.s_thr = -0.5 * float(G["d2_thr"])
                except Exception as e:
                    print(f"[sample-gate] failed to load gate_stats: {e}")
    
            # 2) load mean/cov if needed
            if self.mu is None and args.vgae_mean and os.path.exists(args.vgae_mean):
                try:
                    self.mu = np.load(args.vgae_mean).astype(np.float32)
                except Exception as e:
                    print(f"[sample-gate] failed to load mu: {e}")
            if self.cov is None and args.vgae_cov and os.path.exists(args.vgae_cov):
                try:
                    self.cov = np.load(args.vgae_cov).astype(np.float32)
                except Exception as e:
                    print(f"[sample-gate] failed to load cov: {e}")
    
            # 3) compute s_thr from z_bank if a quantile spec is present (recompute even if s_thr exists)
            spec = getattr(args, "sample_gate", None) or getattr(args, "vgae_gate", "off")
            q = None
            if isinstance(spec, str) and spec.startswith("quantile:"):
                try:
                    q = float(spec.split(":", 1)[1])
                except Exception:
                    q = None
            if (q is not None) and (self.mu is not None) and args.vgae_zbank and os.path.exists(args.vgae_zbank):
                try:
                    Z = np.load(args.vgae_zbank).astype(np.float32)  # (N, D)
                    DZ = Z - self.mu.reshape(1, -1)
                    # metric-aware s_all
                    if (self.args.gate_metric == "maha") and (self.cov is not None):
                        self._ensure_precision()
                        s_all = np.array([-0.5 * float(d @ self._Si @ d) for d in DZ], dtype=np.float32)
                    else:
                        s_all = -0.5 * np.sum(np.square(DZ), axis=1, dtype=np.float32)
                    self.s_thr = float(np.quantile(s_all, q))
                    self.d2_thr = None
                    print(f"[sample-gate] using s-quantile: q={q:.2f} -> s_thr={self.s_thr:.6f} (bigger q = stricter)")
                except Exception as e:
                    print(f"[sample-gate] failed to compute s_thr from z_bank: {e}")
    
            # 4) load a fallback encoder for gate-side SMILES→z
            if getattr(args, "vgae_ckpt", None) and _gate_load_vgae_model is not None:
                try:
                    dev = torch.device("cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu")
                    self._enc = _gate_load_vgae_model(args.vgae_ckpt, dev)
                except Exception:
                    self._enc = None
    
            # 5) final check & diag
            # ==================== MODIFIED START ====================
            # 核心修改：如果未找到初始阈值(vgae-gate off)，但开启了教师校准，则强制初始化为正无穷(inf)。
            # 后续的校准逻辑是 min(self.s_thr, teacher_thr)，inf 会被自动替换为 teacher_thr。
            if (self.s_thr is None) and (self.d2_thr is None):
                if bool(getattr(args, "gate_calibrate_teacher", False)):
                    self.s_thr = float('inf')
                    print("[sample-gate] 'vgae-gate' is OFF but calibration is ON. Init s_thr=inf (waiting for teacher override).")
                else:
                    print("[sample-gate] disabled: missing thresholds (s_thr/d2_thr).")
                    self.enabled = False
            # ==================== MODIFIED END ======================
            
            if self.enabled: # 只有 enabled 才打印
                try:
                    d = None if self.mu is None else self.mu.shape[0]
                    msg = "[sample-gate] init: "
                    if self.s_thr is not None:
                        msg += f"s_thr={self.s_thr:.3f} "
                    if self.d2_thr is not None:
                        msg += f"d2_thr={self.d2_thr:.3f} "
                    msg += f"zdim={d}"
                    print(msg)
                except Exception:
                    pass
            
            try:
                has_rer_enc = (self.rer is not None) and (getattr(self.rer, "_vgae_encoder", None) is not None)
                has_rer_mu = (self.rer is not None) and (getattr(self.rer, "mu", None) is not None)
                has_rer_cov = (self.rer is not None) and (getattr(self.rer, "cov", None) is not None)
                has_gate_mu = self.mu is not None
                has_gate_cov = self.cov is not None
                print(
                    f"[sample-gate][diag] rer_encoder={has_rer_enc} rer_mu={has_rer_mu} rer_cov={has_rer_cov} | "
                    f"gate_mu={has_gate_mu} gate_cov={has_gate_cov} enc={self._enc is not None} | s_thr={self.s_thr}"
                )
            except Exception:
                pass
    
            # 6) teacher coverage calibration (optional)
            s_vals = []
            try:
                if self._teacher_smiles:
                    # 先采样教师 s 分布，供校准与 tighten 共用
                    s_vals = self._collect_teacher_s_vals(max_n=1000)
                    self._teacher_s_vals = list(s_vals)
    
                if self.s_thr is not None and s_vals and bool(getattr(self.args, "gate_calibrate_teacher", False)):
                    s_vals_np = np.asarray(s_vals, dtype=np.float32)
                    cover = float(getattr(self.args, "teacher_pass_cover", 0.8))
                    cover = min(max(cover, 0.0), 0.999)
                    s_thr_teacher = float(np.quantile(s_vals_np, 1.0 - cover))
                    old = float(self.s_thr)
                    # 覆盖率校准：只放宽，不加严。因为初始值是 inf，这里会直接变为 s_thr_teacher。
                    self.s_thr = min(self.s_thr, s_thr_teacher)
                    ok_rate = float((s_vals_np >= self.s_thr).mean())
                    self._last_s_thr_teacher = s_thr_teacher
                    self._last_emp_pass = ok_rate
                    print(
                        f"[sample-gate][calib] teacher cover target={cover:.2f} => s_thr_teacher={s_thr_teacher:.6f} | "
                        f"s_thr {old:.6f} -> {self.s_thr:.6f} (empirical_pass={ok_rate:.3f})"
                    )
                elif self.s_thr is not None and s_vals:
                    # 未启用覆盖率校准也记录 teacher s 分布，便于后续 tighten 和写文件
                    self._last_s_thr_teacher = None
                    self._last_emp_pass = float((np.asarray(s_vals) >= self.s_thr).mean())
            except Exception as e:
                print(f"[sample-gate][calib] 失败: {e}")
    
            # 7) teacher tighten by quantile (在教师 s 分布上二次加严，仍然作用在 self.s_thr 上)
            try:
                tighten_q = getattr(self.args, "gate_tighten_q", None)
                if (tighten_q is not None) and (self.s_thr is not None) and self._teacher_s_vals:
                    qv = float(tighten_q)
                    qv = min(max(qv, 0.0), 0.999)
                    s_q = float(np.quantile(np.asarray(self._teacher_s_vals, dtype=np.float32), qv))
                    old = float(self.s_thr)
                    # 二次加严：与覆盖率阈值取更严格的一侧
                    if self._last_s_thr_teacher is not None:
                        self.s_thr = max(self.s_thr, s_q, self._last_s_thr_teacher)
                    else:
                        self.s_thr = max(self.s_thr, s_q)
                    ok_rate_q = float((np.asarray(self._teacher_s_vals) >= self.s_thr).mean())
                    print(
                        f"[sample-gate][tighten] q={qv:.2f} => s_q={s_q:.6f} | s_thr {old:.6f} -> {self.s_thr:.6f} "
                        f"(empirical_pass={ok_rate_q:.3f})"
                    )
            except Exception as e:
                print(f"[sample-gate][tighten] 失败: {e}")
    
            # 8) ⭐ 采样端重用 reward-tighten-q：在 teacher_thr 基础上直接缩小阈值（不改写 gate_stats 里的 s_thr）
            try:
                rt_q = getattr(self.args, "reward_tighten_q", None)
                if (rt_q is not None) and (self.s_thr is not None):
                    qv = float(rt_q)
                    qv = min(max(qv, 0.0), 0.999)
                    # teacher_thr 统一取当前 self.s_thr（已经包含 quantile + 覆盖率 + gate_tighten）
                    teacher_thr = float(self.s_thr)
                    if teacher_thr < 0.0:
                        # 例：teacher_thr=-0.869, q=0.35 → s_thr_eff≈-0.565（更靠近 0，门更“紧”）
                        s_eff = teacher_thr * (1.0 - qv)
                    else:
                        s_eff = teacher_thr * (1.0 + qv)
                    self._s_thr_reward_eff = float(s_eff)
                    print(
                        f"[sample-gate][reward-tighten] teacher_thr={teacher_thr:.6f} "
                        f"reward_tighten_q={qv:.2f} => s_thr_eff={self._s_thr_reward_eff:.6f}"
                    )
            except Exception as e:
                print(f"[sample-gate][reward-tighten] 失败: {e}")
    
    # ---- metric helpers ----
    def _ensure_precision(self):
        if (getattr(self, "_Si", None) is None) and (self.cov is not None):
            C = np.asarray(self.cov, dtype=np.float32)
            eps = 1e-6 * float(np.trace(C) / max(1, C.shape[0]))
            self._Si = np.linalg.inv(C + np.eye(C.shape[0], dtype=np.float32) * eps)

    def _calc_s_from_z(self, z: np.ndarray) -> float:
        dif = z - self.mu
        if (self.args.gate_metric == "maha") and (self.cov is not None):
            self._ensure_precision()
            d2 = float(dif @ self._Si @ dif)
        else:
            d2 = float(np.sum(np.square(dif)))
        return -0.5 * d2

    # ---- encoding & scoring ----
    def _encode_to_s(self, smiles: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        encs = []
        if hasattr(self.rer, "_vgae_encoder") and (getattr(self.rer, "_vgae_encoder", None) is not None):
            encs.append(getattr(self.rer, "_vgae_encoder"))
        if self._enc is not None and self._enc is not getattr(self.rer, "_vgae_encoder", None):
            encs.append(self._enc)
        if self.mu is None:
            return None, None, "no_mu"
        for enc in encs:
            try:
                z_raw = None
                if hasattr(enc, "encode_smiles"):
                    z_raw = enc.encode_smiles(smiles)
                elif hasattr(enc, "encode_batch"):
                    zz = enc.encode_batch([smiles])
                    z_raw = zz[0] if (zz is not None and len(zz) > 0) else None
                z = np.asarray(z_raw, dtype=np.float32)
                D = int(self.mu.shape[0])
                if z.ndim == 1 and z.shape[0] == D:
                    z_vec = z
                elif z.size % D == 0:
                    z_vec = z.reshape(-1, D).mean(axis=0).astype(np.float32, copy=False)
                elif z.ndim == 2 and (z.shape[0] == D or z.shape[1] == D):
                    z_vec = z.reshape(-1, D).mean(axis=0).astype(np.float32, copy=False)
                else:
                    continue
                s = self._calc_s_from_z(z_vec)
                d2 = max(0.0, -2.0 * s)
                return s, d2, None
            except Exception as e:
                err = str(e)
        return None, None, "enc_fail"

    def _collect_teacher_s_vals(self, max_n: int = 1000) -> List[float]:
        vals: List[float] = []
        if not self._teacher_smiles or (self.mu is None):
            return vals
        samp = self._teacher_smiles[: max(1, int(max_n))]
        for smi in samp:
            s_i, _, _ = self._encode_to_s(smi)
            if s_i is not None:
                vals.append(float(s_i))
        return vals

    # ---- gate decisions ----
    def filter_list(self, seqs: List[str]) -> Tuple[List[str], int]:
        if not self.enabled:
            return seqs, len(seqs)
        kept = []
        for s in seqs:
            if self._passes(s):
                kept.append(s)
        return kept, len(seqs)

    def _passes(self, seq: str) -> bool:
        if not self.enabled:
            return True
        seq_smiles = _seq_to_smiles_local(seq, self.gen_mode)
        if not seq_smiles:
            if getattr(self.args, "sample_report", False):
                print("[gate] decode_fail: empty SMILES")
            return False
        s, d2, err = self._encode_to_s(seq_smiles)

        # 用统一的 s 阈值（优先 reward-tighten 后的 s_thr_eff）
        if getattr(self.args, "sample_report", False):
            thr = (
                self._s_thr_reward_eff
                if (self._s_thr_reward_eff is not None)
                else (
                    self.s_thr
                    if self.s_thr is not None
                    else (-0.5 * self.d2_thr if self.d2_thr is not None else None)
                )
            )
            if s is None:
                print(f"[gate] no s (err={err})")
            else:
                if thr is not None:
                    verdict = "PASS" if s >= thr else "FAIL"
                    print(f"[gate] s={s:.3f} vs thr={thr:.3f} -> {verdict}")
                else:
                    print(f"[gate] s={s:.3f} (no thr)")

        if (s is None) and (d2 is None):
            return True  # be permissive if encoding missing

        # ⭐ 逻辑：优先用 s 阈值（reward-tighten 后），其次才回退 d2_thr
        thr_eff = self._s_thr_reward_eff if (self._s_thr_reward_eff is not None) else self.s_thr
        if (thr_eff is not None) and (s is not None):
            return s >= thr_eff
        if (self.d2_thr is not None) and (d2 is not None):
            return d2 <= self.d2_thr
        return True



# ---------------- auto z_dim inference ----------------
def _auto_infer_and_set_zdim(args) -> int:
    inferred = None
    if args.z_source == "vgae" and args.vgae_ckpt and os.path.exists(args.vgae_ckpt):
        zd = _probe_ckpt_z_dim(args.vgae_ckpt)
        if zd and zd > 0:
            inferred = int(zd)
            print(f"[zdim] inferred from VGAE ckpt: {inferred}")
    if inferred is None and args.z_source == "bank" and args.z_bank and os.path.exists(args.z_bank):
        try:
            Z = load_z_bank(args.z_bank)
            if Z.ndim == 2 and Z.shape[1] > 0:
                inferred = int(Z.shape[1])
                print(f"[zdim] inferred from z_bank: {inferred}")
        except Exception:
            pass
    if inferred is None and args.z_source == "fixed" and args.z_fixed:
        inferred = int(len(args.z_fixed))
        print(f"[zdim] inferred from z_fixed: {inferred}")
    if inferred is not None and inferred != int(args.z_dim):
        print(f"[info] overriding --z-dim {args.z_dim} -> {inferred}")
        args.z_dim = inferred
    return int(args.z_dim)


# ================================================================
#                           Main
# ================================================================
def main():
    ap = argparse.ArgumentParser()

    # sampling-time hard gate (affects only saving samples)
    ap.add_argument(
        "--sample-hard-gate",
        action="store_true",
        help="仅在保存样本时，保留通过 VGAE 硬门控的分子。",
    )
    ap.add_argument(
        "--sample-report",
        action="store_true",
        help="保存样本时打印门控后的保留率与阈值信息，并逐条打印 s vs thr。",
    )
    ap.add_argument(
        "--vgae-gate-stats",
        type=str,
        default=None,
        help="(可选) 直接加载 gate_stats.npz（含 mu/cov/s_thr），用于采样期硬门控。",
    )
    ap.add_argument(
        "--sample-warmup",
        type=int,
        default=0,
        help="采样期硬门控的 warmup（按 seen_samples 计数），未到该值时不做采样期过滤",
    )
    ap.add_argument(
        "--sample-gate",
        type=str,
        default=None,
        help="仅采样期使用的门控规格（如 quantile:0.35）；不填则沿用 --vgae-gate",
    )

    # NEW: unified gate metric + teacher calibration + tighten
    ap.add_argument(
        "--gate-metric",
        choices=["euclid", "maha"],
        default="euclid",
        help="门控与统计统一使用的距离度量（含阈值计算与单分子评分）",
    )
    ap.add_argument(
        "--gate-calibrate-teacher",
        action="store_true",
        help="用教师集的 s 分布校准 s_thr，使至少达到指定覆盖率（只放宽不加严）",
    )
    ap.add_argument(
        "--teacher-pass-cover",
        type=float,
        default=0.8,
        help="教师样本希望通过门控的覆盖率（0~1），仅在 --gate-calibrate-teacher 时生效",
    )
    ap.add_argument(
        "--gate-tighten-q",
        type=float,
        default=None,
        help="在教师 s 分布上‘二次加严’的分位数（例：0.80）。最终 s_thr = max(s_thr_after_cover, s_q)。",
    )

    # tokenizer (SELFIES)
    ap.add_argument("--vocab-csv", required=True)
    ap.add_argument("--seq-col", type=int, default=0)
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("--keep-header", action="store_true")
    ap.add_argument("--max-len", type=int, default=200)

    # policy
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cell", choices=["GRU", "LSTM"], default="GRU")
    ap.add_argument("--resume", type=str, default="")

    # z condition
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--z-source", choices=["gauss", "bank", "fixed", "vgae"], default="gauss")
    ap.add_argument("--z-bank", type=str, default="")
    ap.add_argument(
        "--z-fixed",
        type=lambda s: [float(x) for x in s.split(",")],
        default=None,
    )
    ap.add_argument("--vgae-ckpt", type=str, default="")
    ap.add_argument("--z-scale", type=float, default=2.2)

    # predictor / reward
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--predictor-stats", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--goals", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--target-values", nargs="+", default=None)
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--invalid-penalty", type=float, default=-20.0)
    ap.add_argument("--unique-bonus", type=float, default=0.0)
    ap.add_argument("--featurizer", type=str, default="auto")
    ap.add_argument(
        "--predictor-output-transform",
        type=str,
        default=None,
        help="对预测器输出做的逆变换（如 log1p → expm1），默认不变换",
    )
    ap.add_argument(
        "--predictor-transform-targets",
        nargs="+",
        default=None,
        help="仅对指定目标应用输出逆变换，不提供则作用于全部目标",
    )
    # duplication penalties
    ap.add_argument("--dupe-penalty", type=float, default=0.0)
    ap.add_argument("--dupe-escalate", action="store_true")
    ap.add_argument("--dupe-window", type=int, default=5000)
    # 默认使用 canonical SMILES 做重复检测；--no-dupe-canonical 可显式关闭
    ap.add_argument(
        "--dupe-canonical",
        dest="dupe_canonical",
        action="store_true",
        help="使用 canonical SMILES 判重（默认开启）",
    )
    ap.add_argument(
        "--no-dupe-canonical",
        dest="dupe_canonical",
        action="store_false",
        help="关闭基于 canonical SMILES 的重复检测",
    )
    ap.set_defaults(dupe_canonical=True)

    # VGAE reward/gating (for training-time reward; sampling-time gate reads these stats)
    ap.add_argument("--vgae-mean", type=str, default="")
    ap.add_argument("--vgae-cov", type=str, default="")
    ap.add_argument("--vgae-zbank", type=str, default="")
    ap.add_argument("--vgae-weight", type=float, default=0.0)
    ap.add_argument("--vgae-gate", type=str, default="off")
    ap.add_argument("--vgae-warmup", type=int, default=0)
    ap.add_argument("--vgae-anneal", type=str, default=None)

    # length / heavy-atom penalties
    ap.add_argument("--len-target", type=int, default=0)
    ap.add_argument("--len-lambda", type=float, default=0.0)
    ap.add_argument("--ha-target", type=int, default=0)
    ap.add_argument("--ha-lambda", type=float, default=0.0)

    # RL
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--optim", choices=["adam", "adadelta"], default="adam")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--cuda", action="store_true")

    # sampling
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--eos-bias", type=float, default=0.0)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--repeat-penalty", type=float, default=1.2)
    ap.add_argument(
        "--sample-mode",
        type=str,
        default="new",
        choices=["new", "last_batch", "buffer"],
        help="采样/落盘使用的序列集合方案：new=重新rollout(方案三，默认)；last_batch=最近一次训练batch(方案一)；buffer=训练缓存(方案二)",
    )
    ap.add_argument(
        "--sample-buffer-size",
        type=int,
        default=5000,
        help="sample-mode=buffer 时采样缓存的最大序列数。",
    )

    # regularization
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--entropy-coef", type=float, default=0.003)

    # reward-side tighten & energetic heuristics
    ap.add_argument(
        "--reward-tighten-q",
        type=float,
        default=None,
        help="奖励端基于 teacher s 分布的二次加严分位数，例如 0.35（None 表示关闭）",
    )
    ap.add_argument(
        "--heur-weight",
        type=float,
        default=None,
        help="结构启发式奖励权重；None 表示使用默认（MultiObjConfig 中的值），0 则完全关闭启发式",
    )
    ap.add_argument(
        "--heur-min-nitro-total",
        type=int,
        default=None,
        help="结构启发式：全分子最少硝基数；None 使用默认值",
    )
    ap.add_argument(
        "--heur-min-rings",
        type=int,
        default=None,
        help="结构启发式：最少环数量；None 使用默认值",
    )

    # offline (teacher) injection
    ap.add_argument(
        "--offline-csv",
        type=str,
        default="/public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin.csv",
        help="包含 SELFIES 的 CSV/TSV 路径",
    )
    ap.add_argument(
        "--offline-col",
        type=str,
        default="selfies",
        help="SELFIES 列名或列序（0-based）",
    )
    ap.add_argument("--pulse-every", type=int, default=0, help="每隔多少个 outer step 注入一次离线序列（0=关闭）")
    ap.add_argument("--pulse-k", type=int, default=1, help="每次注入的离线序列条数")
    ap.add_argument(
        "--pulse-sup-weight",
        type=float,
        default=1.0,
        help="教师 SMILES 注入的模仿损失权重（0 表示关闭，仅依赖 RL 优势更新)",
    )

    # logging
    ap.add_argument("--outdir", type=str, default="ckpts/rl_multiobj")
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--save-samples", type=str, default="")
    ap.add_argument("--fresh-log", action="store_true")

    args = ap.parse_args()

    # artifacts diag
    log_vgae_artifacts_info(args, log_fn=lambda s: print(s, flush=True))

    GEN_MODE = "selfies"

    ensure_dir(args.outdir)
    log_csv = os.path.join(args.outdir, "rl_log.csv")
    if args.fresh_log or (not os.path.exists(log_csv)):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "avg_reward", "avg_len", "time", "lr"])

    # tokenizer (SELFIES)
    data = GeneratorData(
        args.vocab_csv,
        max_len=args.max_len,
        cols_to_read=[args.seq_col],
        keep_header=args.keep_header,
        delimiter=args.delimiter,
        mode=GEN_MODE,
    )
    train_pool = list(getattr(data, "sequences", []))
    _lens = [s.count("[") for s in train_pool]
    avg_len = (sum(_lens) / len(_lens)) if _lens else 0.0
    print(f"[data] mode={GEN_MODE}  samples={len(train_pool)}  vocab={data.n_characters}  avg_len={avg_len:.1f}")

    # offline teacher pool
    try:
        if args.offline_csv and os.path.exists(args.offline_csv):
            off_seqs = _read_offline_sequences(args.offline_csv, args.offline_col)
            print(f"[offline] loaded {len(off_seqs)} SELFIES from {args.offline_csv} (col={args.offline_col})")
            OFF = OfflinePool(off_seqs)
        else:
            OFF = OfflinePool(train_pool)
            print(f"[offline] fallback to vocab pool ({len(train_pool)})")
    except Exception as e:
        print(f"[offline] load failed: {e} — fallback to vocab pool ({len(train_pool)})")
        OFF = OfflinePool(train_pool)

    # decode a slice of teacher SELFIES → SMILES for calibration/tighten
    try:
        TEACH_SMILES = []
        for s in list(getattr(OFF, "pool", []))[:2000]:
            smi = _seq_to_smiles_local(s, GEN_MODE)
            if smi:
                TEACH_SMILES.append(smi)
    except Exception:
        TEACH_SMILES = []

    # device
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[device]", device)

    # z_dim inference
    _auto_infer_and_set_zdim(args)

    # generator (conditional on z_dim)
    G = StackAugmentedRNN(
        input_size=data.n_characters,
        hidden_size=args.hidden,
        output_size=data.n_characters,
        layer_type=args.cell,
        n_layers=args.layers,
        has_stack=False,
        lr=args.lr,
        cond_dim=args.z_dim,
        cond_scale=1.0,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_dict = G.state_dict()
        filtered, skipped = {}, []
        for k, v in state.items():
            if k in model_dict and hasattr(v, "shape") and v.shape == model_dict[k].shape:
                filtered[k] = v
            else:
                skipped.append(
                    (
                        k,
                        tuple(getattr(v, "shape", [])),
                        tuple(model_dict[k].shape) if k in model_dict else None,
                    )
                )
        G.load_state_dict(filtered, strict=False)
        if skipped:
            print("[resume] skipped (shape mismatch):")
            for k, s_old, s_new in skipped:
                print(f"  - {k}: ckpt{s_old} != model{s_new}")

    G = G.to(device)

    G_ref = None
    if args.kl_coef and float(args.kl_coef) > 0:
        G_ref = deepcopy(G).to(device)
        for p in G_ref.parameters():
            p.requires_grad = False

    opt = (
        torch.optim.Adam(G.parameters(), lr=args.lr)
        if args.optim == "adam"
        else torch.optim.Adadelta(G.parameters(), lr=args.lr)
    )

    # ============== Build & calibrate sampling-time gate FIRST ==============
    sample_gate = GateForSampling(args, rer=None, gen_mode=GEN_MODE, teacher_smiles=TEACH_SMILES)

    # ============== Write gate_stats.npz (shared ruler) =====================
    gs_path = os.path.join(args.outdir, "gate_stats.npz")
    wrote_gate_stats = False
    if sample_gate.enabled and (sample_gate.mu is not None) and (
        sample_gate.s_thr is not None or sample_gate.d2_thr is not None
    ):
        s_thr_write = (
            sample_gate.s_thr
            if sample_gate.s_thr is not None
            else (-0.5 * float(sample_gate.d2_thr))
        )
        teacher_cover_cfg = (
            float(getattr(args, "teacher_pass_cover", 0.8))
            if bool(getattr(args, "gate_calibrate_teacher", False))
            else float("nan")
        )

        # empirical pass on teacher set
        emp_pass = float("nan")
        try:
            if sample_gate._teacher_s_vals:
                thr = float(s_thr_write)
                emp_pass = float((np.asarray(sample_gate._teacher_s_vals) >= thr).mean())
        except Exception:
            pass

        # build teacher quantile table
        q_grid = np.array(
            [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
            dtype=np.float32,
        )
        if sample_gate._teacher_s_vals:
            s_arr = np.asarray(sample_gate._teacher_s_vals, dtype=np.float32)
            q_vals = np.quantile(s_arr, q_grid)
        else:
            q_vals = np.full_like(q_grid, fill_value=np.nan, dtype=np.float32)

        payload = {
            "mu": np.asarray(sample_gate.mu, dtype=np.float32),
            "s_thr": float(s_thr_write),
            "metric": str(args.gate_metric),
            "teacher_empirical_pass": emp_pass,
            "teacher_target_cover": teacher_cover_cfg,
            "s_thr_teacher": (
                float(sample_gate._last_s_thr_teacher)
                if (sample_gate._last_s_thr_teacher is not None)
                else np.nan
            ),
            "teacher_s_quantiles_q": q_grid,
            "teacher_s_quantiles_s": q_vals,
            "gate_tighten_q": (
                float(getattr(args, "gate_tighten_q", np.nan))
                if getattr(args, "gate_tighten_q", None) is not None
                else np.nan
            ),
        }
        if sample_gate.cov is not None:
            payload["cov"] = np.asarray(sample_gate.cov, dtype=np.float32)
        if getattr(sample_gate, "_Si", None) is not None:
            payload["Si"] = np.asarray(sample_gate._Si, dtype=np.float32)

        np.savez(gs_path, **payload)
        wrote_gate_stats = True
        print(f"[sample-gate] wrote gate_stats.npz -> {gs_path}")
    else:
        print("[sample-gate] gate_stats.npz not written (gate disabled or missing mu/threshold)")

    # ============== rewarder config (read the same ruler) ===================
    cfg_kwargs: Dict[str, Any] = {
        "targets": args.targets,
        "weights": args.weights,
        "goals": [g.lower() for g in args.goals],
        "target_values": [
            (float(x) if x not in (None, "None") else None)
            for x in (args.target_values or [])
        ]
        + [None]
        * (len(args.targets) - (len(args.target_values or []))),
        "use_zscore": args.zscore,
        "invalid_penalty": args.invalid_penalty,
        "unique_bonus": args.unique_bonus,
        "predictor_path": args.predictor,
        "predictor_stats_path": args.predictor_stats,
        "featurizer": args.featurizer,
        "predictor_output_transform": (args.predictor_output_transform or None),
        "predictor_transform_targets": _normalize_transform_targets(
            args.predictor_transform_targets
        ),
        "dupe_penalty": float(getattr(args, "dupe_penalty", 0.0) or 0.0),
        "dupe_escalate": bool(getattr(args, "dupe_escalate", False)),
        "dupe_window": int(getattr(args, "dupe_window", 5000) or 5000),
        "dupe_canonical": bool(getattr(args, "dupe_canonical", True)),
        "offline_csv": (getattr(args, "offline_csv", None) or None),
        "offline_col": (getattr(args, "offline_col", "selfies") or "selfies"),
        "vgae_ckpt": (args.vgae_ckpt or None),
        "vgae_mean_path": (args.vgae_mean or None),
        "vgae_cov_path": (args.vgae_cov or None),
        "vgae_zbank_path": (args.vgae_zbank or None),
        "vgae_weight": float(args.vgae_weight or 0.0),
        "vgae_gate": (args.vgae_gate or "off"),
        "vgae_warmup": int(args.vgae_warmup or 0),
        "vgae_anneal": (args.vgae_anneal or None),
        "len_target": (args.len_target or None),
        "len_lambda": float(args.len_lambda or 0.0),
        "ha_target": (args.ha_target or None),
        "ha_lambda": float(args.ha_lambda or 0.0),
        # unified gate ruler for reward-time
        "gate_metric": str(args.gate_metric),
        "vgae_gate_stats_path": (
            gs_path
            if wrote_gate_stats and os.path.exists(gs_path)
            else (args.vgae_gate_stats or None)
        ),
    }
    # 追加可选的奖励端二次加严与启发式配置
    if getattr(args, "reward_tighten_q", None) is not None:
        cfg_kwargs["reward_tighten_q"] = float(args.reward_tighten_q)
    if getattr(args, "heur_weight", None) is not None:
        cfg_kwargs["heur_weight"] = float(args.heur_weight)
    if getattr(args, "heur_min_nitro_total", None) is not None:
        cfg_kwargs["heur_min_nitro_total"] = int(args.heur_min_nitro_total)
    if getattr(args, "heur_min_rings", None) is not None:
        cfg_kwargs["heur_min_rings"] = int(args.heur_min_rings)
    cfg = MultiObjConfig(**cfg_kwargs)

    # rewarder
    Rer = MultiObjectiveReward(
        model_path=args.predictor,
        stats_path=args.predictor_stats,
        cfg=cfg,
        gen_mode=GEN_MODE,
        featurizer=args.featurizer,
    )

    # let sampling gate reuse rewarder's encoder if available (optional, for speed)
    try:
        sample_gate.rer = Rer
    except Exception:
        pass

    # output sink
    sample_sink = None
    if args.save_samples:
        ensure_dir(os.path.dirname(args.save_samples) or ".")
        sample_sink = open(args.save_samples, "a", encoding="utf-8")

    total, batch, accum = int(args.iters), int(args.batch), max(1, int(args.accum))
    outer_steps = math.ceil(total / batch)
    print("[info] Start multi-objective RL (SELFIES)…")
    pbar = trange(outer_steps, dynamic_ncols=True)
    seen_samples, baseline = 0, None

    ZP = ZSource(args, device)

    # === 为“方案二/三”准备：训练缓存与最近 batch 信息 ===
    sample_buffer = SampleBuffer(max_size=int(getattr(args, "sample_buffer_size", 5000) or 0))
    last_batch_seqs: List[str] = []
    last_batch_rewards: List[float] = []
    last_batch_lens: List[int] = []

    def add_offline_sequences(
        k: int,
        seqs_all,
        logps_all,
        rewards_all,
        lens_all,
        ent_all,
        logits_paths_all,
    ) -> int:
        if k <= 0:
            return 0
        off_list = OFF.sample(k)
        injected = 0
        pulse_log_path = os.path.join(args.outdir, "offline_pulses.selfies")
        for s in off_list:
            logits_list = forward_logits_along(G, data, s, device)
            logp_sum = torch.tensor(0.0, device=device)
            toks = _split_tokens_like_generator(s) if ("[" in s and "]" in s) else list(s)
            steps = 0
            for t in range(min(len(toks) - 1, len(logits_list))):
                nxt = toks[t + 1]
                if nxt not in data.all_characters:
                    continue
                idx = data.all_characters.index(nxt)
                logp_sum = logp_sum + F.log_softmax(logits_list[t], dim=-1)[idx]
                steps += 1
            L = max(steps, 1)
            R, _detail = Rer.score_one(s)

            seqs_all.append(s)
            logps_all.append(logp_sum)
            rewards_all.append(float(R))
            lens_all.append(L)
            ent_all.append(torch.tensor(0.0, device=device))
            logits_paths_all.append(logits_list)

            # teacher diagnostics against current gate (only logs)
            if getattr(args, "sample_report", False) and sample_gate and sample_gate.enabled:
                try:
                    smi = _seq_to_smiles_local(s, GEN_MODE)
                    s_v, d2_v, _ = sample_gate._encode_to_s(smi)
                    thr_print = None
                    if getattr(sample_gate, "_s_thr_reward_eff", None) is not None:
                        thr_print = float(sample_gate._s_thr_reward_eff)
                    elif sample_gate.s_thr is not None:
                        thr_print = float(sample_gate.s_thr)
                    elif sample_gate.d2_thr is not None:
                        thr_print = -0.5 * float(sample_gate.d2_thr)
                    if s_v is not None and thr_print is not None:
                        verdict = "PASS" if s_v >= thr_print else "FAIL"
                        print(
                            f"[pulse] inject teacher: s={s_v:.3f} vs thr={thr_print:.3f} -> {verdict}"
                        )
                except Exception:
                    pass


            try:
                with open(pulse_log_path, "a", encoding="utf-8") as pf:
                    pf.write(s.strip("<>").replace(" ", "") + "\n")
            except Exception:
                pass
            injected += 1
        return injected

    for outer in pbar:
        last_pulse_count_total = 0
        for _ in range(accum):
            opt.zero_grad()
            seqs, logps_t, rewards, lens = [], [], [], []
            entropies, logits_paths = [], []

            # on-policy rollouts
            for _b in range(batch):
                z = ZP.sample()
                if z is not None:
                    G.set_condition(z, scale=args.z_scale)
                else:
                    G.clear_condition()

                seq, logp_sum, L, logits_list, ent_sum = rollout_with_logprobs(
                    G,
                    data,
                    max_len=args.max_len,
                    device=device,
                    top_p=args.top_p,
                    temperature=args.temp,
                    prime="<",
                    end_token=">",
                    eos_bias=args.eos_bias,
                    min_len=int(args.min_len or 0),
                    no_repeat_ngram=int(args.no_repeat_ngram or 0),
                    repeat_penalty=float(args.repeat_penalty or 1.0),
                )
                R, _detail = Rer.score_one(seq)

                seqs.append(seq)
                logps_t.append(logp_sum)
                rewards.append(float(R))
                lens.append(L)
                entropies.append(ent_sum)
                logits_paths.append(logits_list)

            # pulse teacher
            if int(args.pulse_every or 0) > 0 and ((outer + 1) % int(args.pulse_every) == 0):
                cnt = add_offline_sequences(
                    int(args.pulse_k or 0),
                    seqs,
                    logps_t,
                    rewards,
                    lens,
                    entropies,
                    logits_paths,
                )
                last_pulse_count_total += int(cnt)

            # === 关键：记录本次 batch（含 pulse），供方案一/二使用 ===
            last_batch_seqs = list(seqs)
            last_batch_rewards = list(rewards)
            last_batch_lens = list(lens)
            sample_buffer.add_many(seqs)

            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            avg_R = rewards_t.mean().item()
            baseline = avg_R if (baseline is None) else (0.95 * baseline + 0.05 * avg_R)
            adv_t = rewards_t - baseline

            logps_tensor = torch.stack(logps_t)
            L_t = torch.tensor(lens, device=device, dtype=torch.float32).clamp_min(1)
            policy_loss = -((logps_tensor / L_t) * adv_t.detach()).mean()

            if entropies:
                ent_mean = (torch.stack(entropies) / torch.tensor(lens, device=device, dtype=torch.float32)).mean()
            else:
                ent_mean = torch.tensor(0.0, device=device)
            entropy_loss = -float(args.entropy_coef or 0.0) * ent_mean

            kl_loss = torch.tensor(0.0, device=device)
            if getattr(G, "decoder", None) is not None and float(args.kl_coef or 0.0) > 0:
                if "G_ref" in locals() and (G_ref is not None):
                    kls = []
                    for seq_i, cur_logits in zip(seqs, logits_paths):
                        with torch.no_grad():
                            ref_logits = forward_logits_along(G_ref, data, seq_i, device)
                        T = min(len(cur_logits), len(ref_logits))
                        if T == 0:
                            continue
                        vals = []
                        for t in range(T):
                            p_log = F.log_softmax(cur_logits[t], dim=-1)
                            q_log = F.log_softmax(ref_logits[t], dim=-1)
                            p = torch.exp(p_log)
                            vals.append(torch.sum(p * (p_log - q_log)))
                        kls.append(torch.stack(vals).mean())
                    if kls:
                        kl_loss = float(args.kl_coef) * torch.stack(kls).mean()

            loss = policy_loss + kl_loss + entropy_loss
            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                clip_grad_norm_(G.parameters(), max_norm=args.clip_grad)
            opt.step()
            seen_samples += len(seqs)

        avg_len = float(sum(lens) / max(1, len(lens)))
        lr_show = opt.param_groups[0]["lr"]
        pulse_info = (
            "off"
            if int(args.pulse_every or 0) <= 0
            else f"{last_pulse_count_total}/{int(args.pulse_k or 0)}@{int(args.pulse_every or 0)}"
        )
        pbar.set_postfix(R=f"{avg_R:.3f}", len=f"{avg_len:.1f}", lr=lr_show, pulse=pulse_info)

        # log
        if (outer + 1) % 10 == 0 or seen_samples >= total:
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        seen_samples,
                        f"{avg_R:.6f}",
                        f"{avg_len:.3f}",
                        f"{time.time():.3f}",
                        f"{lr_show:.6g}",
                    ]
                )

        # =============== sampling & saving (gate applies here) ===============
        if seen_samples % max(1, args.sample_every) == 0:
            mode = getattr(args, "sample_mode", "new")
            candidates: List[str] = []

            # 方案一：last_batch —— 使用最近一次训练 batch（含 pulse）
            if mode == "last_batch":
                candidates = list(last_batch_seqs) if last_batch_seqs else []

            # 方案二：buffer —— 从训练缓存中采样
            elif mode == "buffer":
                # 默认抽 10 条；你也可以改成和 batch 一致
                candidates = sample_buffer.sample(k=10)

            # 若上述两种模式得到的候选为空，或明确选择 new，则回退到方案三
            if mode == "new" or not candidates:
                candidates = []
                for _ in range(10):
                    z = ZP.sample()
                    if z is not None:
                        G.set_condition(z, scale=args.z_scale)
                    else:
                        G.clear_condition()
                    seq, _, _, _, _ = rollout_with_logprobs(
                        G,
                        data,
                        max_len=args.max_len,
                        device=device,
                        top_p=args.top_p,
                        temperature=args.temp,
                        prime="<",
                        end_token=">",
                        eos_bias=args.eos_bias,
                        min_len=int(args.min_len or 0),
                        no_repeat_ngram=int(args.no_repeat_ngram or 0),
                        repeat_penalty=float(args.repeat_penalty or 1.0),
                    )
                    candidates.append(seq)

            # 去重 & 规范包裹
            if candidates:
                seen_set = set()
                uniq = []
                for s in candidates:
                    s_norm = ensure_wrapped_angles(s)
                    if s_norm in seen_set:
                        continue
                    seen_set.add(s_norm)
                    uniq.append(s_norm)
                candidates = uniq

            n_raw = len(candidates)

            # 门控（可设 warmup）
            if int(getattr(args, "sample_warmup", 0) or 0) > 0 and seen_samples < int(
                args.sample_warmup
            ):
                kept = list(candidates)
            else:
                kept, _ = sample_gate.filter_list(candidates)

            print(f"[sample] step={seen_samples} mode={mode} raw={n_raw} kept={len(kept)}")
            if args.sample_report and sample_gate.enabled:
                thr_info = ""
                if getattr(sample_gate, "_s_thr_reward_eff", None) is not None:
                    thr_info = f"s_thr={sample_gate._s_thr_reward_eff:.3f}"
                elif sample_gate.s_thr is not None:
                    thr_info = f"s_thr={sample_gate.s_thr:.3f}"
                elif sample_gate.d2_thr is not None:
                    thr_info = f"d2_thr={sample_gate.d2_thr:.3f}"
                print(f"[sample] gate_info: {thr_info}")


            if sample_sink:
                for s in kept:
                    sample_sink.write(s.strip("<>").replace(" ", "") + "\n")
                sample_sink.flush()

            # per-candidate debug print (metric-consistent, 只看 s vs thr，不再调用奖励，避免污染 unique 计数)
            if args.sample_report and sample_gate.enabled:
                print_thr = None
                if getattr(sample_gate, "_s_thr_reward_eff", None) is not None:
                    print_thr = float(sample_gate._s_thr_reward_eff)
                elif sample_gate.s_thr is not None:
                    print_thr = float(sample_gate.s_thr)
                elif sample_gate.d2_thr is not None:
                    print_thr = -0.5 * float(sample_gate.d2_thr)
                for s in candidates:
                    try:
                        smi_dbg = _seq_to_smiles_local(s, GEN_MODE)
                        sval, _, err = sample_gate._encode_to_s(smi_dbg)
                        line = f"[sample-detail] step={seen_samples} mode={mode} seq={s}"
                        if sval is None:
                            line += f" | s=None err={err}"
                        elif print_thr is not None:
                            line += (
                                f" | s={sval:.3f} vs thr={print_thr:.3f} -> "
                                f"{'PASS' if sval >= print_thr else 'FAIL'}"
                            )
                        else:
                            line += f" | s={sval:.3f} (no thr)"
                        print(line)
                    except Exception as e:
                        print(f"[sample-detail] step={seen_samples} seq={s} | score_error: {e}")

        # save ckpt
        if seen_samples % max(1, args.save_every) == 0:
            ckpt = os.path.join(args.outdir, f"rl_step_{seen_samples}.pt")
            save_checkpoint(G, ckpt)
            print(f"[ckpt] saved: {ckpt}")

        if seen_samples >= total:
            break

    final_ckpt = os.path.join(args.outdir, f"final_rl_step_{seen_samples}.pt")
    save_checkpoint(G, final_ckpt)
    print(f"[done] final checkpoint: {final_ckpt}")

    # close sink
    try:
        if sample_sink:
            sample_sink.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
