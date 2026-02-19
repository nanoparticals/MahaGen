# energetic_heuristics.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

# --- deps ---
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

try:
    import selfies as sf
    def _decode_selfies(x: str) -> str:
        try:
            return sf.decoder(x)
        except Exception:
            return x
except Exception:
    def _decode_selfies(x: str) -> str:
        return x


# ---------------- Utils ----------------
def _safe_mol(seq: str, gen_mode: str = "smiles") -> Chem.Mol | None:
    if gen_mode.lower() == "selfies":
        seq = _decode_selfies(seq)
    mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def _soft_ge(x: float, thr: float, k: float = 6.0) -> float:
    # reward increases as x >= thr
    return _sigmoid(k * (x - thr))

def _soft_le(x: float, thr: float, k: float = 6.0) -> float:
    # reward increases as x <= thr
    return _sigmoid(k * (thr - x))

def _soft_between(x: float, lo: float, hi: float, k: float = 6.0) -> float:
    return min(_soft_ge(x, lo, k), _soft_le(x, hi, k))

def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


# ---------------- Core metrics ----------------
def oxygen_balance_ob100(mol: Chem.Mol) -> float:
    """
    经典氧平衡（以 100 g 为基准）：
    OB_100 = 1600/MW * (n_O - 2*n_C - 0.5*n_H)
    """
    molH = Chem.AddHs(mol)
    nC = sum(1 for a in molH.GetAtoms() if a.GetAtomicNum() == 6)
    nH = sum(1 for a in molH.GetAtoms() if a.GetAtomicNum() == 1)
    nO = sum(1 for a in molH.GetAtoms() if a.GetAtomicNum() == 8)
    mw = Descriptors.MolWt(molH)
    if mw <= 0:
        return -100.0
    ob = 1600.0 / mw * (nO - 2.0 * nC - 0.5 * nH)
    return float(ob)


# Nitro family: nitro (C–NO2 / Ar–NO2)、nitramine (–NH–NO2)、nitrate ester (–ONO2)
_SMARTS_NITRO = Chem.MolFromSmarts("[N+](=O)[O-]")  # core –N(=O)=O
def count_nitro_families(mol: Chem.Mol) -> Dict[str, int]:
    nitro_total = 0
    nitro_c = 0          # C–NO2
    nitramine = 0        # N–NO2
    nitrate = 0          # O–NO2
    if _SMARTS_NITRO is None:
        return dict(nitro_total=0, nitro_c=0, nitramine=0, nitrate=0)
    for match in mol.GetSubstructMatches(_SMARTS_NITRO, uniquify=True):
        n_idx = match[0]  # the N in –N(=O)=O
        n_atom = mol.GetAtomWithIdx(n_idx)
        nitro_total += 1
        # The third neighbor is the attachment atom (besides the two O in =O and –O-)
        # Identify one non-oxygen neighbor
        attach_type = None
        for nb in n_atom.GetNeighbors():
            if nb.GetAtomicNum() != 8:  # not an oxygen
                attach_type = nb.GetAtomicNum()
                break
        if attach_type == 6:
            nitro_c += 1
        elif attach_type == 7:
            nitramine += 1
        elif attach_type == 8:
            nitrate += 1
        else:
            # rare cases (S/P/halogens), ignore but still count in nitro_total
            pass
    return dict(nitro_total=nitro_total, nitro_c=nitro_c,
                nitramine=nitramine, nitrate=nitrate)


def ring_metrics(mol: Chem.Mol) -> Dict[str, int | float]:
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    # hetero rings & fused pairs
    rings = [set(r) for r in Chem.GetSymmSSSR(mol)]
    hetero = 0
    fused_pairs = 0
    atoms = mol.GetAtoms()
    for i, r in enumerate(rings):
        if any(atoms[idx].GetAtomicNum() not in (1, 6) for idx in r):
            hetero += 1
        for j in range(i + 1, len(rings)):
            if len(r & rings[j]) > 0:
                fused_pairs += 1
    return dict(num_rings=int(num_rings),
                num_aromatic=int(num_arom),
                num_hetero=int(hetero),
                fused_pairs=int(fused_pairs))


def compactness_proxy(mol: Chem.Mol) -> float:
    """
    一个稳健的“紧凑度”代理：[更多环/稠环/芳环，较少可旋转键，较小图直径] → 分数更高
    """
    rm = ring_metrics(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    try:
        dmat = Chem.GetDistanceMatrix(mol)
        diameter = float(np.max(dmat)) if dmat.size else 0.0
    except Exception:
        diameter = 0.0

    score = (
        0.55 * (rm["num_rings"] + 0.8 * rm["num_aromatic"] + 1.2 * rm["fused_pairs"] + 0.7 * rm["num_hetero"])
        - 0.40 * rot
        - 1.10 * (diameter / max(1.0, heavy))
    )
    # squash to 0..1
    return _clip01(_sigmoid(score))


def hydrogen_fraction(mol: Chem.Mol) -> float:
    """H/(H+heavy) —— 含氢分数，越小越有利于高ρ0与更好的氧平衡"""
    molH = Chem.AddHs(mol)
    nH = sum(1 for a in molH.GetAtoms() if a.GetAtomicNum() == 1)
    heavy = mol.GetNumHeavyAtoms()
    if nH + heavy <= 0:
        return 0.0
    return float(nH / (nH + heavy))


# ---------------- Scorer ----------------
@dataclass
class EnergeticHeuristicsCfg:
    # 氧平衡的目标窗口（常用 -20% ~ +5%）
    ob_lo: float = -20.0
    ob_hi: float = 5.0
    ob_k: float = 6.0

    # 官能团最少计数（鼓励硝基家族）
    min_nitro_total: int = 3
    min_nitramine: int = 1
    min_nitrate: int = 1

    # 环系指标
    min_rings: int = 2
    min_hetero_rings: int = 1

    # 含氢分数（希望尽量偏小）
    max_hfrac: float = 0.33

    # 各项权重（可以在外面再乘一个总权重）
    w_ob: float = 0.6
    w_nitro: float = 0.4
    w_rings: float = 0.25
    w_compact: float = 0.25
    w_hfrac: float = 0.3


class EnergeticHeuristics:
    def __init__(self, cfg: EnergeticHeuristicsCfg | None = None):
        self.cfg = cfg or EnergeticHeuristicsCfg()

    def score_mol(self, mol: Chem.Mol) -> Dict[str, float]:
        if mol is None:
            return dict(R_OB=0.0, R_NOx=0.0, R_Rings=0.0, R_Compact=0.0, R_Hfrac=0.0, R_sum=0.0)

        # 1) 氧平衡（窗口奖励）
        ob = oxygen_balance_ob100(mol)
        R_OB = _soft_between(ob, self.cfg.ob_lo, self.cfg.ob_hi, self.cfg.ob_k)

        # 2) 硝基家族（梯度化地鼓励“至少达到 X 个”）
        nit = count_nitro_families(mol)
        r_nitro_total = _soft_ge(nit["nitro_total"], self.cfg.min_nitro_total, k=4.0)
        r_nitramine   = _soft_ge(nit["nitramine"],   self.cfg.min_nitramine,   k=5.0)
        r_nitrate     = _soft_ge(nit["nitrate"],     self.cfg.min_nitrate,     k=5.0)
        # 核心是“总硝基充足”，辅以“硝胺 / 硝酸酯”的存在
        R_NOx = _clip01(0.5 * r_nitro_total + 0.25 * r_nitramine + 0.25 * r_nitrate)

        # 3) 环 / 稠环 / 杂环
        rm = ring_metrics(mol)
        r_rings  = _soft_ge(rm["num_rings"], self.cfg.min_rings, k=5.0)
        r_hrings = _soft_ge(rm["num_hetero"], self.cfg.min_hetero_rings, k=5.0)
        r_fused  = _clip01(min(1.0, rm["fused_pairs"] / 2.0))  # >=2 对计满
        R_Rings = _clip01(0.45 * r_rings + 0.35 * r_hrings + 0.20 * r_fused)

        # 4) 紧凑度代理（已是 0..1）
        R_Compact = compactness_proxy(mol)

        # 5) 含氢分数（越小越好 → 惩罚超标）
        hfrac = hydrogen_fraction(mol)
        R_Hfrac = _soft_le(hfrac, self.cfg.max_hfrac, k=8.0)

        # 线性拼接
        R_sum = (
            self.cfg.w_ob * R_OB
            + self.cfg.w_nitro * R_NOx
            + self.cfg.w_rings * R_Rings
            + self.cfg.w_compact * R_Compact
            + self.cfg.w_hfrac * R_Hfrac
        )
        return dict(R_OB=float(R_OB), R_NOx=float(R_NOx), R_Rings=float(R_Rings),
                    R_Compact=float(R_Compact), R_Hfrac=float(R_Hfrac),
                    R_sum=float(R_sum))

    def score(self, seq: str, gen_mode: str = "smiles") -> Dict[str, float]:
        mol = _safe_mol(seq, gen_mode)
        return self.score_mol(mol)
