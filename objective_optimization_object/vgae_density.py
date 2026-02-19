#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust VGAE density scorer with:
- Ledoit-Wolf style covariance shrinkage
- Eigenvalue flooring (relative + absolute)
- Whitened Mahalanobis score: s = -0.5 * || W @ (z - mu) ||^2
- Quantile-based gating with linear warmup (q_start -> q_end)
- Optional scale calibration using a small batch of online z to match train-time variance
- Drop-in replacement for projects that import `VGAEDensityScorer` from vgae_density.py

CLI (prepare stats from z_bank):
  python vgae_density.py --z-bank ckpts/z_stats_16/z_bank.npy \
    --out ckpts/z_stats_16/gate_stats.npz --quantile 0.20 \
    --shrink 0.15 --eps-rel 0.02 --eps-abs 1e-3
"""

from __future__ import annotations
import os, sys, json, math, argparse
from typing import Tuple, Optional, Dict, Any

import numpy as np
import numpy.linalg as npl

def _shrink_cov(S: np.ndarray, alpha: float) -> np.ndarray:
    """Ledoit-Wolf style shrinkage toward scaled identity."""
    d = S.shape[0]
    trace = float(np.trace(S))
    tgt = (trace / d) * np.eye(d, dtype=S.dtype)
    return (1.0 - alpha) * S + alpha * tgt

def _eigh_floor(S: np.ndarray, eps_rel: float, eps_abs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose and floor eigenvalues to avoid tiny variances."""
    ev, U = npl.eigh(S)
    med = float(np.median(ev))
    floor_val = max(eps_abs, eps_rel * med)
    ev = np.maximum(ev, floor_val)
    return ev, U

def _make_whitener(ev: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Return W such that ||W @ x||^2 = x^T S^{-1} x ."""
    inv_sqrt = np.diag(1.0 / np.sqrt(ev))
    # W = inv_sqrt @ U^T
    return inv_sqrt @ U.T

def _mahal2(W: np.ndarray, x: np.ndarray) -> float:
    y = W @ x
    return float(y @ y)

def _batch_mahal2(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Y = (W @ (X.T)).T  # [n, d]
    return np.einsum("nd,nd->n", Y, Y)

class VGAEDensityScorer:
    """
    Drop-in scorer.

    Public attrs:
      mu: (d,) mean
      cov: (d,d) shrunk & floored covariance actually used
      W:  (d,d) whitener so that d2 = ||W @ (z - mu)||^2
      s_thr, d2_thr: thresholds from quantile over training z_bank
      q_start, q_end, warmup_steps: schedule for gating quantile (optional)
      z_scale: extra multiplicative scale on (z - mu) to fix runtime/train mismatch

    Methods:
      score(z) -> (s, d2)
      batch_score(Z) -> (s_arr, d2_arr)
      gate(s or d2, step) -> 0/1 with warmup schedule
    """

    def __init__(self,
                 mu: np.ndarray,
                 cov_raw: np.ndarray,
                 quantile: float = 0.10,
                 shrink: float = 0.10,
                 eps_rel: float = 0.02,
                 eps_abs: float = 1e-3,
                 z_scale: float = 1.0,
                 q_start: Optional[float] = None,
                 q_end: Optional[float] = None,
                 warmup_steps: int = 0,
                 train_z_sample: Optional[np.ndarray] = None):
        assert mu.ndim == 1
        assert cov_raw.shape == (mu.size, mu.size)
        self.mu = mu.astype(np.float32)
        self.d = mu.size
        self.quantile = float(quantile)
        self.q_start = float(q_start) if q_start is not None else float(quantile)
        self.q_end   = float(q_end)   if q_end   is not None else float(quantile)
        self.warmup_steps = int(warmup_steps)
        self.z_scale = float(z_scale)

        # --- covariance processing ---
        S = cov_raw.astype(np.float64)
        if shrink > 0:
            S = _shrink_cov(S, float(shrink))
        ev, U = _eigh_floor(S, eps_rel=float(eps_rel), eps_abs=float(eps_abs))
        self.cov = (U @ np.diag(ev) @ U.T).astype(np.float32)
        self.ev = ev.astype(np.float32)
        self.U = U.astype(np.float32)
        self.W = _make_whitener(ev, U).astype(np.float32)  # whitening for S^{-1}

        # --- thresholds from train_z_sample if provided ---
        self.s_thr = None
        self.d2_thr = None
        if train_z_sample is not None and train_z_sample.size > 0:
            s_arr, d2_arr = self.batch_score(train_z_sample)
            q = np.clip(self.quantile, 0.0, 1.0)
            self.s_thr = float(np.quantile(s_arr, q))
            self.d2_thr = float(np.quantile(d2_arr, 1.0 - q))
        # logging
        ev_min, ev_max = float(ev.min()), float(ev.max())
        print(f"[VGAEDensityScorer] cov eig: min={ev_min:.3e}, max={ev_max:.3e}")
        if self.s_thr is not None:
            print(f"[VGAEDensityScorer] quantile={self.quantile:.3f} -> s_thr={self.s_thr:.6f}, d2_thr={self.d2_thr:.2f}")

    # -------- scoring ----------
    def _delta(self, z: np.ndarray) -> np.ndarray:
        return (z.astype(np.float32) - self.mu) * self.z_scale

    def score(self, z: np.ndarray) -> Tuple[float, float]:
        dz = self._delta(z)
        d2 = _mahal2(self.W, dz)
        s  = -0.5 * d2
        return float(s), float(d2)

    def batch_score(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        DZ = (Z.astype(np.float32) - self.mu) * self.z_scale
        d2 = _batch_mahal2(self.W, DZ)
        s  = -0.5 * d2
        return s.astype(np.float32), d2.astype(np.float32)

    # -------- gating -----------
    def _curr_q(self, step: int) -> float:
        if self.warmup_steps <= 0: return self.quantile
        step = max(0, int(step))
        t = min(1.0, step / float(self.warmup_steps))
        return (1.0 - t) * self.q_start + t * self.q_end

    def _curr_thresholds(self, step: int, train_s: np.ndarray, train_d2: np.ndarray) -> Tuple[float, float]:
        q = self._curr_q(step)
        s_thr = float(np.quantile(train_s, q))
        d2_thr = float(np.quantile(train_d2, 1.0 - q))
        return s_thr, d2_thr

    def gate(self, s: float = None, d2: float = None,
             step: Optional[int] = None,
             train_s: Optional[np.ndarray] = None,
             train_d2: Optional[np.ndarray] = None) -> int:
        """
        If step & train stats are provided, uses scheduled thresholds; otherwise fall back to fixed ones.
        """
        if step is not None and train_s is not None and train_d2 is not None:
            s_thr, d2_thr = self._curr_thresholds(step, train_s, train_d2)
        else:
            s_thr = self.s_thr if self.s_thr is not None else -np.inf
            d2_thr = self.d2_thr if self.d2_thr is not None else np.inf

        ok = True
        if s is not None:   ok = ok and (s >= s_thr)
        if d2 is not None:  ok = ok and (d2 <= d2_thr)
        return 1 if ok else 0

    # -------- calibration (optional) ----------
    def calibrate_scale(self, online_z: np.ndarray, train_z: np.ndarray, robust: bool = True) -> float:
        """
        Match variance (median eigenvalue proxy) between online_z and train_z in whitened space,
        returns multiplicative z_scale to apply henceforth.
        """
        if online_z.shape[0] < 16 or train_z.shape[0] < 16:
            return self.z_scale
        _, d2_online = self.batch_score(online_z)
        _, d2_train = self.batch_score(train_z)

        # in whitened space, E[d2] ~ d if matched; use robust ratio on medians
        med_online = float(np.median(d2_online))
        med_train  = float(np.median(d2_train))
        if med_online <= 0 or med_train <= 0:
            return self.z_scale
        # scaling on z multiplies d2 by scale^2
        s = math.sqrt(med_train / med_online)
        s = float(np.clip(s, 0.25, 4.0)) if robust else float(s)
        self.z_scale *= s
        print(f"[VGAEDensityScorer] calibrated z_scale *= {s:.3f} -> {self.z_scale:.3f}")
        return self.z_scale

    # -------- serialization --------
    def to_npz(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path,
            mu=self.mu, cov=self.cov, ev=self.ev, U=self.U, W=self.W,
            s_thr=self.s_thr if self.s_thr is not None else np.nan,
            d2_thr=self.d2_thr if self.d2_thr is not None else np.nan,
            z_scale=self.z_scale,
            meta=json.dumps(extra or {}))
        print(f"[VGAEDensityScorer] saved → {path}")

    @staticmethod
    def from_npz(path: str) -> "VGAEDensityScorer":
        data = np.load(path, allow_pickle=True)
        mu = data["mu"]
        cov = data["cov"]
        ev = data["ev"]; U = data["U"]; W = data["W"]
        s_thr = float(data["s_thr"]) if "s_thr" in data.files else np.nan
        d2_thr = float(data["d2_thr"]) if "d2_thr" in data.files else np.nan
        z_scale = float(data["z_scale"]) if "z_scale" in data.files else 1.0
        obj = VGAEDensityScorer(mu, cov, quantile=0.1, shrink=0.0, eps_rel=0.0, eps_abs=0.0, z_scale=z_scale)
        obj.ev, obj.U, obj.W = ev, U, W
        obj.s_thr = None if not np.isfinite(s_thr) else s_thr
        obj.d2_thr = None if not np.isfinite(d2_thr) else d2_thr
        return obj

    @staticmethod
    def fit_from_zbank(z_bank: np.ndarray,
                       quantile: float = 0.10,
                       shrink: float = 0.10,
                       eps_rel: float = 0.02,
                       eps_abs: float = 1e-3,
                       q_start: Optional[float] = None,
                       q_end: Optional[float] = None,
                       warmup_steps: int = 0,
                       z_scale: float = 1.0) -> "VGAEDensityScorer":
        assert z_bank.ndim == 2
        mu = z_bank.mean(axis=0)
        cov_raw = np.cov(z_bank.T, bias=False)
        scorer = VGAEDensityScorer(mu, cov_raw,
                                   quantile=quantile, shrink=shrink,
                                   eps_rel=eps_rel, eps_abs=eps_abs,
                                   z_scale=z_scale,
                                   q_start=q_start, q_end=q_end,
                                   warmup_steps=warmup_steps,
                                   train_z_sample=z_bank)
        return scorer

# ---------------- CLI: prepare stats from z_bank ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-bank", type=str, required=True, help="np.ndarray (n, d) of train z (μ encodings)")
    ap.add_argument("--out", type=str, required=True, help="path to save gate_stats.npz")
    ap.add_argument("--quantile", type=float, default=0.10)
    ap.add_argument("--shrink", type=float, default=0.10)
    ap.add_argument("--eps-rel", type=float, default=0.02)
    ap.add_argument("--eps-abs", type=float, default=1e-3)
    ap.add_argument("--q-start", type=float, default=None)
    ap.add_argument("--q-end", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--z-scale", type=float, default=1.0)
    args = ap.parse_args()

    Z = np.load(args.z_bank).astype(np.float32)
    print(f"[load] z_bank: shape={Z.shape}, mean_norm={np.linalg.norm(Z.mean(axis=0)):.3f}")
    scorer = VGAEDensityScorer.fit_from_zbank(
        Z, quantile=args.quantile, shrink=args.shrink,
        eps_rel=args.eps_rel, eps_abs=args.eps_abs,
        q_start=args.q_start, q_end=args.q_end,
        warmup_steps=args.warmup_steps, z_scale=args.z_scale
    )
    # Save with train thresholds baked in
    scorer.to_npz(args.out, extra=dict(
        quantile=args.quantile, shrink=args.shrink,
        eps_rel=args.eps_rel, eps_abs=args.eps_abs,
        q_start=args.q_start, q_end=args.q_end,
        warmup_steps=args.warmup_steps, z_scale=args.z_scale
    ))

if __name__ == "__main__":
    main()
