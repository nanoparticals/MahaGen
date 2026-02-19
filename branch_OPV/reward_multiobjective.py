#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, tempfile, csv
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from collections import OrderedDict
import copy
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem

# SELFIES 可选
try:
    import selfies as sf
except Exception:
    class sf:  # type: ignore
        @staticmethod
        def decoder(x: str) -> str:
            return x

# VGAE 相关（可选）
try:
    from vgae_density import VGAEDensityScorer as _VGAEDensityScorer  # type: ignore
except Exception:
    _VGAEDensityScorer = None  # type: ignore

try:
    from z_generator_stub import load_vgae_model as _load_vgae_model  # type: ignore
except Exception:
    _load_vgae_model = None  # type: ignore


# ======================
# Config
# ======================
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MultiObjConfig:
    targets: List[str]
    goals:   List[str]
    weights: List[float]

    target_values: Optional[List[Optional[float]]] = None
    use_zscore: bool = False
    zscore: Optional[bool] = None

    invalid_penalty: float = -10.0
    unique_bonus: float = 0.0

    predictor_path: Optional[str] = None
    predictor_stats_path: Optional[str] = None
    featurizer: str = "morgan2048"
    predictor_output_transform: Optional[str] = None
    predictor_transform_targets: Optional[List[str]] = None

    # VGAE 统计/密度
    vgae_ckpt: Optional[str] = None
    vgae_mean_path: Optional[str] = None
    vgae_cov_path: Optional[str]  = None
    vgae_zbank_path: Optional[str] = None
    vgae_weight: float = 0.0
    vgae_gate: str = "off"          # "off" | "threshold:<s_thr>" | "quantile:<q>"
    vgae_warmup: int = 0
    vgae_anneal: Optional[str] = None  # "cosine:lo,hi[,steps]"

    # 统一门控度量 & 共享阈值
    gate_metric: str = "euclid"                  # "euclid" | "maha"
    vgae_gate_stats_path: Optional[str] = None   # gate_stats.npz（含 mu/cov/Si/s_thr/可选 teacher_s_quantiles）
    reward_tighten_q: Optional[float] = None     # 奖励端二次收紧分位数（例如 0.35）

    # 教师分布重算（当 gate_stats 无分位数表时）
    offline_csv: Optional[str] = None
    offline_col: str = "selfies"

    # 长度 / 体量
    len_target: Optional[int] = None
    len_lambda: float = 0.0
    ha_target: Optional[int] = None
    ha_lambda: float = 0.0

    # 重复惩罚
    dupe_penalty: float = 0.0
    dupe_escalate: bool = False
    dupe_window: int = 5000
    dupe_canonical: bool = True

    # VGAE 分数标准化（保留选项；默认不强依赖）
    vgae_center: bool = True
    vgae_zscore: bool = True

    # 启发式（可选）
    heur_weight: float = 1.0
    heur_ob_lo: float = -20.0
    heur_ob_hi: float = 5.0
    heur_max_hfrac: float = 0.33
    heur_min_nitro_total: int = 3
    heur_min_nitramine: int = 1
    heur_min_nitrate: int = 1
    heur_min_rings: int = 1
    heur_min_hetero_rings: int = 1

    heur_w_ob: float = 0.6
    heur_w_nitro: float = 0.4
    heur_w_rings: float = 0.5
    heur_w_compact: float = 0.25
    heur_w_hfrac: float = 0.3


# ======================
# Featurization / Predictor
# ======================
def _morgan2048(smiles_list: List[str]) -> np.ndarray:
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            fps.append(np.zeros(2048, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
            arr = np.zeros((1, 2048), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr[0])
            fps.append(arr[0].astype(np.float32))
    return np.vstack(fps)


def _featurize(smiles_list: List[str], name: str) -> np.ndarray:
    name = (name or "none").lower()
    if name in ("none", "", "off"):
        raise RuntimeError("featurizer disabled")
    if name == "morgan2048":
        return _morgan2048(smiles_list)
    raise ValueError(f"Unknown featurizer: {name}")


class _PredictorWrapper:
    def __init__(self, model_path: str, featurizer: str = "morgan2048"):
        self.model_path = model_path
        self.featurizer = featurizer
        self.backend = "chemprop"  # 强制 chemprop

        # -----------------------------------------------------------
        # 1. 内部辅助函数 (保持在你原来的位置)
        # -----------------------------------------------------------
        def _find_chemprop_args_json(p: str) -> Optional[str]:
            import glob, os
            d = p if os.path.isdir(p) else os.path.dirname(p)
            cands = []
            for depth in range(3):  # 当前/父级/祖父级搜 args.json
                dd = d if depth == 0 else os.path.abspath(os.path.join(d, *([".."] * depth)))
                cands += glob.glob(os.path.join(dd, "args.json"))
                cands += glob.glob(os.path.join(dd, "train_args.json"))
            return cands[0] if cands else None

        # -----------------------------------------------------------
        # 2. 检查路径与环境
        # -----------------------------------------------------------
        is_pt = str(model_path).endswith(".pt")
        is_dir = os.path.isdir(model_path)

        if not (is_pt or is_dir):
            raise ValueError(
                f"强制使用 Chemprop：predictor 路径必须是 .pt 或 checkpoint_dir，但得到: {model_path}"
            )

        try:
            import chemprop  # noqa: F401
            from chemprop.train import make_predictions
            from chemprop.args import PredictArgs
        except Exception as e:
            raise RuntimeError(
                "强制使用 Chemprop 但当前环境无法导入 chemprop 或其接口不匹配。"
            ) from e

        self._cp_make_predictions = make_predictions

        # -----------------------------------------------------------
        # 3. 读取 args.json 配置 (为了获取 features_generator)
        # -----------------------------------------------------------
        feats_gen: List[str] = []
        no_feats_scaling: bool = False
        
        aj = _find_chemprop_args_json(model_path)
        if aj and os.path.exists(aj):
            try:
                with open(aj, "r", encoding="utf-8") as f:
                    train_cfg = json.load(f)
                fg = train_cfg.get("features_generator") or train_cfg.get("features_generators")
                if isinstance(fg, str):
                    feats_gen = [fg]
                elif isinstance(fg, list):
                    feats_gen = fg

                if "features_scaling" in train_cfg:
                    no_feats_scaling = (not bool(train_cfg["features_scaling"]))
                elif "no_features_scaling" in train_cfg:
                    no_feats_scaling = bool(train_cfg["no_features_scaling"])
            except Exception:
                pass

        # -----------------------------------------------------------
        # 4. 【修复核心】构建参数列表
        # -----------------------------------------------------------
        # 修复逻辑：在最开始初始化 args_list，绝不在中间重置它
        args_list: List[str] = []

        # (A) 添加模型路径
        if is_dir:
            args_list += ["--checkpoint_dir", model_path]
        else:
            args_list += ["--checkpoint_path", model_path]

        # (B) 添加特征生成器 (如果有)
        if feats_gen:
            for fg in feats_gen:
                args_list += ["--features_generator", fg]
            
            # (C) 处理特征缩放逻辑
            if "rdkit_2d_normalized" in feats_gen:
                args_list += ["--no_features_scaling"]
            elif no_feats_scaling:
                args_list += ["--no_features_scaling"]

        # (D) 准备临时文件 (chemprop 需要)
        self._tmpdir = tempfile.mkdtemp(prefix="chemprop_tmp_")
        self._test_csv = os.path.join(self._tmpdir, "test.csv")
        self._preds_csv = os.path.join(self._tmpdir, "preds.csv")
        
        # 写入一个 dummy 文件通过检查
        with open(self._test_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["smiles"])
            w.writerow(["C"])

        # (E) 添加通用预测参数
        args_list += [
            "--test_path", self._test_csv,
            "--preds_path", self._preds_csv,
            "--num_workers", "0",
            "--batch_size", "256",
        ]

        # -----------------------------------------------------------
        # 5. 解析参数
        # -----------------------------------------------------------
        try:
            self._cp_args = PredictArgs().parse_args(args_list)
            # 修正列名，防止报错
            if hasattr(self._cp_args, "smiles_columns"):
                self._cp_args.smiles_columns = ["smiles"]
            else:
                self._cp_args.smiles_column = "smiles"
        except Exception as e:
            # 打印调试信息，万一还有错能看到参数长什么样
            print(f"DEBUG: Failed args_list: {args_list}")
            raise RuntimeError(
                "Chemprop PredictArgs 构建失败（可能是 chemprop 版本/API 不匹配，或参数不兼容）。"
            ) from e

    # -----------------------------------------------------------
    # 注意：predict 方法必须与 __init__ 对齐 (缩进4个空格)，不能在 __init__ 里面
    # -----------------------------------------------------------
    def predict(self, smiles: List[str]) -> np.ndarray:
            try:
                # 必须有这一句深拷贝！
                args_copy = copy.deepcopy(self._cp_args) 
                
                smi2d = [[s] for s in smiles]
                preds = self._cp_make_predictions(args=args_copy, smiles=smi2d)
                return np.asarray(preds, dtype=np.float32)
            except Exception as e:
                # 必须有这个报错打印，万一再崩，我们能看到是哪个分子搞的鬼
                print(f"\n[CRITICAL ERROR] Crashed on: {smiles[:3]}...")
                import traceback
                traceback.print_exc()
                raise e
    

    @staticmethod
    def _extract_predictor(obj: Any) -> Any:  # type: ignore[name-defined]
        if hasattr(obj, "predict"):
            return obj
        if isinstance(obj, dict):
            for k in ("model", "pipeline", "predictor", "estimator"):
                v = obj.get(k)
                if v is not None and hasattr(v, "predict"):
                    return v
            if "models" in obj and isinstance(obj["models"], (list, tuple)):
                models = [m for m in obj["models"] if hasattr(m, "predict")]
                class _Ensemble:
                    def __init__(self, ms): self.ms = ms
                    def predict(self, X):
                        ys = [np.asarray(m.predict(X)) for m in self.ms]
                        return np.mean(np.stack(ys, axis=0), axis=0)
                return _Ensemble(models)
        raise TypeError("Loaded predictor is not usable; need .predict or a dict containing it.")

# ======================
# Stats + Gate
# ======================
class Stats:
    def __init__(self, means: Dict[str, float], stds: Dict[str, float]):
        self.means = {k: float(v) for k, v in means.items()}
        self.stds  = {k: (1.0 if (v is None or float(v) == 0.0) else float(v)) for k, v in stds.items()}

    @staticmethod
    def from_json(path: Optional[str], targets: List[str]) -> "Stats":
        if not path:
            means = {t: 0.0 for t in targets}; stds = {t: 1.0 for t in targets}
            return Stats(means, stds)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "means" in obj and "stds" in obj:
            means = {t: float(obj["means"][t]) for t in targets}
            stds  = {t: float(obj["stds"][t])  for t in targets}
        elif isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
            means = {t: float(obj.get(t, {}).get("mean", 0.0)) for t in targets}
            stds  = {t: float(obj.get(t, {}).get("std",  1.0)) for t in targets}
        elif isinstance(obj, dict) and any(k.endswith("_mean") for k in obj.keys()):
            means = {t: float(obj.get(f"{t}_mean", 0.0)) for t in targets}
            stds  = {t: float(obj.get(f"{t}_std",  1.0)) for t in targets}
        elif isinstance(obj, list):
            mp = {d["target"]: float(d["mean"]) for d in obj}
            sp = {d["target"]: float(d["std"])  for d in obj}
            means = {t: mp[t] for t in targets}
            stds  = {t: sp[t] for t in targets}
        else:
            raise ValueError("stats.json 结构不被识别；需要 mean/std per target。")
        return Stats(means, stds)


class VGaeGate:
    def __init__(self, mode: str, threshold: float):
        self.mode = mode
        self.threshold = float(threshold)

    @staticmethod
    def from_string(spec: str) -> Optional["VGaeGate"]:
        spec = (spec or "off").lower().strip()
        if spec == "off":
            return None
        if spec.startswith("threshold:"):
            thr = float(spec.split(":", 1)[1])
            return VGaeGate("threshold", thr)
        return None

    def check(self, s: float) -> Tuple[bool, Dict[str, Any]]:
        ok = (float(s) >= self.threshold)
        return ok, {"gate": self.mode, "threshold": float(self.threshold)}


# ======================
# Reward
# ======================
class MultiObjectiveReward:
    def __init__(self, model_path: str, stats_path: str, cfg: MultiObjConfig,
                 gen_mode: str = "smiles", featurizer: str = "auto"):
        self.cfg = cfg
        self.gen_mode = gen_mode.lower()
        self.featurizer = featurizer

        path = model_path or (cfg.predictor_path or "")
        if not path:
            raise ValueError("必须提供 predictor 模型路径")
        self.pred = _PredictorWrapper(path, cfg.featurizer if featurizer == "auto" else featurizer)

        spath = stats_path or cfg.predictor_stats_path
        self.stats = Stats.from_json(spath, cfg.targets)

        self.mu = None; self.Si = None
        if cfg.vgae_mean_path and cfg.vgae_cov_path and os.path.exists(cfg.vgae_mean_path) and os.path.exists(cfg.vgae_cov_path):
            self.mu = np.load(cfg.vgae_mean_path).astype(np.float32)
            cov = np.load(cfg.vgae_cov_path).astype(np.float32)
            self.Si = np.linalg.inv(cov + np.eye(cov.shape[0], dtype=cov.dtype) * 1e-8)

        self._vgae_encoder = None
        if cfg.vgae_ckpt and _load_vgae_model is not None:
            try:
                import torch
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._vgae_encoder = _load_vgae_model(cfg.vgae_ckpt, dev)
            except Exception:
                self._vgae_encoder = None

        self._gate_spec = (cfg.vgae_gate or "off").lower().strip()
        if self._gate_spec.startswith("quantile:") and (cfg.vgae_zbank_path and self.mu is not None and self.Si is not None):
            try:
                q = float(self._gate_spec.split(":", 1)[1])
                Z = np.load(cfg.vgae_zbank_path).astype(np.float32)
                dif = Z - self.mu
                maha = np.einsum("ni,ij,nj->n", dif, self.Si, dif)
                scores = -0.5 * maha
                thr = float(np.quantile(scores, q))
                self._gate_spec = f"threshold:{thr:.6f}"
            except Exception:
                pass

        self.gate = VGaeGate.from_string(self._gate_spec)
        self._call_count = 0

        self.t2i = {t: i for i, t in enumerate(cfg.targets)}

        # --- 唯一/重复跟踪结构 ---
        # 用于 unique_bonus 的集合仍然保留（向后兼容）
        self._seen_unique: set[str] = set()
        # 用于重复惩罚的 LRU 计数字典
        self._seen_counts: OrderedDict[str, int] = OrderedDict()

    # 兼容 use_zscore / zscore
    def _use_z(self) -> bool:
        uz = getattr(self.cfg, "use_zscore", False)
        z = getattr(self.cfg, "zscore", None)
        return bool(uz or (z is True))

    def _to_canonical_smiles(self, seq: str) -> Optional[str]:
        core = seq.strip("<>").replace(" ", "")
        try:
            smi = sf.decoder(core) if self.gen_mode == "selfies" else core
        except Exception:
            smi = core
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    def _noncanonical_key(self, seq: str) -> Optional[str]:
        """当 dupe_canonical=False 时，用“非规范化口径”的 key 判重。"""
        core = seq.strip("<>").replace(" ", "")
        try:
            return sf.decoder(core) if self.gen_mode == "selfies" else core
        except Exception:
            return core

    def _vgae_score(self, smiles: str) -> Optional[float]:
        if _VGAEDensityScorer is not None and self.cfg.vgae_ckpt:
            try:
                scorer = getattr(self, "_density_scorer", None)
                if scorer is None:
                    self._density_scorer = _VGAEDensityScorer(
                        ckpt_path=self.cfg.vgae_ckpt,
                        mean_path=self.cfg.vgae_mean_path,
                        cov_path=self.cfg.vgae_cov_path
                    )
                    scorer = self._density_scorer
                s = scorer.score_smiles(smiles)
                return float(s)
            except Exception:
                pass
        if self._vgae_encoder is not None and (self.mu is not None) and (self.Si is not None):
            try:
                z = None
                if hasattr(self._vgae_encoder, "encode_smiles"):
                    z = self._vgae_encoder.encode_smiles(smiles)
                elif hasattr(self._vgae_encoder, "encode_batch"):
                    zz = self._vgae_encoder.encode_batch([smiles])
                    z = zz[0] if zz is not None and len(zz) > 0 else None
                if z is None:
                    return None
                z = np.asarray(z, dtype=np.float32).reshape(-1)
                dif = z - self.mu
                maha = float(dif @ self.Si @ dif)
                return -0.5 * maha
            except Exception:
                return None
        return None

    def _goal_score(self, t: str, used: float, raw: float, idx: int) -> float:
        g = self.cfg.goals[idx].lower()
        if g == "max":
            return used
        if g == "min":
            return -used
        if g == "target":
            tv = None
            if self.cfg.target_values is not None and idx < len(self.cfg.target_values):
                tv = self.cfg.target_values[idx]
            if self._use_z():
                target_z = 0.0 if tv is None else (tv - self.stats.means[t]) / self.stats.stds[t]
                return -abs(used - target_z)
            else:
                if tv is None:
                    tv = self.stats.means[t]
                return -abs(raw - tv)
        raise ValueError(f"Unknown goal: {g}")

    def _anneal_weight(self, step: int) -> float:
        base = float(self.cfg.vgae_weight or 0.0)
        spec = self.cfg.vgae_anneal
        if not spec:
            return base
        mode, _, params = spec.partition(":")
        if mode == "cosine":
            # 这里可以接入 step 作为进度；若无外部步数，先给个平滑常数
            hi, lo = [float(x.strip()) for x in params.split(",")]
            t = 0.5 * (1 + math.cos(math.pi))
            return lo + (hi - lo) * t
        return base

    def _apply_unique_and_dupe(self, seq: str, smi_canon: str) -> Tuple[float, Dict[str, Any]]:
        """返回 (delta_reward, flags)；含 unique_bonus 与重复扣分。"""
        info: Dict[str, Any] = {}
        delta = 0.0

        # unique_bonus：对“规范 SMILES 首见”加分（保持旧口径）
        if (self.cfg.unique_bonus or 0.0) > 0.0 and smi_canon not in self._seen_unique:
            delta += float(self.cfg.unique_bonus)
            self._seen_unique.add(smi_canon)
            info["unique_bonus"] = True

        # duplicate penalty（新）
        if (self.cfg.dupe_penalty or 0.0) > 0.0:
            if bool(self.cfg.dupe_canonical):
                key = smi_canon
            else:
                key = self._noncanonical_key(seq) or smi_canon

            cnt = self._seen_counts.get(key, 0)
            if cnt >= 1:
                factor = (cnt + 1) if bool(self.cfg.dupe_escalate) else 1
                delta -= float(self.cfg.dupe_penalty) * float(factor)
                info["duplicate_penalized"] = True
                info["duplicate_count_prior"] = int(cnt)

            # 更新 LRU 计数
            new_cnt = cnt + 1
            if key in self._seen_counts:
                self._seen_counts.pop(key, None)
            self._seen_counts[key] = new_cnt

            # 裁剪 LRU：只保留最近 dupe_window 个“不同 SMILES”
            win = max(1, int(self.cfg.dupe_window or 5000))
            while len(self._seen_counts) > win:
                self._seen_counts.popitem(last=False)

        return delta, info

    def score_one(self, seq: str) -> Tuple[float, Dict[str, Any]]:
        self._call_count += 1

        smi = self._to_canonical_smiles(seq)
        if smi is None:
            return float(self.cfg.invalid_penalty), {"valid": False, "reason": "invalid_mol"}

        info: Dict[str, Any] = {"valid": True, "smiles": smi}
        total = 0.0

        # ---------- SELFIES token 长度 / 重原子数 惩罚（线性） ----------
        penalty_total = 0.0
        penalty_info: Dict[str, Any] = {}
        core_seq = seq.strip("<>").replace(" ", "")
        if (self.cfg.len_target is not None and self.cfg.len_target > 0 and self.cfg.len_lambda > 0.0):
            if self.gen_mode == "selfies":
                L_tok = core_seq.count("[")
            else:
                L_tok = len(core_seq)
            overflow = max(0, int(L_tok) - int(self.cfg.len_target))
            if overflow > 0:
                len_pen = - float(self.cfg.len_lambda) * float(overflow)
                penalty_total += len_pen
                penalty_info.update({"len_tokens": int(L_tok), "len_overflow": int(overflow), "len_pen": float(len_pen)})
            else:
                penalty_info.update({"len_tokens": int(L_tok), "len_overflow": 0, "len_pen": 0.0})

        m_for_pen = Chem.MolFromSmiles(smi)
        if m_for_pen is not None and (self.cfg.ha_target is not None and self.cfg.ha_target > 0 and self.cfg.ha_lambda > 0.0):
            heavy = int(m_for_pen.GetNumHeavyAtoms())
            overflow = max(0, heavy - int(self.cfg.ha_target))
            if overflow > 0:
                ha_pen = - float(self.cfg.ha_lambda) * float(overflow)
                penalty_total += ha_pen
                penalty_info.update({"heavy_atoms": heavy, "ha_overflow": int(overflow), "ha_pen": float(ha_pen)})
            else:
                penalty_info.update({"heavy_atoms": heavy, "ha_overflow": 0, "ha_pen": 0.0})

        # ---------- VGAE 合理性（软约束 / 门控） ----------
        vgae_bonus = 0.0
        apply_vgae = (self.cfg.vgae_weight or 0.0) != 0.0 and (self._call_count >= int(self.cfg.vgae_warmup or 0))
        if apply_vgae:
            s = self._vgae_score(smi)
            if s is not None:
                info["vgae_score"] = float(s)
                if (self.gate is not None) and (self._call_count >= int(self.cfg.vgae_warmup or 0)):
                    ok, detail = self.gate.check(s)
                    info.update(detail)
                    if not ok:
                        return float(self.cfg.invalid_penalty), {**info, **penalty_info, "ood": True}
                vgae_bonus = self._anneal_weight(self._call_count) * float(s)

        # ---------- 调用预测器并聚合多目标 ----------
        y = self.pred.predict([smi])[0]
        parts: Dict[str, Any] = {}
        for i, t in enumerate(self.cfg.targets):
            raw = float(y[i])
            used = (raw - self.stats.means[t]) / self.stats.stds[t] if self._use_z() else raw
            s_t = self._goal_score(t, used, raw, i)
            w_t = float(self.cfg.weights[i])
            total += w_t * s_t
            parts[t] = {"raw": raw, "used": used, "score": s_t, "weight": w_t}

        # ---------- 合并 VGAE / 唯一性 / 重复 / 其它惩罚 ----------
        total += vgae_bonus

        # unique + duplicate
        dupe_delta, dupe_info = self._apply_unique_and_dupe(seq, smi)
        total += dupe_delta
        info.update(dupe_info)

        # 长度&重原子惩罚
        total += penalty_total

        info["parts"] = parts
        info.update(penalty_info)
        return float(total), info
