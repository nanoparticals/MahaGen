#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised training for a sequence generator (SMILES/SELFIES as plain text),
with optional conditional pretraining using a row-aligned z bank (e.g., VGAE z).

What's added (minimally invasive):
- Auto-compute μ and Σ^{-1} from --z-bank for density awareness.
- Density-aware cond_scale: weaken z injection when far from μ (no extra loss/backward).
- Lightweight discriminator D(z) for monitoring + self-training (no generator adversarial loss).
- Progress bar shows d2 (Mahalanobis^2) and adv (-log D(z)).

Command line stays the same as your original script.
"""
import os
import csv
import math
import argparse
import time
import contextlib
import warnings
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from data import GeneratorData
from stackRNN import StackAugmentedRNN


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, path: str):
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def collect_optimizers(model):
    """Collect optimizers possibly attached on model/model.core."""
    opts = []
    if hasattr(model, "optimizer") and model.optimizer is not None:
        opts.append(model.optimizer)
    if hasattr(model, "core") and hasattr(model.core, "optimizer") and model.core.optimizer is not None:
        opts.append(model.core.optimizer)
    # deduplicate
    uniq, seen = [], set()
    for o in opts:
        if id(o) not in seen:
            seen.add(id(o)); uniq.append(o)
    return uniq


@contextlib.contextmanager
def patched_optimizer_for_accum(opt_list, lr_scale: float, clip_grad: float, params):
    """
    During the context, zero_grad/step of each optimizer are patched to no-op so that
    multiple inner steps can accumulate gradients. On exit:
      1) optionally clip gradients
      2) temporarily scale LR by lr_scale
      3) call real step() once
      4) restore LR and call zero_grad()
    """
    backups = []
    for opt in opt_list:
        backups.append({
            "opt": opt,
            "orig_zero": opt.zero_grad,
            "orig_step": opt.step,
            "lrs": [pg.get('lr', None) for pg in opt.param_groups],
        })
        opt.zero_grad = lambda *a, **k: None
        opt.step = lambda *a, **k: None
    try:
        yield
    finally:
        if clip_grad and clip_grad > 0:
            clip_grad_norm_(params, max_norm=clip_grad)
        for b in backups:
            opt = b["opt"]
            opt.zero_grad = b["orig_zero"]
            opt.step = b["orig_step"]
            for pg, base_lr in zip(opt.param_groups, b["lrs"]):
                if base_lr is not None:
                    pg["lr"] = float(base_lr) * lr_scale
            opt.step()
            for pg, base_lr in zip(opt.param_groups, b["lrs"]):
                if base_lr is not None:
                    pg["lr"] = base_lr
            opt.zero_grad()


def parse_milestones(v):
    if v is None:
        return []
    return [int(x) for x in v]


# ------------------------ z helpers ------------------------
class ZBank:
    def __init__(self, path: str | None):
        self.path = path
        self.arr: np.ndarray | None = None
        self.n = 0
        self.d = 0
        if path and os.path.exists(path):
            self.arr = np.load(path)
            if self.arr.ndim != 2:
                raise ValueError(f"z_bank at {path} must be 2D, got shape {self.arr.shape}")
            self.n, self.d = int(self.arr.shape[0]), int(self.arr.shape[1])
            print(f"[z] loaded z_bank: shape={self.arr.shape} from {path}")
        else:
            if path:
                warnings.warn(f"[z] z_bank not found at {path}; z conditioning disabled.")

    def get_by_index(self, idx: int) -> np.ndarray:
        if self.arr is None or self.n == 0:
            raise RuntimeError("z_bank not loaded")
        if idx < 0: idx = 0
        if idx >= self.n: idx = self.n - 1
        return self.arr[idx]

    def get_random(self) -> np.ndarray:
        if self.arr is None or self.n == 0:
            raise RuntimeError("z_bank not loaded")
        ridx = np.random.randint(0, self.n)
        return self.arr[ridx]

    def get_hard_negative(self, idx: int) -> np.ndarray:
        """Return a different row than idx (for hard-negative injection)."""
        if self.arr is None or self.n <= 1:
            return self.get_random()
        ridx = idx
        trials = 0
        while ridx == idx and trials < 10:
            ridx = np.random.randint(0, self.n)
            trials += 1
        if ridx == idx:
            ridx = (idx + 1) % self.n
        return self.arr[ridx]


def set_model_condition_if_supported(G, z_tensor: torch.Tensor, scale: float) -> bool:
    if hasattr(G, "set_condition"):
        try:
            G.set_condition(z_tensor, scale=scale)
            return True
        except TypeError:
            # legacy signature set_condition(z) without scale
            G.set_condition(z_tensor)
            return True
        except Exception as e:
            warnings.warn(f"[z] set_condition failed: {e}")
            return False
    return False


def maybe_freeze_backbone(G, freeze: bool):
    if not freeze:
        return 0
    frozen, total = 0, 0
    for n, p in G.named_parameters():
        total += 1
        key = n.lower()
        # keep condition branches trainable if they exist
        if ("cond2h" in key) or ("cond2in" in key) or ("condition" in key and ("linear" in key or "proj" in key)):
            p.requires_grad = True
        else:
            p.requires_grad = False
            frozen += 1
    print(f"[freeze] backbone frozen params: {frozen}/{total}")
    return frozen


# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data", type=str, required=True, help="CSV path containing sequences (SMILES/SELFIES)")
    ap.add_argument("--smiles-col", type=int, default=0, help="Zero-based column index for the sequence column")
    ap.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    ap.add_argument("--keep-header", action="store_true", help="Set if the CSV contains a header row")
    ap.add_argument("--max-len", type=int, default=120, help="Max sequence length (<= data tokenizer)")
    ap.add_argument("--mode", type=str, default="smiles", choices=["smiles", "selfies"],
                    help="Input type (only affects tokenization in data.GeneratorData)")

    # model & train
    ap.add_argument("--iters", type=int, default=2000, help="Training *sample* steps (one sample per inner step)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cell", type=str, default="GRU", choices=["GRU", "LSTM"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", choices=["adam", "adamw", "adadelta", "sgd"], default="adam",
                    help="optimizer")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch size)")
    ap.add_argument("--clip-grad", type=float, default=0.0, help="max grad-norm clipping; 0=off")

    # LR decay by milestones (in *sample steps*)
    ap.add_argument("--lr-milestones", nargs="+", default=None,
                    help="sample steps to decay LR (e.g., 400000 900000 1200000)")
    ap.add_argument("--lr-gamma", type=float, default=0.5, help="LR decay factor applied at each milestone")

    # logging / ckpt / sampling
    ap.add_argument("--outdir", type=str, default="ckpts/gen_sup", help="Directory to save logs & checkpoints")
    ap.add_argument("--fresh-log", action="store_true", help="If set, truncate/create a fresh train_gen_log.csv")
    ap.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N *sample* steps")
    ap.add_argument("--sample-every", type=int, default=200, help="Print samples every N *sample* steps")
    ap.add_argument("--sample-len", type=int, default=100, help="Sampling predict length")
    ap.add_argument("--n-sample-each", type=int, default=3, help="How many samples to print each time")
    ap.add_argument("--save-samples", type=str, default="", help="If set, append printed samples to this file")
    ap.add_argument("--resume", type=str, default="", help="Resume from a checkpoint .pt file")
    ap.add_argument("--resume-strict", action="store_true", help="Strictly load resume state_dict (default: False)")
    ap.add_argument("--sample-only", type=int, default=0, help="If >0, only sample N sequences and exit")

    # ---------- NEW: conditional ----------
    ap.add_argument("--cond-dim", type=int, default=16, help="latent z dim; >0 to enable conditional generator")
    ap.add_argument("--z-bank", type=str, default=None, help="Path to z_bank.npy aligned with CSV rows")
    ap.add_argument("--cond-scale", type=float, default=1.5, help="Strength of z injection at runtime")
    ap.add_argument("--freeze-backbone", action="store_true", help="Train only cond branches if present")
    ap.add_argument("--z-hardneg-prob", type=float, default=0.0, help="Prob. to replace aligned z with a different row")
    ap.add_argument("--z-mode", type=str, default="align", choices=["align", "random"],
                    help="How to choose z per sample: align by index or random")

    args = ap.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_dir(args.outdir)
    log_csv = os.path.join(args.outdir, "train_gen_log.csv")

    # Data
    data = GeneratorData(
        args.data,
        max_len=args.max_len,
        cols_to_read=[args.smiles_col],
        delimiter=args.delimiter,
        keep_header=args.keep_header,
        mode=args.mode,
    )
    # quick data info
    try:
        seqs = list(getattr(data, "sequences", []))
        if args.mode == "selfies":
            lens = [s.count("[") for s in seqs]
        else:
            lens = [len(s) for s in seqs]
        avg_len = (sum(lens) / len(lens)) if lens else 0.0
        print(f"[data] mode={args.mode}  samples={len(seqs)}  vocab={data.n_characters}  avg_len={avg_len:.1f}")
    except Exception:
        pass

    # ---- z bank ----
    zbank = ZBank(args.z_bank)
    if zbank.arr is not None:
        if args.cond_dim <= 0:
            raise ValueError("--cond-dim must be > 0 when --z-bank is provided.")
        if zbank.d != int(args.cond_dim):
            raise ValueError(f"z_bank dim {zbank.d} != --cond-dim {args.cond_dim}. "
                             "Please set --cond-dim to z_bank.shape[1].")

    # Model
    G = StackAugmentedRNN(
        input_size=data.n_characters,
        hidden_size=args.hidden,
        output_size=data.n_characters,
        layer_type=args.cell,
        n_layers=args.layers,
        has_stack=False,
        lr=args.lr,
        # >>> enable conditional branches
        cond_dim=int(args.cond_dim) if int(args.cond_dim) > 0 else 0,
    )
    print(f"[debug] cond_dim = {getattr(G, 'cond_dim', args.cond_dim)}  cond_scale(run-time) = {args.cond_scale}")

    # Optimizer
    opt_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adadelta": torch.optim.Adadelta,
        "sgd": torch.optim.SGD,
    }
    opt_cls = opt_map[args.optim]
    weight_decay = 1e-5 if args.optim in ("adam", "adamw") else 0.0
    extra = {}
    if args.optim == "sgd":
        extra.update(dict(momentum=0.9, nesterov=True))

    G.optimizer_instance = opt_cls
    G.optimizer = opt_cls(G.parameters(), lr=args.lr, weight_decay=weight_decay, **extra)
    print(f"[opt] {opt_cls.__name__} lr={args.lr} wd={weight_decay}")

    # CUDA
    use_cuda = args.cuda and torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    if use_cuda:
        G.cuda()

    # ---- NEW: VGAE stats (from z_bank) + Discriminator (monitor only) ----
    vgae_mu = None
    vgae_Sinv = None
    VGAE_REG_W = 0.30  # monitor only (no backward to generator)
    ADV_W = 0.20       # monitor only (no backward to generator)

    if zbank.arr is not None and int(args.cond_dim) > 0:
        Z = zbank.arr.astype(np.float64)                   # (N,D)
        mu = Z.mean(axis=0)
        C = np.cov(Z, rowvar=False)                        # (D,D)
        # numerical ridge
        eps = 1e-3 * (np.trace(C) / C.shape[0] + 1e-12)
        C = C + eps * np.eye(C.shape[0], dtype=C.dtype)
        Si = np.linalg.pinv(C)

        vgae_mu = torch.tensor(mu, dtype=torch.float32, device=device_str).unsqueeze(0)  # (1,D)
        vgae_Sinv = torch.tensor(Si, dtype=torch.float32, device=device_str)             # (D,D)
        print(f"[vgae-reg] μ.shape={mu.shape}, Σ^-1 ready (ridge={eps:.2e})")

        class ZDiscriminator(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 1),
                    torch.nn.Sigmoid(),
                )
            def forward(self, z):  # (B,D)
                return self.net(z)

        D_adv = ZDiscriminator(zbank.d).to(device_str)
        opt_adv = torch.optim.Adam(D_adv.parameters(), lr=1e-3, weight_decay=1e-5)
        print(f"[adv] enabled: λ_adv={ADV_W} | [vgae-reg] λ={VGAE_REG_W}")
    else:
        D_adv = None; opt_adv = None

    # optional: freeze backbone
    maybe_freeze_backbone(G, args.freeze_backbone)

    # ---- Resume ----
    if args.resume:
        try:
            state = torch.load(args.resume, map_location=device_str)
            def _shape(k): return tuple(state[k].shape) if (isinstance(state, dict) and k in state) else None
            dec_w = _shape("decoder.weight")
            rnn_ih0 = _shape("rnn.weight_ih_l0")
            vocab_ckpt = dec_w[0] if dec_w else None
            gate = 3 if args.cell.upper() == "GRU" else 4
            hidden_ckpt = (rnn_ih0[0] // gate) if rnn_ih0 else None
            problems = []
            if vocab_ckpt is not None and vocab_ckpt != data.n_characters:
                problems.append(f"vocab(size) mismatch: ckpt={vocab_ckpt} vs data={data.n_characters}")
            if hidden_ckpt is not None and hidden_ckpt != args.hidden:
                problems.append(f"hidden mismatch: ckpt={hidden_ckpt} vs arg.hidden={args.hidden}")
            if problems:
                print("[resume-check] " + " | ".join(problems))
            G.load_state_dict(state, strict=args.resume_strict)
            print(f"[info] Resumed from {args.resume}")
        except RuntimeError as e:
            print("[error] load_state_dict failed:\n", e)
            print("提示：通常是 hidden 或 词表(size) 不一致。请确认："
                  "\n  - --hidden / --layers 与 ckpt 一致；"
                  "\n  - 数据文件（决定 tokenizer 词表）与 ckpt 训练时一致。")
            return

    # ---- sampling only ----
    sample_sink = None
    if args.save_samples:
        ensure_dir(os.path.dirname(args.save_samples) or ".")
        sample_sink = open(args.save_samples, "a", encoding="utf-8")

    if args.sample_only > 0:
        G.eval()
        print("[info] Sampling only (no training)...")
        for i in range(args.sample_only):
            # set a random z for sampling if available
            if zbank.arr is not None and args.cond_scale > 0:
                z = torch.from_numpy(zbank.get_random()).unsqueeze(0).to(device_str).float()
                set_model_condition_if_supported(G, z, args.cond_scale)
            seq = G.evaluate(data=data, prime_str="<", predict_len=args.sample_len)
            line = f"SAMPLE[{i+1}]: {seq}"
            print(line)
            if sample_sink:
                sample_sink.write(seq.strip("<>").replace(" ", "") + "\n")
        if sample_sink:
            sample_sink.close()
        return

    # ---- train mode ----
    G.train()

    # CSV log
    if args.fresh_log or (not os.path.exists(log_csv)):
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sample_step", "avg_loss", "time", "lr", "cond"])

    OPTS = collect_optimizers(G)
    total_samples = int(args.iters)
    accum = max(1, int(args.accum))
    outer_steps = math.ceil(total_samples / accum)
    seen_samples = 0
    ema = None
    milestones = sorted([m for m in parse_milestones(args.lr_milestones) if m and m > 0])

    print("[info] Start supervised training…")
    pbar = trange(outer_steps, desc="Supervised", dynamic_ncols=True)

    zs_fake_buf = []
    used_condition_once = False

    for outer in pbar:
        # LR decay by milestones (based on *sample steps*)
        next_seen_after = min(total_samples, seen_samples + accum)
        for ms in [m for m in milestones if seen_samples < m <= next_seen_after]:
            for opt in OPTS:
                for pg in opt.param_groups:
                    pg["lr"] *= args.lr_gamma
            print(f"[lr-decay] step={ms}  gamma={args.lr_gamma}")

        remaining = total_samples - seen_samples
        inner_runs = min(accum, remaining)
        if inner_runs <= 0:
            break

        avg_loss = 0.0
        last_maha_val = 0.0
        last_adv_val = 0.0

        for opt in OPTS:
            opt.zero_grad()

        with patched_optimizer_for_accum(
            OPTS, lr_scale=1.0 / float(inner_runs), clip_grad=args.clip_grad, params=G.parameters()
        ):
            for _ in range(inner_runs):
                # ----- fetch one training sample -----
                got_index = False
                idx_val = None
                try:
                    # try API with index
                    inp, tgt, idx = data.random_training_set(return_index=True)
                    got_index = True
                    if isinstance(idx, torch.Tensor):
                        idx_val = int(idx.item())
                    else:
                        idx_val = int(idx)
                except TypeError:
                    # fallback to legacy signature
                    inp, tgt = data.random_training_set(smiles_augmentation=None)
                except Exception:
                    inp, tgt = data.random_training_set(smiles_augmentation=None)

                if use_cuda:
                    inp = inp.cuda(non_blocking=True)
                    tgt = tgt.cuda(non_blocking=True)

                # ----- set z condition if available -----
                cond_used = False
                if zbank.arr is not None and args.cond_scale > 0 and int(args.cond_dim) > 0:
                    if args.z_mode == "align" and got_index and (idx_val is not None):
                        z_hardneg_prob = float(getattr(args, "z_hardneg_prob", 0.0))
                        use_hardneg = (np.random.rand() < z_hardneg_prob)
                        if use_hardneg:
                            z_np = zbank.get_hard_negative(idx_val)
                        else:
                            z_np = zbank.get_by_index(idx_val)
                    else:
                        if args.z_mode == "align" and not got_index and not used_condition_once:
                            warnings.warn("[z] Could not obtain sample index from GeneratorData; "
                                          "falling back to --z-mode random. "
                                          "To enable alignment, make random_training_set return (inp,tgt,idx).")
                        z_np = zbank.get_random()

                    z = torch.from_numpy(z_np).unsqueeze(0).to(device_str).float()
                    cond_used = set_model_condition_if_supported(G, z, args.cond_scale)
                    used_condition_once = used_condition_once or cond_used

                    # ------- density-aware cond_scale (no backward) -------
                    if cond_used and (vgae_mu is not None) and (vgae_Sinv is not None):
                        with torch.no_grad():
                            delta = z - vgae_mu
                            d2 = float((delta @ vgae_Sinv @ delta.transpose(1, 0)).squeeze().item())
                            D = max(1, int(zbank.d))
                            scale_eff = float(args.cond_scale) * float(np.exp(-0.5 * d2 / D))
                            scale_eff = float(np.clip(scale_eff, 0.2, float(args.cond_scale)))
                        set_model_condition_if_supported(G, z, scale=scale_eff)
                        last_maha_val = d2

                    # ------- discriminator monitor (no generator grad) -------
                    if cond_used and (D_adv is not None):
                        with torch.no_grad():
                            p_fake = float(D_adv(z).mean().item())
                            last_adv_val = -math.log(max(1e-8, p_fake))
                        zs_fake_buf.append(z.detach())

                # ----- one supervised step (your original loss/backward inside) -----
                loss = G.train_step(inp, tgt)
                avg_loss += float(loss)
                seen_samples += 1

        # ---- discriminator update (batched) ----
        if D_adv is not None and zs_fake_buf:
            z_fake = torch.cat(zs_fake_buf, dim=0)  # (B,D)
            with torch.no_grad():
                idx_real = np.random.randint(0, zbank.n, size=z_fake.size(0))
                z_real = torch.from_numpy(zbank.arr[idx_real]).to(device_str).float()
            pred_real = D_adv(z_real)
            pred_fake = D_adv(z_fake)
            loss_d = -torch.log(pred_real + 1e-8).mean() - torch.log(1 - pred_fake + 1e-8).mean()
            opt_adv.zero_grad()
            loss_d.backward()
            opt_adv.step()
            zs_fake_buf.clear()

        avg_loss /= float(inner_runs)
        ema = (0.9 * ema + 0.1 * avg_loss) if (ema is not None) else avg_loss
        show_lr = OPTS[0].param_groups[0]["lr"] if OPTS else args.lr
        pbar.set_postfix(
            loss=f"{ema:.4f}", lr=show_lr, cond=("Y" if used_condition_once else "N"),
            d2=f"{last_maha_val:.2f}", adv=f"{last_adv_val:.2f}"
        )

        # sampling preview
        if seen_samples % args.sample_every == 0:
            G.eval()
            with torch.no_grad():
                for k in range(args.n_sample_each):
                    if zbank.arr is not None and args.cond_scale > 0 and int(args.cond_dim) > 0:
                        z = torch.from_numpy(zbank.get_random()).unsqueeze(0).to(device_str).float()
                        set_model_condition_if_supported(G, z, args.cond_scale)
                    seq = G.evaluate(data=data, prime_str="<", predict_len=args.sample_len)
                    print(f"[{seen_samples}] sample[{k+1}] = {seq}")
                    if sample_sink:
                        sample_sink.write(seq.strip("<>").replace(" ", "") + "\n")
                        sample_sink.flush()
            G.train()

        # save checkpoint
        if seen_samples % args.save_every == 0:
            ckpt_path = os.path.join(args.outdir, f"step_{seen_samples}.pt")
            save_checkpoint(G, ckpt_path)
            print(f"[info] Saved checkpoint: {ckpt_path}")

        # append log
        if (outer + 1) % 10 == 0 or seen_samples == total_samples:
            with open(log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([seen_samples, f"{avg_loss:.6f}", f"{time.time():.3f}", f"{show_lr:.6g}",
                            ("Y" if used_condition_once else "N")])

    # done
    final_ckpt = os.path.join(args.outdir, f"final_step_{seen_samples}.pt")
    save_checkpoint(G, final_ckpt)
    print(f"[info] Final checkpoint: {final_ckpt}")

    G.eval()
    print("[info] Final sampling (5 molecules):")
    with torch.no_grad():
        for k in range(5):
            if zbank.arr is not None and args.cond_scale > 0 and int(args.cond_dim) > 0:
                z = torch.from_numpy(zbank.get_random()).unsqueeze(0).to(device_str).float()
                set_model_condition_if_supported(G, z, args.cond_scale)
            seq = G.evaluate(data=data, prime_str="<", predict_len=args.sample_len)
            print(f"FINAL_SAMPLE[{k+1}] = {seq}")
            if sample_sink:
                sample_sink.write(seq.strip("<>").replace(" ", "") + "\n")
    if sample_sink:
        sample_sink.close()


if __name__ == "__main__":
    main()
