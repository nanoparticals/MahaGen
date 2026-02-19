#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze generated SELFIES:
1) Decode to SMILES and compute SA score (Ertl's synthetic accessibility score).
2) Keep molecules with SA < 5.
3) Use your multi-objective chemprop predictor to compute rewards via
   reward_multiobjective.MultiObjectiveReward (also records VGAE soft-constraint score/bonus).
4) Export a CSV with SMILES, SA, per-target preds (and z-scores if enabled), VGAE score/bonus and total reward.
5) Draw molecule images batched as 20-per-page grids (5x4) in outputs/<outdir>/mols/page_xxxx.png; each tile has props text.
6) From the SA<5 set, select top-20 by total reward and draw radar charts (one PNG each)
   with the molecule image in the center and the targets as axes.

Dependencies:
  - rdkit
  - selfies
  - matplotlib
  - numpy, pandas
  - pillow (PIL)
  - Your repository modules on PYTHONPATH: reward_multiobjective.py and its transitive deps

Example:
python analyze_selfies_sa_reward.py \
  --input /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/samples_rnn_1e4.selfies \
  --predictor /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/chemprop_DPER0_cv5_new_4/fold_2/model_14/model.pt \
  --predictor-stats /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/et_stats_DPER0.json \
  --targets D P EG r0 --goals max max max max --weights 0.35 0.30 0.25 0.10 --zscore \
  --vgae-ckpt /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/vgae_struct_only_z16.pt --vgae-mean /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/z_stats_struct16/z_mean.npy \
  --outdir /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/samples_rnn_1e4/outputs
"""
from __future__ import annotations
import os, sys, json, math, argparse
from typing import Optional, Dict
import numbers

import numpy as np
import pandas as pd
import selfies as sf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import Crippen

# --- Import your reward class (in repo root) ---
from reward_multiobjective import MultiObjectiveReward, MultiObjConfig

# ---------------- SA score (approx of Ertl) -----------------
# 若你本地有官方 sascorer.py，可替换为：
#   import sascorer
#   sa = sascorer.calculateScore(mol)

def _fragment_score(mol: Chem.Mol) -> float:
    fr_benz = rdmd.CalcNumAromaticRings(mol)
    fr_aliph = rdmd.CalcNumAliphaticRings(mol)
    fr_bridge = rdmd.CalcNumBridgeheadAtoms(mol)
    fr_spiro = rdmd.CalcNumSpiroAtoms(mol)
    n_rings = fr_benz + fr_aliph
    return 0.3 * n_rings + 0.5 * fr_benz + 0.5 * fr_bridge + 0.4 * fr_spiro

def _complexity_penalty(mol: Chem.Mol) -> float:
    n_atoms = mol.GetNumHeavyAtoms()
    n_chiral = rdmd.CalcNumAtomStereoCenters(mol)
    n_bridge = rdmd.CalcNumBridgeheadAtoms(mol)
    n_spiro  = rdmd.CalcNumSpiroAtoms(mol)
    if hasattr(rdmd, "CalcNumFusedRings"):
        n_fused = rdmd.CalcNumFusedRings(mol)
    else:
        ri = mol.GetRingInfo()
        n_rings_local = ri.NumRings()
        if hasattr(ri, "NumRingSystems"):
            ring_systems = ri.NumRingSystems()
        else:
            try:
                ring_systems = len({frozenset(r) for r in ri.BondRings()})
            except Exception:
                ring_systems = 1 if n_rings_local > 0 else 0
        n_fused = max(0, n_rings_local - ring_systems)
    n_rings = rdmd.CalcNumRings(mol)
    return (0.007 * n_atoms ** 1.3
            + 0.3 * n_chiral
            + 0.5 * n_bridge
            + 0.4 * n_spiro
            + 0.25 * n_fused
            + 0.1 * max(0, n_rings - 3))

def _stereo_penalty(mol: Chem.Mol) -> float:
    return 0.2 * rdmd.CalcNumAtomStereoCenters(mol)

def synthetic_accessibility_score(mol: Chem.Mol) -> float:
    if mol is None:
        return 10.0
    n_atoms = mol.GetNumHeavyAtoms()
    size_term = 0.02 * max(0, n_atoms - 10)
    frag = _fragment_score(mol)
    comp = _complexity_penalty(mol)
    stereo = _stereo_penalty(mol)
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))
    hetero_term = -0.02 * min(hetero, 15)
    try:
        clogp = float(Crippen.MolLogP(mol))
    except Exception:
        clogp = 2.0
    lip_pen = 0.05 * max(0.0, clogp - 3.0)
    raw = 1.5 + size_term + comp + stereo + lip_pen - 0.2 * frag + hetero_term
    return float(np.clip(1.0 + (raw - 1.0), 1.0, 10.0))

# -------------- I/O helpers ----------------

def read_selfies(input_path: str, is_csv: bool = False, col: str | int = "selfies", keep_header: bool = False):
    if not is_csv:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s
    else:
        df = pd.read_csv(input_path, header=0 if keep_header else None)
        if isinstance(col, str) and col.isdigit():
            col = int(col)
        series = df.iloc[:, col] if isinstance(col, int) else df[col]
        for x in series.astype(str).tolist():
            s = x.strip()
            if s:
                yield s

def decode_to_canonical_smiles(selfies_seq: str) -> Optional[str]:
    try:
        smi = sf.decoder(selfies_seq)
    except Exception:
        smi = selfies_seq
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)

# 从可能的字典中提取数值（兼容 info['parts'][t] 返回结构）
def extract_numeric(x, prefer: str = 'pred') -> float:
    if isinstance(x, dict):
        # [Fix] 优先查找 'raw' (原始物理值) 和 'used' (归一化值)
        # 这样就不会错误地回退到 'score' (被取反的值) 了
        order = ['raw', 'used', prefer, 'pred', 'z', 'value', 'y_pred', 'score']
        
        for k in order:
            if k in x and isinstance(x[k], numbers.Real):
                return float(x[k])
        for v in x.values():
            if isinstance(v, numbers.Real):
                return float(v)
        return float('nan')
    try:
        return float(x)
    except Exception:
        return float('nan')

# -------------- Plot helpers ----------------

def draw_mol_with_props(smiles: str, props: Dict[str, float], out_png: str, dpi: int = 200):
    """(Legacy) Draw one molecule per PNG with properties below. Not used in 20-per-page mode."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return
    img = Draw.MolToImage(m, size=(400, 300))
    fig = plt.figure(figsize=(4, 4), dpi=dpi)
    ax = fig.add_axes([0, 0.25, 1, 0.75])
    ax.imshow(img)
    ax.axis('off')
    ax2 = fig.add_axes([0.05, 0.0, 0.9, 0.25])
    ax2.axis('off')
    lines = [f"{k} = {v:.3g}" for k, v in props.items()]
    txt = "  |  ".join(lines)
    ax2.text(0.5, 0.5, txt, ha='center', va='center', fontsize=9)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# --- New: card rendering (PIL) and 20-per-page grid stitch ---

def render_mol_card(smiles: str, props: Dict[str, float],
                    card_w: int = 360, mol_h: int = 240, text_h: int = 80) -> Image.Image:
    m = Chem.MolFromSmiles(smiles)
    card = Image.new('RGB', (card_w, mol_h + text_h), (255, 255, 255))
    if m is None:
        return card
    mol_img = Draw.MolToImage(m, size=(card_w, mol_h))  # PIL.Image
    card.paste(mol_img, (0, 0))
    draw = ImageDraw.Draw(card)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # build compact text, wrap to multiple lines
    kv = [f"{k}={v:.3g}" for k, v in props.items()]
    text = "  ".join(kv)
    max_chars = 60
    lines = []
    while len(text) > max_chars and len(lines) < 2:
        cut = text.rfind(' ', 0, max_chars)
        if cut <= 0:
            cut = max_chars
        lines.append(text[:cut])
        text = text[cut:].lstrip()
    lines.append(text)
    y0 = mol_h + 6
    for i, line in enumerate(lines[:3]):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((card_w - w) // 2, y0 + i * (h + 2)), line, fill=(0, 0, 0), font=font)
    return card

def save_cards_in_pages(cards: list[Image.Image], out_dir: str,
                        n_cols: int = 5, n_rows: int = 4, pad: int = 10, prefix: str = 'page') -> list[str]:
    """Stitch cards into pages of n_cols*n_rows; return list of saved page paths."""
    if not cards:
        return []
    per_page = n_cols * n_rows
    w, h = cards[0].size
    page_paths = []
    for page_idx, start in enumerate(range(0, len(cards), per_page), start=1):
        chunk = cards[start:start + per_page]
        rows = math.ceil(len(chunk) / n_cols)
        canvas_w = n_cols * w + (n_cols + 1) * pad
        canvas_h = rows * h + (rows + 1) * pad
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        for i, card in enumerate(chunk):
            r = i // n_cols
            c = i % n_cols
            x = pad + c * (w + pad)
            y = pad + r * (h + pad)
            canvas.paste(card, (x, y))
        out_path = os.path.join(out_dir, f"{prefix}_{page_idx:04d}.png")
        canvas.save(out_path)
        page_paths.append(out_path)
    return page_paths

# -------------- Main pipeline ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='SELFIES list (.txt each line) or CSV')
    ap.add_argument('--csv', action='store_true', help='Treat input as CSV')
    ap.add_argument('--col', default='selfies', help='CSV column name or index')
    ap.add_argument('--keep-header', action='store_true')

    # Reward / predictor
    ap.add_argument('--predictor', required=True, help='chemprop ckpt/dir or joblib model path')
    ap.add_argument('--predictor-stats', required=True, help='stats json (means/stds) for targets')
    ap.add_argument('--targets', nargs='+', required=True)
    ap.add_argument('--goals', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, required=True)
    ap.add_argument('--target-values', nargs='*', default=None)
    ap.add_argument('--zscore', action='store_true')
    ap.add_argument('--invalid-penalty', type=float, default=-2.0)

    # VGAE soft constraint (optional)
    ap.add_argument('--vgae-ckpt', default='ckpts/vgae_li2022_DPEr0_z16.pt')
    ap.add_argument('--vgae-mean', default='ckpts/z_stats_16/z_mean.npy ')
    ap.add_argument('--vgae-cov', default='ckpts/z_stats_16/z_cov_fixed.npy')
    ap.add_argument('--vgae-zbank', default='ckpts/z_stats_16/z_bank.npy')
    ap.add_argument('--vgae-weight', type=float, default=0.0)
    ap.add_argument('--vgae-gate', default='off')
    ap.add_argument('--vgae-warmup', type=int, default=0)
    ap.add_argument('--vgae-anneal', default=None)

    # Output
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--sa-thr', type=float, default=5.0)
    ap.add_argument('--n-top-radar', type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img_dir = os.path.join(args.outdir, 'mols'); os.makedirs(img_dir, exist_ok=True)
    radar_dir = os.path.join(args.outdir, 'radars'); os.makedirs(radar_dir, exist_ok=True)

    selfies_list = list(read_selfies(args.input, is_csv=args.csv, col=args.col, keep_header=args.keep_header))
    print(f"[info] loaded SELFIES rows: {len(selfies_list)}")

    cfg = MultiObjConfig(
        targets=args.targets,
        goals=[g.lower() for g in args.goals],
        weights=[float(w) for w in args.weights],
        target_values=[(float(x) if x not in (None, 'None') else None) for x in (args.target_values or [])]
                      + [None] * (len(args.targets) - len(args.target_values or [])),
        use_zscore=bool(args.zscore),
        invalid_penalty=float(args.invalid_penalty),
        predictor_path=args.predictor,
        predictor_stats_path=args.predictor_stats,
        vgae_ckpt=(args.vgae_ckpt or None),
        vgae_mean_path=(args.vgae_mean or None),
        vgae_cov_path=(args.vgae_cov or None),
        vgae_zbank_path=(args.vgae_zbank or None),
        vgae_weight=float(args.vgae_weight or 0.0),
        vgae_gate=(args.vgae_gate or 'off'),
        vgae_warmup=int(args.vgae_warmup or 0),
        vgae_anneal=(args.vgae_anneal or None),
    )
    R = MultiObjectiveReward(model_path=args.predictor, stats_path=args.predictor_stats,
                             cfg=cfg, gen_mode='selfies', featurizer='auto')

    rows = []
    kept = 0
    cards: list[Image.Image] = []

    for idx, s in enumerate(selfies_list):
        core = s.strip().replace(' ', '')
        seq = f"<{core}>"
        total, info = R.score_one(seq)

        smi = info.get('smiles') if info.get('valid', False) else None
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        sa = synthetic_accessibility_score(mol)
        vgae_raw = info.get('vgae_score', np.nan)
        vgae_score = float(vgae_raw) if isinstance(vgae_raw, numbers.Real) else np.nan

        parts = info.get('parts', {}) if isinstance(info.get('parts'), dict) else {}
        per_target = {}
        for t in args.targets:
            v = parts.get(t, np.nan)
            per_target[t] = extract_numeric(v, prefer='z' if cfg.use_zscore else 'pred')

        show_props = {**{t: per_target[t] for t in args.targets}, 'SA': float(sa), 'Reward': float(total)}
        # Create card (PIL image) and collect for paging (20 per image)
        card = render_mol_card(smi, show_props, card_w=360, mol_h=240, text_h=80)
        cards.append(card)

        rows.append({
            'index': idx,
            'selfies': core,
            'smiles': smi,
            'SA': float(sa),
            **{f"pred_{t}": float(per_target[t]) for t in args.targets},
            'vgae_score': vgae_score,
            'reward': float(total),
        })
        kept += 1
        if kept % 200 == 0:
            print(f"[prog] processed {kept} valid molecules ...")

    if not rows:
        print('[ERR] No valid molecules decoded from SELFIES.')
        return

    # Save 20-per-page images under mols/
    page_paths = save_cards_in_pages(cards, img_dir, n_cols=5, n_rows=4, pad=10, prefix='page')
    per_page = 20

    df = pd.DataFrame(rows)
    df['img_page'] = [os.path.join('mols', f"page_{(i//per_page)+1:04d}.png") for i in range(len(df))]
    df['img_tile'] = [i % per_page for i in range(len(df))]

    df_all_csv = os.path.join(args.outdir, 'all_molecules.csv')
    df.to_csv(df_all_csv, index=False)
    print('[OK] Wrote:', df_all_csv)
    print('[OK] Saved', len(page_paths), 'pages of molecule grids to', img_dir)

    # Filter SA < threshold
    df_ok = df[df['SA'] < float(args.sa_thr)].copy()
    if df_ok.empty:
        print(f"[WARN] No molecules with SA < {args.sa_thr}.")
        return
    df_ok = df_ok.sort_values('reward', ascending=False).reset_index(drop=True)
    df_ok_csv = os.path.join(args.outdir, f'sa_lt_{int(args.sa_thr)}.csv')
    df_ok.to_csv(df_ok_csv, index=False)
    print('[OK] Wrote:', df_ok_csv)

    # Top-N radar charts (unchanged)
    topN = int(min(args.n_top_radar, len(df_ok)))
    tops = df_ok.head(topN)

    if cfg.use_zscore:
        vmin, vmax = None, None
    else:
        vals = {t: tops[f'pred_{t}'].astype(float).values.tolist() for t in args.targets}
        vmin = min(min(v) for v in vals.values() if len(v) > 0)
        vmax = max(max(v) for v in vals.values() if len(v) > 0)

    for i, row in tops.iterrows():
        smi = row['smiles']
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mol_img = Draw.MolToImage(m, size=(300, 225))
        props = {t: float(row.get(f'pred_{t}', np.nan)) for t in args.targets}
        title = f"Top{i+1}: Reward={row['reward']:.3f}, SA={row['SA']:.2f}"
        out_png = os.path.join(radar_dir, f"radar_top_{i+1:02d}.png")
        # simple radar chart with molecule centered
        labels = list(props.keys())
        data = [float(props[k]) for k in labels]
        N = len(labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        data += data[:1]
        angles += angles[:1]
        fig = plt.figure(figsize=(4, 4))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, data, linewidth=1)
        ax.fill(angles, data, alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        if vmin is not None and vmax is not None and vmax > vmin:
            ax.set_ylim(vmin, vmax)
        else:
            lo, hi = min(data), max(data)
            pad = 0.05 * (hi - lo + 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10, pad=12)
        try:
            imagebox = OffsetImage(mol_img, zoom=0.35)
            from matplotlib.offsetbox import AnnotationBbox
            ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
            ax.add_artist(ab)
        except Exception:
            pass
        fig.savefig(out_png, bbox_inches='tight', dpi=200)
        plt.close(fig)

    print(f"[OK] Saved {topN} radar charts to {radar_dir}")

if __name__ == '__main__':
    main()
