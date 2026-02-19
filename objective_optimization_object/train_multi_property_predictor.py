#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_multi_property_predictor.py
---------------------------------
- Multi-objective property prediction (outputs multiple targets at once) for scoring generated molecules.
- Compatible with your single-task style: RDKit Morgan fingerprints + ExtraTreesRegressor.
- Performs K-fold CV per-target (MAE/R2), saves summary, and trains on full data to save final model.
Required columns: SMILES + multiple target columns (or use --smiles-col to pick another name).
"""

import argparse, json, numpy as np, pandas as pd, joblib
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------- featurization -----------------
def morgan_fp(smiles: str, radius=2, nBits=2048):
    arr = np.zeros((nBits,), dtype=np.int8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def featurize(smiles_list: List[str], radius=2, nBits=2048) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    ok = []
    for s in smiles_list:
        fp = morgan_fp(s, radius, nBits)
        if fp is None:
            ok.append(False)
        else:
            ok.append(True); X.append(fp)
    X = np.stack(X, axis=0) if len(X)>0 else np.zeros((0, nBits), dtype=np.float32)
    return X, np.array(ok, dtype=bool)

# ----------------- training -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV path')
    ap.add_argument('--smiles-col', default='SMILES', help='SMILES column name (default: SMILES)')
    ap.add_argument('--label-cols', required=True, help='Comma-separated targets, e.g., D,P,EG,r0')
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--nbits', type=int, default=2048)
    ap.add_argument('--trees', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cv-k', type=int, default=5)
    ap.add_argument('--model', required=True, help='Output model .joblib path')
    ap.add_argument('--stats', required=True, help='Output summary .json path')
    ap.add_argument('--cv-report', default='cv_metrics_multi.json', help='CV detailed report .json')
    ap.add_argument('--target-transform', default='log1p', choices=['none', 'log1p'],
                    help='Label transform applied before fitting (default: log1p for all targets).')
    return ap.parse_args()

def main():
    a = parse_args()
    df = pd.read_csv(a.data)
    if a.smiles_col not in df.columns:
        raise SystemExit(f'Missing SMILES column: {a.smiles_col}. Available: {list(df.columns)}')
    target_names = [t.strip() for t in a.label_cols.split(',') if t.strip()!='']
    for t in target_names:
        if t not in df.columns:
            raise SystemExit(f'Missing target column: {t}. Available: {list(df.columns)}')

    smiles = df[a.smiles_col].astype(str).tolist()
    Y_df = df[target_names].apply(pd.to_numeric, errors='coerce')
    # filter: valid SMILES and no NaN in any selected target
    X_all, ok = featurize(smiles, a.radius, a.nbits)
    mask = ok & (~Y_df.isna().any(axis=1).values)
    X = X_all[mask]
    Y_raw = Y_df[mask].values.astype(np.float32)
    kept = int(mask.sum()); dropped = int(len(mask) - kept)
    if kept == 0:
        raise SystemExit('No valid data (check SMILES parsing or missing labels).')
    transform_name = (a.target_transform or 'none').strip().lower()
    transform_desc: Optional[Dict[str, object]] = None
    if transform_name == 'none':
        Y_model = Y_raw.copy()
    elif transform_name == 'log1p':
        if float(Y_raw.min()) <= -1.0:
            raise SystemExit('log1p transform requires all targets > -1.')
        Y_model = np.log1p(Y_raw)
        transform_desc = {
            'name': 'log1p',
            'inverse': 'expm1',
            'targets': target_names,
            'note': 'Applied log1p to all targets before fitting; inverse expm1 required for scoring.'
        }
    else:
        raise SystemExit(f'Unsupported target transform: {transform_name}')

    def inverse_transform(arr: np.ndarray) -> np.ndarray:
        if transform_name == 'log1p':
            return np.expm1(arr)
        return arr

    # cross-validated metrics
    kf = KFold(n_splits=a.cv_k, shuffle=True, random_state=a.seed)
    maes = []; r2s = []
    fold_reports = []
    for fi, (tr, va) in enumerate(kf.split(X), 1):
        model = ExtraTreesRegressor(
            n_estimators=a.trees, random_state=a.seed+fi, n_jobs=-1
        )
        model.fit(X[tr], Y_model[tr])
        P_model = np.asarray(model.predict(X[va]), dtype=np.float32)
        P_raw = inverse_transform(P_model)
        fold_mae = [float(mean_absolute_error(Y_raw[va][:,j], P_raw[:,j])) for j in range(Y_raw.shape[1])]
        fold_r2  = [float(r2_score(Y_raw[va][:,j], P_raw[:,j])) for j in range(Y_raw.shape[1])]
        maes.append(fold_mae); r2s.append(fold_r2)
        fold_reports.append({'fold': fi, 'mae': dict(zip(target_names, fold_mae)),
                             'r2': dict(zip(target_names, fold_r2))})

    mae_mean = np.array(maes).mean(axis=0).tolist()
    r2_mean  = np.array(r2s).mean(axis=0).tolist()
    summary = {
        'n_total': int(len(df)), 'n_kept': kept, 'n_dropped': dropped,
        'targets': target_names, 'cv_k': a.cv_k, 'trees': a.trees,
        'mae_mean': dict(zip(target_names, [float(x) for x in mae_mean])),
        'r2_mean':  dict(zip(target_names, [float(x) for x in r2_mean])),
        'mae_mean_avg': float(np.mean(mae_mean)),
        'r2_mean_avg' : float(np.mean(r2_mean)),
    }
    raw_means = dict(zip(target_names, [float(np.mean(Y_raw[:, i])) for i in range(Y_raw.shape[1])]))
    raw_stds  = dict(zip(target_names, [float(np.std(Y_raw[:, i])) for i in range(Y_raw.shape[1])]))
    summary['means'] = raw_means
    summary['stds'] = raw_stds

    if transform_desc is not None:
        summary['target_transform'] = transform_desc
        summary['model_space'] = {
            'transform': transform_desc['name'],
            'inverse': transform_desc['inverse'],
            'targets': transform_desc['targets']
        }
        summary['model_space_means'] = dict(zip(target_names, [float(np.mean(Y_model[:, i])) for i in range(Y_model.shape[1])]))
        summary['model_space_stds'] = dict(zip(target_names, [float(np.std(Y_model[:, i])) for i in range(Y_model.shape[1])]))
    # fit full data
    final_model = ExtraTreesRegressor(n_estimators=a.trees, random_state=a.seed, n_jobs=-1)
    final_model.fit(X, Y_model)
    meta = {
        'smiles_col': a.smiles_col,
        'label_cols': target_names,
        'radius': a.radius, 'nbits': a.nbits,
        'kept': kept, 'dropped': dropped
    }
    if transform_desc is not None:
        meta['target_transform'] = transform_desc    
    joblib.dump({'model': final_model, 'meta': meta}, a.model)

    with open(a.stats, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(a.cv_report, 'w', encoding='utf-8') as f:
        json.dump(fold_reports, f, ensure_ascii=False, indent=2)

    print(json.dumps({'saved_model': a.model, 'stats': a.stats, 'targets': target_names,
                      'kept': kept, 'dropped': dropped}, ensure_ascii=False))

if __name__ == '__main__':
    main()
