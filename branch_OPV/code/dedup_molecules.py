#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deduplicate generated molecules:
1) Canonicalize SMILES with RDKit
2) Remove overlap with original dataset
3) Remove duplicates inside generated set
Outputs a deduplicated CSV + prints a small report.

Usage:
  python /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/plt/code/dedup_molecules.py \
    --gen /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_18/outputs/all_molecules_physical.csv \
    --origin /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/Data_origin.csv \
    --out /public/home/users/haoxw/generate_AI/objective_optimization_object/project_multy_objective_rl/ckpts/gpu_rl_vgae_soft_congen_18/outputs/all_molecules_dedup.csv
"""

import argparse
import pandas as pd
from rdkit import Chem


def pick_smiles_col(df: pd.DataFrame) -> str:
    for c in ["smiles", "SMILES", "Smiles"]:
        if c in df.columns:
            return c
    raise KeyError("No SMILES column found. Expected one of: smiles/SMILES/Smiles")


def canonical_smiles(s):
    if not isinstance(s, str) or not s.strip():
        return None
    m = Chem.MolFromSmiles(s.strip())
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="generated csv, e.g. all_molecules_physical.csv")
    ap.add_argument("--origin", required=True, help="original csv, e.g. Data_origin.csv")
    ap.add_argument("--out", required=True, help="output deduplicated csv")
    ap.add_argument("--keep-invalid", action="store_true",
                    help="keep rows with invalid/empty SMILES (default: drop them)")
    args = ap.parse_args()

    gen = pd.read_csv(args.gen)
    orig = pd.read_csv(args.origin)

    gen_col = pick_smiles_col(gen)
    orig_col = pick_smiles_col(orig)

    gen["__cansmi__"] = gen[gen_col].map(canonical_smiles)
    orig["__cansmi__"] = orig[orig_col].map(canonical_smiles)

    n_gen = len(gen)
    n_orig = len(orig)
    n_gen_invalid = gen["__cansmi__"].isna().sum()
    n_orig_invalid = orig["__cansmi__"].isna().sum()

    if not args.keep_invalid:
        gen = gen[gen["__cansmi__"].notna()].copy()

    orig_set = set(orig["__cansmi__"].dropna().unique())

    # remove overlap with original dataset
    mask_not_in_orig = ~gen["__cansmi__"].isin(orig_set)
    gen_no_orig = gen[mask_not_in_orig].copy()

    # remove duplicates within generated set
    mask_unique = ~gen_no_orig["__cansmi__"].duplicated(keep="first")
    gen_dedup = gen_no_orig[mask_unique].copy()

    # store canonical smiles for traceability
    gen_dedup["smiles_canonical"] = gen_dedup["__cansmi__"]
    gen_dedup = gen_dedup.drop(columns=["__cansmi__"])

    gen_dedup.to_csv(args.out, index=False)

    print("Dedup report (canonical SMILES key)")
    print(f"  generated rows: {n_gen}")
    print(f"  original rows:  {n_orig}")
    print(f"  generated invalid SMILES: {n_gen_invalid}")
    print(f"  original invalid SMILES:  {n_orig_invalid}")
    print(f"  removed overlap with original: {(~mask_not_in_orig).sum()}")
    print(f"  removed duplicates in generated (after overlap removal): {(~mask_unique).sum()}")
    print(f"  remaining unique novel molecules: {len(gen_dedup)}")
    print(f"  written to: {args.out}")


if __name__ == "__main__":
    main()
