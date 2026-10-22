"""Independent verification script for Dynare 8 on MATLAB R2026a parity vs puremacro.

Verifies:
1. Economically meaningful RBC model with g = gy*STEADY_STATE(y) (parity <= 1e-8).
2. Minimal #177 reproduction model with x = STEADY_STATE(y) (parity <= 1e-8).
3. Permanent-shock model (endval followed by steady) (parity <= 1e-8).
4. Permanent-shock model (endval without steady) (parity <= 1e-8).
"""
import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent.parent)
if sys.path[0] != repo_root:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
from puremacro.dsge import load_mod


def main():
    ref_dir = Path(__file__).parent

    # 1. RBC Model Parity
    rbc_dynare_df = pd.read_csv(ref_dir / "rbc_dynare_out.csv")
    with open(ref_dir / "rbc_steady_state.mod") as f:
        rbc_mod_text = f.read()

    m_rbc = load_mod(rbc_mod_text)
    shocks = np.zeros((40, 1))
    shocks[0, 0] = 0.05
    res_rbc = m_rbc.solve_perfect_foresight(periods=40, shocks=shocks, tol=1e-12)

    dynare_sim_40 = rbc_dynare_df.iloc[1:41].reset_index(drop=True)
    puremacro_sim_40 = res_rbc.path[dynare_sim_40.columns].reset_index(drop=True)

    diff_rbc = np.abs(puremacro_sim_40.values - dynare_sim_40.values)
    max_diff_rbc = float(np.max(diff_rbc))
    print("=== 1. RBC Model Parity (g = gy*STEADY_STATE(y)) ===")
    for col in dynare_sim_40.columns:
        col_diff = float(np.max(np.abs(puremacro_sim_40[col].values - dynare_sim_40[col].values)))
        print(f"  {col}: max abs diff = {col_diff:.3e}")
    print(f"Overall Max Abs Diff: {max_diff_rbc:.3e}")
    assert max_diff_rbc <= 1e-8, f"RBC parity failed: {max_diff_rbc} > 1e-8"
    print(">>> PASS: RBC parity <= 1e-8 verified!\n")

    # 2. Minimal 177 Model Parity
    min_dynare_df = pd.read_csv(ref_dir / "minimal_177_dynare_out.csv")
    with open(ref_dir / "minimal_177.mod") as f:
        min_mod_text = f.read()
    m_min = load_mod(min_mod_text)
    e_min = np.zeros((10, 1))
    e_min[0, 0] = 0.5
    res_min = m_min.solve_perfect_foresight(periods=10, shocks=e_min, tol=1e-12)
    min_dynare_10 = min_dynare_df.iloc[1:11].reset_index(drop=True)
    min_pure_10 = res_min.path[min_dynare_10.columns].reset_index(drop=True)
    diff_min = np.abs(min_pure_10.values - min_dynare_10.values)
    max_diff_min = float(np.max(diff_min))
    print("=== 2. Minimal 177 Parity (x = STEADY_STATE(y)) ===")
    for col in min_dynare_10.columns:
        col_diff = float(np.max(np.abs(min_pure_10[col].values - min_dynare_10[col].values)))
        print(f"  {col}: max abs diff = {col_diff:.3e}")
    print(f"Overall Max Abs Diff: {max_diff_min:.3e}")
    assert max_diff_min <= 1e-8, f"Minimal parity failed: {max_diff_min} > 1e-8"
    print(">>> PASS: Minimal 177 parity <= 1e-8 verified!\n")

    # 3. Endval with steady Parity
    es_dynare_df = pd.read_csv(ref_dir / "endval_with_steady_dynare_out.csv")
    with open(ref_dir / "endval_with_steady.mod") as f:
        es_mod_text = f.read()
    m_es = load_mod(es_mod_text)
    res_es = m_es.solve_perfect_foresight(periods=10, tol=1e-12)
    es_dynare_10 = es_dynare_df.iloc[1:11].reset_index(drop=True)
    es_pure_10 = res_es.path[es_dynare_10.columns].reset_index(drop=True)
    diff_es = np.abs(es_pure_10.values - es_dynare_10.values)
    max_diff_es = float(np.max(diff_es))
    print("=== 3. Endval WITH Steady Parity ===")
    for col in es_dynare_10.columns:
        col_diff = float(np.max(np.abs(es_pure_10[col].values - es_dynare_10[col].values)))
        print(f"  {col}: max abs diff = {col_diff:.3e}")
    print(f"Overall Max Abs Diff: {max_diff_es:.3e}")
    assert max_diff_es <= 1e-8, f"Endval with steady parity failed: {max_diff_es} > 1e-8"
    print(">>> PASS: Endval with steady parity <= 1e-8 verified!\n")

    # 4. Endval WITHOUT steady Parity
    ens_dynare_df = pd.read_csv(ref_dir / "endval_no_steady_dynare_out.csv")
    with open(ref_dir / "endval_no_steady.mod") as f:
        ens_mod_text = f.read()
    m_ens = load_mod(ens_mod_text)
    res_ens = m_ens.solve_perfect_foresight(periods=10, tol=1e-12)
    ens_dynare_10 = ens_dynare_df.iloc[1:11].reset_index(drop=True)
    ens_pure_10 = res_ens.path[ens_dynare_10.columns].reset_index(drop=True)
    diff_ens = np.abs(ens_pure_10.values - ens_dynare_10.values)
    max_diff_ens = float(np.max(diff_ens))
    print("=== 4. Endval WITHOUT Steady Parity ===")
    for col in ens_dynare_10.columns:
        col_diff = float(np.max(np.abs(ens_pure_10[col].values - ens_dynare_10[col].values)))
        print(f"  {col}: max abs diff = {col_diff:.3e}")
    print(f"Overall Max Abs Diff: {max_diff_ens:.3e}")
    assert max_diff_ens <= 1e-8, f"Endval without steady parity failed: {max_diff_ens} > 1e-8"
    print(">>> PASS: Endval without steady parity <= 1e-8 verified!\n")

    print("ALL 4 DYNARE REFERENCE MODELS MATCH WITHIN TOLERANCE <= 1e-8.")


if __name__ == "__main__":
    main()
