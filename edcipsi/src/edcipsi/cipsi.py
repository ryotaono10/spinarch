from __future__ import annotations
from typing import List, Dict, Tuple
import math, numpy as np
from collections import defaultdict
from .basis import apply_local_op, diag_energy_bit
from .solver import solve_ground
from .nbkernels import NUMBA_OK

def connected_amplitudes(basis_bits, coeffs, bilinear_terms, hb_gamma=None, max_abs_coeff=None, terms_sorted=False):
    idx_to_bit = list(basis_bits)
    M = defaultdict(complex)
    whole_a_cut = (hb_gamma / max_abs_coeff) if (hb_gamma is not None and max_abs_coeff and max_abs_coeff>0) else 0.0
    for i, b in enumerate(idx_to_bit):
        ci = coeffs[i]
        abs_ci = abs(ci)
        if abs_ci < 1e-16:
            continue
        if hb_gamma is not None and abs_ci < whole_a_cut:
            continue
        for (ii,si,jj,sj, kk,sk,ll,sl, c) in bilinear_terms:
            if hb_gamma is not None and abs_ci * abs(c) < hb_gamma:
                if terms_sorted: break
                else: continue
            ok1, s1 = apply_local_op(b, kk, sl, sk)
            if not ok1: continue
            ok2, s2 = apply_local_op(s1, ii, sj, si)
            if not ok2 or s2 == b: continue
            M[s2] += c * ci
    return M

def select_new_configs(E, M_dict, diag_terms, used_set, add_max, eps, delta=1e-12):
    cands = []
    for bit, M in M_dict.items():
        if bit in used_set: continue
        Haa = diag_energy_bit(bit, diag_terms)
        denom = E - Haa
        w = (abs(M)**2) / max(abs(denom), delta)
        if w >= eps:
            cands.append((w, bit))
    cands.sort(key=lambda x: x[0], reverse=True)
    return [b for (_,b) in cands[:add_max]]

def compute_PT2(E, M_dict, diag_terms, level_shift=0.0):
    total = 0.0; n = 0
    for bit, M in M_dict.items():
        Hii = diag_energy_bit(bit, diag_terms)
        denom = (E - Hii)
        if level_shift:
            denom = denom + (level_shift if denom.real >= 0 else -level_shift)
        if abs(denom) < 1e-16:
            continue
        total += (abs(M)**2) / denom
        n += 1
    return float(total.real), n

def prune_by_coeff(basis_bits, vec, keep_max):
    if len(basis_bits) <= keep_max: return list(basis_bits)
    mags = np.abs(vec)
    order = np.argsort(-mags)
    keep_idx = set(order[:keep_max].tolist())
    return [b for i,b in enumerate(basis_bits) if i in keep_idx]

def run_cipsi_once(N:int, diag_terms, bilinear_terms,
                   grand_canonical:bool, seeds:int, cycles:int, add_per_cycle:int, prune:int, eps:float,
                   hb_gamma:float|None, hb_sorted:bool, max_abs_coeff:float,
                   threads:int|None, accel_matvec:bool, nb_parallel:bool,
                   build_blocked:bool, block_size:int, build_procs:int,
                   seed_mode:str, seed_pool:int, sector_Sz, rng):
    # 初期基底（あなたの元コードに合わせて簡約）
    basis = []
    used = set()

    def random_gc():
        b = 0
        for site in range(N):
            if rng.random() < 0.5:
                b |= (1<<site)
        return b

    if not grand_canonical:
        target_Sz = sector_Sz
        if target_Sz is None:
            target_up = N//2
        else:
            target_up = int(round(float(target_Sz) + N/2))
            target_up = max(0, min(N, target_up))
        import random as _r
        while len(basis) < seeds:
            positions = rng.sample(range(N), target_up)
            b=0
            for p in positions: b |= (1<<p)
            if b not in used:
                used.add(b); basis.append(b)
    else:
        while len(basis) < seeds:
            b = random_gc()
            if b not in used:
                used.add(b); basis.append(b)

    use_nb = bool(accel_matvec and NUMBA_OK)
    use_nb_parallel = bool(use_nb and nb_parallel)

    # 反復
    for cyc in range(cycles):
        E, vec = solve_ground(basis, N, diag_terms, bilinear_terms,
                              use_nb=use_nb, use_nb_parallel=use_nb_parallel,
                              build_blocked=build_blocked, block_size=block_size, build_procs=build_procs)
        print(f"[Cycle {cyc+1}/{cycles}] Basis={len(basis)}, E0={E:.8f}  E0/site={E.real/N:.6f} (|Im|={abs(E.imag):.2e})")
        M = connected_amplitudes(basis, vec, bilinear_terms, hb_gamma=hb_gamma, max_abs_coeff=max_abs_coeff, terms_sorted=hb_sorted)
        new_bits = select_new_configs(E, M, diag_terms, set(basis), add_per_cycle, eps)
        if not new_bits:
            break
        for b in new_bits:
            if b not in used:
                used.add(b); basis.append(b)

        # prune の前にもう一回だけ軽く固有計算（あなたの元コード同様）
        E, vec = solve_ground(basis, N, diag_terms, bilinear_terms,
                              use_nb=use_nb, use_nb_parallel=use_nb_parallel,
                              build_blocked=build_blocked, block_size=block_size, build_procs=build_procs)
        if len(basis) > prune:
            basis = prune_by_coeff(basis, vec, prune)

    # 最終
    E, vec = solve_ground(basis, N, diag_terms, bilinear_terms,
                          use_nb=use_nb, use_nb_parallel=use_nb_parallel,
                          build_blocked=build_blocked, block_size=block_size, build_procs=build_procs)
    return E, vec, basis
