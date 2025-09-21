from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
from .basis import diag_energy_bit, apply_local_op

def build_subspace_matrix(basis_bits: List[int], N:int, diag_terms, bilinear_terms):
    B = len(basis_bits)
    index = {b:i for i,b in enumerate(basis_bits)}
    rows=[]; cols=[]; data=[]
    for i,b in enumerate(basis_bits):
        e = diag_energy_bit(b, diag_terms)
        rows.append(i); cols.append(i); data.append(e)
    for i,b in enumerate(basis_bits):
        for (ii,si,jj,sj, kk,sk,ll,sl, c) in bilinear_terms:
            ok1, s1 = apply_local_op(b, kk, sl, sk)
            if not ok1: continue
            ok2, s2 = apply_local_op(s1, ii, sj, si)
            if not ok2: continue
            j = index.get(s2, -1)
            if j < 0: continue
            rows.append(j); cols.append(i); data.append(c)
    H = csr_matrix((np.array(data,dtype=np.complex128),
                    (np.array(rows),np.array(cols))), shape=(B,B))
    H = (H + H.getH())*0.5
    return H

def _build_range_block(range_start:int, range_end:int, basis_bits, diag_terms, bilinear_terms, index):
    rows = []; cols = []; data = []
    for i in range(range_start, range_end):
        b = basis_bits[i]
        e = diag_energy_bit(b, diag_terms)
        rows.append(i); cols.append(i); data.append(e)
    for i in range(range_start, range_end):
        b = basis_bits[i]
        for (ii,si,jj,sj, kk,sk,ll,sl, c) in bilinear_terms:
            ok1, s1 = apply_local_op(b, kk, sl, sk)
            if not ok1: continue
            ok2, s2 = apply_local_op(s1, ii, sj, si)
            if not ok2: continue
            j = index.get(s2, -1)
            if j < 0: continue
            rows.append(j); cols.append(i); data.append(c)
    return (np.array(rows, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(data, dtype=np.complex128))

def build_subspace_matrix_blocked(basis_bits, N, diag_terms, bilinear_terms, block_size=4096, procs=0, verbose=True):
    B = len(basis_bits)
    index = {b:i for i,b in enumerate(basis_bits)}
    ranges = [(s, min(s+block_size, B)) for s in range(0, B, block_size)]
    rows_all = []; cols_all = []; data_all = []
    if procs and len(ranges) > 1:
        if verbose:
            print(f"[Build] Blocked CSR: B={B}, blocks={len(ranges)}, block_size={block_size}, procs={procs}")
        with ProcessPoolExecutor(max_workers=procs) as ex:
            futs = [ex.submit(_build_range_block, s, e, basis_bits, diag_terms, bilinear_terms, index) for (s,e) in ranges]
            for fut in as_completed(futs):
                r, c, d = fut.result()
                rows_all.append(r); cols_all.append(c); data_all.append(d)
    else:
        if verbose:
            print(f"[Build] Blocked CSR (serial): B={B}, blocks={len(ranges)}, block_size={block_size}")
        for (s,e) in ranges:
            r, c, d = _build_range_block(s, e, basis_bits, diag_terms, bilinear_terms, index)
            rows_all.append(r); cols_all.append(c); data_all.append(d)

    rows = np.concatenate(rows_all) if rows_all else np.array([], dtype=np.int32)
    cols = np.concatenate(cols_all) if cols_all else np.array([], dtype=np.int32)
    data = np.concatenate(data_all) if data_all else np.array([], dtype=np.complex128)
    H = csr_matrix((data, (rows, cols)), shape=(B, B))
    H = (H + H.getH()) * 0.5
    return H
