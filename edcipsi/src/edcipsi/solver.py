from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from .hbuilder import build_subspace_matrix, build_subspace_matrix_blocked
from .nbkernels import NUMBA_OK, _h_matvec_nb, _h_matvec_nb_par, pack_terms_arrays  # type: ignore

def lowest_eigpair(H):
    w, v = eigsh(H, k=1, which='SA', tol=1e-8, maxiter=5000)
    return w[0], v[:,0]

def solve_ground(basis, N, diag_terms, bilinear_terms,
                 use_nb=False, use_nb_parallel=False,
                 build_blocked=False, block_size=4096, build_procs=0):
    """Numba LinearOperator → 失敗時CSRのフォールバック"""
    if use_nb and NUMBA_OK:
        (di,dsi,dk,dsk,dcr,dci, bi,bsi,bj,bsj,bk,bsk,bl,bsl,bcr,bci) = pack_terms_arrays(diag_terms, bilinear_terms)
        basis_arr = np.array(basis, dtype=np.int64)
        sort_idx = np.argsort(basis_arr)
        sorted_bits = basis_arr[sort_idx]
        invperm = np.empty_like(sort_idx); invperm[sort_idx] = np.arange(sort_idx.size)
        def _matvec(v):
            xr = np.asarray(v).real.astype(np.float64, copy=False)
            xi = np.asarray(v).imag.astype(np.float64, copy=False)
            if use_nb_parallel:
                yr, yi = _h_matvec_nb_par(xr, xi, basis_arr, sorted_bits, invperm,
                                          di,dsi,dk,dsk,dcr,dci,
                                          bi,bsi,bj,bsj,bk,bsk,bl,bsl,bcr,bci)
            else:
                yr, yi = _h_matvec_nb(xr, xi, basis_arr, sorted_bits, invperm,
                                      di,dsi,dk,dsk,dcr,dci,
                                      bi,bsi,bj,bsj,bk,bsk,bl,bsl,bcr,bci)
            return yr + 1j*yi
        Lop = LinearOperator((len(basis), len(basis)), matvec=_matvec, dtype=np.complex128)
        try:
            w, v = eigsh(Lop, k=1, which='SA', tol=1e-8, maxiter=5000)
            return w[0], v[:,0]
        except Exception:
            pass
    # fallback CSR
    H = (build_subspace_matrix_blocked(basis, N, diag_terms, bilinear_terms,
                                       block_size=block_size, procs=build_procs, verbose=True)
         if build_blocked else
         build_subspace_matrix(basis, N, diag_terms, bilinear_terms))
    return lowest_eigpair(H)
