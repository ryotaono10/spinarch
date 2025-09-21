from __future__ import annotations
import numpy as np

# ---- Numba Availability ------------------------------------------------------
try:
    import numba as nb
    NUMBA_OK = True
except Exception:
    nb = None  # type: ignore
    NUMBA_OK = False

# ---- Kernels (Numba or Stubs) -----------------------------------------------
if NUMBA_OK:
    @nb.njit(cache=True)
    def _diag_energy_bit_nb(bit, di, dsi, dk, dsk, dcr, dci):
        e_real = 0.0
        e_imag = 0.0
        L = di.shape[0]
        for t in range(L):
            ui = (bit >> di[t]) & 1
            uk = (bit >> dk[t]) & 1
            n_i = ui if dsi[t] == 0 else (1 - ui)
            n_k = uk if dsk[t] == 0 else (1 - uk)
            prod = n_i * n_k
            e_real += dcr[t] * prod
            e_imag += dci[t] * prod
        return e_real, e_imag

    @nb.njit(cache=True)
    def _apply_local_nb(bit, site, s_from, s_to):
        up = (bit >> site) & 1
        if s_to == s_from:
            n = up if s_to == 0 else (1 - up)
            return (1 if n != 0 else 0), bit
        need_up = 1 if s_from == 0 else 0
        if up != need_up:
            return 0, bit
        set_up = 1 if s_to == 0 else 0
        newbit = (bit | (1 << site)) if set_up == 1 else (bit & ~(1 << site))
        return 1, newbit

    @nb.njit(cache=True)
    def _binsearch(sorted_bits, target):
        lo = 0
        hi = sorted_bits.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            v = sorted_bits[mid]
            if v < target:
                lo = mid + 1
            else:
                hi = mid
        if lo < sorted_bits.shape[0] and sorted_bits[lo] == target:
            return lo
        return -1

    @nb.njit(cache=True)
    def _h_matvec_nb(xr, xi, basis_bits, sorted_bits, invperm,
                     di, dsi, dk, dsk, dcr, dci,
                     bi, bsi, bj, bsj, bk, bsk, bl, bsl, bcr, bci):
        B = basis_bits.shape[0]
        yr = np.zeros(B, dtype=np.float64)
        yi = np.zeros(B, dtype=np.float64)

        # diagonal part
        for i in range(B):
            b = basis_bits[i]
            de_r, de_i = _diag_energy_bit_nb(b, di, dsi, dk, dsk, dcr, dci)
            xr_i = xr[i]; xi_i = xi[i]
            yr[i] += de_r * xr_i - de_i * xi_i
            yi[i] += de_r * xi_i + de_i * xr_i

        # off-diagonal part
        T = bi.shape[0]
        for i in range(B):
            b = basis_bits[i]
            xr_i = xr[i]; xi_i = xi[i]
            for t in range(T):
                ok1, s1 = _apply_local_nb(b, bk[t], bsl[t], bsk[t])
                if ok1 == 0:
                    continue
                ok2, s2 = _apply_local_nb(s1, bi[t], bsj[t], bsi[t])
                if ok2 == 0:
                    continue
                pos = _binsearch(sorted_bits, s2)
                if pos < 0:
                    continue
                j = invperm[pos]
                cr = bcr[t]; ci = bci[t]
                yr[j] += cr * xr_i - ci * xi_i
                yi[j] += cr * xi_i + ci * xr_i

        return yr, yi

    # 並列要求が来ても、ここでは安全にシリアル関数を使う（原子加算なしで簡潔に）
    def _h_matvec_nb_par(xr, xi, basis_bits, sorted_bits, invperm,
                         di, dsi, dk, dsk, dcr, dci,
                         bi, bsi, bj, bsj, bk, bsk, bl, bsl, bcr, bci):
        return _h_matvec_nb(xr, xi, basis_bits, sorted_bits, invperm,
                            di, dsi, dk, dsk, dcr, dci,
                            bi, bsi, bj, bsj, bk, bsk, bl, bsl, bcr, bci)

else:
    # ---- Numbaが無い場合：インポートだけ通すスタブ ----
    def _h_matvec_nb(*args, **kwargs):
        raise RuntimeError("Numba acceleration is unavailable (NUMBA_OK=False).")

    def _h_matvec_nb_par(*args, **kwargs):
        raise RuntimeError("Numba acceleration is unavailable (NUMBA_OK=False).")

# ---- Packing helpers (Numbaの有無に関係なく使用) ----------------------------
def pack_terms_arrays(diag_terms, bilinear_terms):
    """Python dict/list → （Numba/JITも扱いやすい）ndarray 群にパック"""
    # diag
    if len(diag_terms) == 0:
        di = dsi = dk = dsk = np.empty(0, dtype=np.int32)
        dcr = dci = np.empty(0, dtype=np.float64)
    else:
        keys = list(diag_terms.keys())
        di  = np.fromiter((k[0] for k in keys), dtype=np.int32, count=len(keys))
        dsi = np.fromiter((k[1] for k in keys), dtype=np.int32, count=len(keys))
        dk  = np.fromiter((k[2] for k in keys), dtype=np.int32, count=len(keys))
        dsk = np.fromiter((k[3] for k in keys), dtype=np.int32, count=len(keys))
        vals = np.fromiter((diag_terms[k] for k in keys), dtype=np.complex128, count=len(keys))
        dcr = vals.real.astype(np.float64, copy=False)
        dci = vals.imag.astype(np.float64, copy=False)

    # bilinear
    if len(bilinear_terms) == 0:
        bi = bsi = bj = bsj = bk = bsk = bl = bsl = np.empty(0, dtype=np.int32)
        bcr = bci = np.empty(0, dtype=np.float64)
    else:
        T = len(bilinear_terms)
        bi  = np.empty(T, dtype=np.int32)
        bsi = np.empty(T, dtype=np.int32)
        bj  = np.empty(T, dtype=np.int32)
        bsj = np.empty(T, dtype=np.int32)
        bk  = np.empty(T, dtype=np.int32)
        bsk = np.empty(T, dtype=np.int32)
        bl  = np.empty(T, dtype=np.int32)
        bsl = np.empty(T, dtype=np.int32)
        bcr = np.empty(T, dtype=np.float64)
        bci = np.empty(T, dtype=np.float64)
        for t, row in enumerate(bilinear_terms):
            # row = (ii,si,jj,sj, kk,sk,ll,sl, c)
            bi[t]  = int(row[0]); bsi[t] = int(row[1])
            bj[t]  = int(row[2]); bsj[t] = int(row[3])
            bk[t]  = int(row[4]); bsk[t] = int(row[5])
            bl[t]  = int(row[6]); bsl[t] = int(row[7])
            c = row[8]
            bcr[t] = float(np.real(c))
            bci[t] = float(np.imag(c))
    return di, dsi, dk, dsk, dcr, dci, bi, bsi, bj, bsj, bk, bsk, bl, bsl, bcr, bci
