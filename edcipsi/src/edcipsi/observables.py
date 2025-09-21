from __future__ import annotations
from typing import List
import numpy as np
from .basis import apply_local_op

def expect_greenone(basis_bits, vec, ops):
    index = {b:i for i,b in enumerate(basis_bits)}
    out = []
    for (i, si, j, sj) in ops:
        val = 0.0 + 0.0j
        for a, b in enumerate(basis_bits):
            ok, b2 = apply_local_op(b, j, sj, si)
            if not ok: continue
            jidx = index.get(b2, -1)
            if jidx < 0: continue
            val += np.conjugate(vec[jidx]) * vec[a]
        out.append(val)
    return out

def expect_greentwo(basis_bits, vec, N, ops):
    index = {b:i for i,b in enumerate(basis_bits)}
    out = []
    for (i,si,j,sj,k,sk,l,sl) in ops:
        val = 0.0 + 0.0j
        for a, b in enumerate(basis_bits):
            ok1, s1 = apply_local_op(b, l, sl, sk)
            if not ok1: continue
            ok2, s2 = apply_local_op(s1, j, sj, si)
            if not ok2: continue
            jidx = index.get(s2, -1)
            if jidx < 0: continue
            val += np.conjugate(vec[jidx]) * vec[a]
        out.append(val)
    return out
