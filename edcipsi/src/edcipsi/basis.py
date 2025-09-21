from __future__ import annotations
from typing import List, Optional
import random


def ensure_unique_basis(basis_bits: List[int], where: str = "") -> None:
    """Raise ValueError if any duplicate computational-basis states are present.
    The computational basis here is orthonormal, so duplicates would create
    spurious overlap (=1) between distinct indices. This check is O(B).
    """
    seen = set()
    for idx, b in enumerate(basis_bits):
        if b in seen:
            msg = f"Duplicate basis state detected at index {idx}: bit={b} (hex={b:#x})."
            if where:
                msg += f" Location: {where}"
            raise ValueError(msg)
        seen.add(b)


def n_on_site(bit: int, site: int, spin: int) -> int:
    """spin: 0→↑, 1→↓"""
    up = (bit >> site) & 1
    return up if spin==0 else (1-up)

def diag_energy_bit(bit: int, diag_terms) -> complex:
    e = 0.0 + 0.0j
    for (i,si,k,sk), c in diag_terms.items():
        e += c * (n_on_site(bit,i,si) * n_on_site(bit,k,sk))
    return e

def apply_local_op(bit: int, site: int, s_from: int, s_to: int):
    """Apply |s_to><s_from| to site Success(1,newbit)/Failier(0,bit)"""
    up = (bit >> site) & 1
    if s_to == s_from:
        n = up if s_to==0 else (1-up)
        return (1 if n!=0 else 0), bit
    need_up = 1 if s_from==0 else 0
    if up != need_up:
        return 0, bit
    set_up = 1 if s_to==0 else 0
    newbit = (bit | (1<<site)) if set_up==1 else (bit & ~(1<<site))
    return 1, newbit

def pick_low_diag_seeds(N:int, n_keep:int, pool_size:int, diag_terms, gc:bool, target_up:Optional[int], rng:random.Random):
    """E_diag が低い順に n_keep 個ビットを返す。If diag_terms empty, return None."""
    if not diag_terms:
        return None
    pool_size = max(pool_size, n_keep)
    cands = set()
    while len(cands) < pool_size:
        if gc:
            b = 0
            for site in range(N):
                if rng.random() < 0.5:
                    b |= (1 << site)
        else:
            positions = rng.sample(range(N), target_up)  # type: ignore
            b = 0
            for p in positions:
                b |= (1 << p)
        cands.add(b)

    scored = []
    for b in cands:
        e = diag_energy_bit(b, diag_terms)
        scored.append((float(e.real), b))
    scored.sort(key=lambda t: t[0])
    return [b for _, b in scored[:n_keep]]

