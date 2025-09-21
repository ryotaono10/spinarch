# edcipsi_gen/parse.py
from __future__ import annotations
import os, math
import numpy as np

def _strip(line: str) -> str: return line.split("#", 1)[0].strip()

def _try_parse_two_floats(s: str):
    try:
        toks = s.replace(",", " ").split()
        if len(toks) == 2:
            return float(toks[0]), float(toks[1])
    except Exception:
        pass
    try:
        toks = s.split()
        if len(toks) == 2:
            return float(toks[0]), float(toks[1])
    except Exception:
        pass
    return None

def _try_parse_tagged_vec(s: str, tag: str):
    low = s.lower().replace("=", " ").replace(":", " ").replace(",", " ")
    toks = low.split()
    if len(toks) >= 3 and toks[0] == tag:
        return float(toks[1]), float(toks[2])
    return None

def parse_spec(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    lines = [s for s in (_strip(x) for x in raw) if s]
    it = iter(lines)

    Lx = int(next(it)); Ly = int(next(it))
    a1 = (1.0, 0.0)
    a2 = (0.5, math.sqrt(3)/2.0)

    peek = next(it)
    n = None

    v1 = _try_parse_tagged_vec(peek, "a1")
    if v1 is not None:
        a1 = v1
        line = next(it)
        v2 = _try_parse_tagged_vec(line, "a2")
        if v2 is None:
            raise ValueError("spec: expected 'a2: bx by' after a1 line.")
        a2 = v2
        n = int(next(it))
    else:
        maybe = _try_parse_two_floats(peek)
        if maybe is not None:
            a1 = maybe
            line = next(it)
            maybe2 = _try_parse_two_floats(line)
            if maybe2 is None:
                raise ValueError("spec: expected two floats for a2 after a1 line.")
            a2 = maybe2
            n = int(next(it))
        else:
            n = int(peek)

    items = []
    for _ in range(n):
        toks = next(it).split()
        if len(toks) != 12:
            raise ValueError("Each R/J line must have 12 numbers.")
        Rx,Ry,Rz = (int(toks[0]), int(toks[1]), int(toks[2]))
        vals = [complex(t.replace('i','j')) for t in toks[3:]]
        J = np.array([[vals[0], vals[1], vals[2]],
                      [vals[3], vals[4], vals[5]],
                      [vals[6], vals[7], vals[8]]], dtype=complex)
        items.append((Rx,Ry,Rz,J))

    outfile = next(it)
    try:
        plotfile = next(it)
    except StopIteration:
        plotfile = f"lattice_{os.path.basename(outfile)}.png"

    return Lx, Ly, a1, a2, items, outfile, plotfile

def parse_cli_pairs(pairs: list):
    items = []
    for s in pairs:
        if ":" not in s:
            raise ValueError("Each --pair must be 'Rx Ry Rz : 9 J entries'")
        left, right = s.split(":", 1)
        Rx,Ry,Rz = (int(t) for t in left.strip().split())
        nums = [complex(t.replace('i','j')) for t in right.strip().split()]
        if len(nums) != 9: raise ValueError("J must have 9 numbers.")
        J = np.array([[nums[0], nums[1], nums[2]],
                      [nums[3], nums[4], nums[5]],
                      [nums[6], nums[7], nums[8]]], dtype=complex)
        items.append((Rx,Ry,Rz,J))
    return items

