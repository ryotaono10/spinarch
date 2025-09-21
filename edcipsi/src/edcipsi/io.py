from __future__ import annotations
from typing import List, Tuple, Dict
from collections import defaultdict

def read_greenone_def(path: str) -> List[tuple]:
    ops = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#",1)[0].strip()
            if not s: continue
            if s[0].isalpha() or s[0] in "=_-":  # headers
                continue
            toks = s.replace(",", " ").split()
            if len(toks) < 4: continue
            i, si, j, sj = (int(toks[t]) for t in range(4))
            ops.append((i, si, j, sj))
    return ops

def read_greentwo_def(path: str) -> List[tuple]:
    ops = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#",1)[0].strip()
            if not s: continue
            if s[0].isalpha() or s[0] in "=_-":
                continue
            toks = s.replace(",", " ").split()
            if len(toks) < 8: continue
            i,si,j,sj,k,sk,l,sl = (int(toks[t]) for t in range(8))
            ops.append((i,si,j,sj,k,sk,l,sl))
    return ops

def read_interall(path: str):
    """
     HPhi format InterAll: 8 ints + 2 floats(Re, Im).
    """
    diag_terms: Dict[tuple, complex] = defaultdict(complex)
    bilinear_terms: List[tuple] = []
    header = 0; ignored = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#",1)[0].strip()
            if not s: continue
            header += 1
            if header <= 4:  # skip header zone
                continue
            toks = s.replace(",", " ").split()
            if len(toks) < 10:
                continue
            i,si,j,sj,k,sk,l,sl = (int(toks[t]) for t in range(8))
            re, im = float(toks[8]), float(toks[9])
            c = complex(re, im)
            if (i==j) and (k==l):
                if (si==sj) and (sk==sl):
                    diag_terms[(i,si,k,sk)] += c
                else:
                    bilinear_terms.append((i,si,j,sj,k,sk,l,sl,c))
            else:
                ignored += 1
    if ignored>0:
        print(f"[INFO] Ignored {ignored} InterAll rows not matching local-local form (i==j,k==l).")
    return diag_terms, bilinear_terms
