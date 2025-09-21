from __future__ import annotations
from typing import Dict
from .utils import parse_bool

def read_namelist(path: str) -> Dict[str, str]:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            toks = s.split()
            if len(toks) >= 2:
                m[toks[0]] = toks[1]
    return m

def read_modpara(path: str) -> Dict[str, float | int | bool | str]:
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#",1)[0].strip()
            if not s: continue
            toks = s.split()
            if len(toks) < 2: continue
            mp[toks[0]] = toks[1]

    N = int(mp.get("Nsite", "0"))
    out: Dict[str, float | int | bool | str] = {
        "Nsite": N,
        "CIPSIGrandCanonical": parse_bool(mp.get("CIPSIGrandCanonical", "true")),
        "CIPSISeeds": int(mp.get("CIPSISeeds", "32")),
        "CIPSICycles": int(mp.get("CIPSICycles", "12")),
        "CIPSIAddPerCycle": int(mp.get("CIPSIAddPerCycle", "64")),
        "CIPSIPrune": int(mp.get("CIPSIPrune", str(2**N if N>0 else 0))),
        "CIPSIEps": float(mp.get("CIPSIEps", "1e-6")),
        "CIPSISectorSz": mp.get("CIPSISectorSz", None),
        "CIPSIRandomSeed": int(mp.get("CIPSIRandomSeed", "1337")),
        "CIPSISeedMode": mp.get("CIPSISeedMode", "random").lower(),
        "CIPSISeedPool": int(mp.get("CIPSISeedPool", "0")),
    }
    if out["CIPSISectorSz"] is not None:
        try:
            out["CIPSISectorSz"] = float(out["CIPSISectorSz"])  # type: ignore
        except Exception:
            out["CIPSISectorSz"] = None
    return out
