def write_namelist(
    namelist_path: str,
    *,
    modpara: str,
    interall: str,
    locspin: str | None = None,
    greenone: str | None = None,
    greentwo: str | None = None,
) -> None:
    import os

    base = os.path.dirname(os.path.abspath(namelist_path))

    def rel(p: str | None) -> str | None:
        if p is None:
            return None
        ap = os.path.abspath(p)
        try:
            return os.path.relpath(ap, base)
        except Exception:
            return ap

    KEYW = 12
    def line(k: str, v: str) -> str:
        return f"{k:<{KEYW}} {v}\n"

    with open(namelist_path, "w", encoding="utf-8") as f:
        f.write(line("ModPara",  rel(modpara)))
        f.write(line("InterAll", rel(interall)))
        if locspin:
            f.write(line("LocSpin",  rel(locspin)))
        if greenone:
            f.write(line("OneBodyG", rel(greenone)))
        if greentwo:
            f.write(line("TwoBodyG", rel(greentwo)))


def default_seed_pool(N:int, seeds:int, grand:bool, sector_sz:float|None) -> int:
    import math as _math
    """
    CIPSISeedPool のデフォルト値。
    - 基本: pool ≈ k × seeds（Nが大きいほど k を少し下げる）
    - 過小を防ぐ床: 2^(sqrt(N)/2)
    - 固定セクター: C(N, N↑) を上限に
    - GC: 緩い安全上限 2^min(N,26)
    """
    if N <= 64:
        k = 8
    elif N <= 144:
        k = 7
    elif N <= 256:
        k = 6
    else:
        k = 5

    floor_explore = int(2 ** (_math.sqrt(N)/2))
    pool = max(seeds * k, floor_explore, seeds)

    if not grand and sector_sz is not None:
        n_up = int(round(sector_sz + N/2))
        if 0 <= n_up <= N:
            try:
                dim = _math.comb(N, n_up)
                pool = min(pool, max(seeds, dim))
            except Exception:
                pass
    else:
        dim_soft_cap = 1 << min(N, 26)
        pool = min(pool, max(seeds, dim_soft_cap))

    return int(pool)

def write_modpara_cipsi(Lx:int, Ly:int, path:str,
                        grand:bool=True, seeds:int=32, cycles:int=20,
                        add_per:int=100, prune:int|None=None, eps:float=1e-6,
                        sector_sz:float|None=None, rng:int|None=1337,
                        seed_mode:str|None=None, seed_pool:int|None=None) -> None:
    """modpara.def（整形出力）。
    - CIPSISeedMode は未指定なら 'diag'
    - CIPSISeedPool は未指定なら default_seed_pool() で決定
    """
    N = Lx * Ly
    if prune is None:
        prune = 2**N

    if seed_mode is None:
        seed_mode = "diag"
    if seed_pool is None:
        seed_pool = default_seed_pool(N, int(seeds), bool(grand), sector_sz)

    def _b2s(b:bool) -> str: return "true" if b else "false"
    KEYW = 20
    def _line(k, v): return f"{k:<{KEYW}} {v}\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(_line("Nsite",               N))
        f.write(_line("CIPSIGrandCanonical", _b2s(grand)))
        if sector_sz is not None:
            f.write(_line("CIPSISectorSz",   sector_sz))
        f.write(_line("CIPSISeeds",          seeds))
        f.write(_line("CIPSICycles",         cycles))
        f.write(_line("CIPSIAddPerCycle",    add_per))
        f.write(_line("CIPSIPrune",          prune))
        f.write(_line("CIPSIEps",            f"{eps:.6g}"))
        # 常に出力
        f.write(_line("CIPSISeedMode",       seed_mode))
        f.write(_line("CIPSISeedPool",       int(seed_pool)))
        # if rng is not None:
        #     f.write(_line("CIPSIRandomSeed", int(rng)))

def write_greenone(Lx: int, Ly: int, greenone_path: str) -> None:
    N = Lx * Ly
    count = 2 * N
    with open(greenone_path, "w", encoding="utf-8") as f:
        f.write("===============================\n")
        f.write(f"NCisAjs {count}\n")
        f.write("===============================\n")
        f.write("======== Green functions ======\n")
        f.write("===============================\n")
        for i in range(N):
            f.write(f"{i} 0 {i} 0\n")
            f.write(f"{i} 1 {i} 1\n")

def write_locspin(Lx: int, Ly: int, twoSz: int, path: str) -> None:
    N = Lx * Ly
    with open(path, "w", encoding="utf-8") as f:
        f.write("================================\n")
        f.write(f"NlocalSpin    {N}\n")
        f.write("================================\n")
        f.write("========i_1LocSpn_0IteElc ======\n")
        f.write("================================\n")
        for i in range(N):
            f.write(f"{i:5d}{twoSz:5d}\n")

def write_greentwo(Lx: int, Ly: int, greentwo_path: str, include_spinflip: bool = True) -> None:
    N = Lx * Ly
    per_pair = 4 + (2 if include_spinflip else 0)
    total = N * N * per_pair
    with open(greentwo_path, "w", encoding="utf-8") as f:
        f.write("============================================\n")
        f.write(f"NCisAjsCktAltDC {total}\n")
        f.write("============================================\n")
        if include_spinflip:
            f.write("===== Green functions for SzSz, S+S-, and N =====\n")
        else:
            f.write("===== Green functions for SzSz and N =====\n")
        f.write("============================================\n")
        for i in range(N):
            for j in range(N):
                f.write(f"{i} 0 {i} 0 {j} 0 {j} 0\n")
                f.write(f"{i} 0 {i} 0 {j} 1 {j} 1\n")
                f.write(f"{i} 1 {i} 1 {j} 0 {j} 0\n")
                f.write(f"{i} 1 {i} 1 {j} 1 {j} 1\n")
                if include_spinflip:
                    f.write(f"{i} 0 {i} 1 {j} 1 {j} 0\n")
                    f.write(f"{i} 1 {i} 0 {j} 0 {j} 1\n")
