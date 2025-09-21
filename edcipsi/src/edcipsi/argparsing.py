# edcipsi/argparsing.py
from argparse import ArgumentParser

def build_parser() -> ArgumentParser:
    ap = ArgumentParser(description="ED-CIPSI (spin-1/2, HPhi InterAll) with optional Numba Hx and threading.")
    ap.add_argument("namelist", type=str, help="HPhi-style namelist.def")
    ap.add_argument("--grand-canonical", action="store_true", help="override: do not fix Sz sector")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--add-per-cycle", type=int, default=None)
    ap.add_argument("--prune", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--outfile", default=None)

    # perf
    ap.add_argument("--threads", type=int, default=None, help="set OMP/MKL thread env vars")
    ap.add_argument("--accel-matvec", action="store_true", help="use Numba LinearOperator H·x if available")
    ap.add_argument("--nb-parallel", action="store_true",
                    help="parallelize Numba matvec with OpenMP (prange+atomics)")
    ap.add_argument("--build-blocked", action="store_true",
                    help="build CSR H in row blocks (memory-friendly; can parallelize)")
    ap.add_argument("--block-size", type=int, default=4096,
                    help="rows per block when building CSR")
    # Heat-Bath style branch preselection
    ap.add_argument("--hb-preselect", action="store_true",
                    help="enable Heat-Bath style preselection by |c_a|*|alpha_t| >= Gamma")
    ap.add_argument("--hb-gamma", type=float, default=None,
                    help="Gamma threshold for preselection; default auto: sqrt(eps)*max|alpha_t|")
    # PT2 estimation
    ap.add_argument("--pt2", action="store_true",
                    help="compute Epstein–Nesbet PT2 from connected amplitudes")
    ap.add_argument("--level-shift", type=float, default=0.0,
                    help="optional level shift added to denominators in PT2 (stabilization)")
    ap.add_argument("--build-procs", type=int, default=0,
                    help="use N processes to build blocks in parallel (0=serial)")
    # CIPSISeedMode
    ap.add_argument("--seed-mode", choices=["random","diag"], default=None)
    ap.add_argument("--seed-pool", type=int, default=None)
    # RNG
    ap.add_argument("--seed", type=int, default=None, help="random seed override")
    return ap
