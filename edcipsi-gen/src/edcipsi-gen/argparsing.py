# edcipsi/argparsing.py
from argparse import ArgumentParser

def build_parser() -> ArgumentParser:
    ap = ArgumentParser(description="InterAll generator (triangular rhombus PBC) with ED-CIPSI namelist/modpara writer.")
    ap.add_argument("--spec", type=str, default=None, help="spec file path")
    ap.add_argument("--Lx", type=int, default=None); ap.add_argument("--Ly", type=int, default=None)
    ap.add_argument("--a1", type=float, nargs=2, metavar=("ax","ay"), default=None, help="lattice vector a1 (embedded)")
    ap.add_argument("--a2", type=float, nargs=2, metavar=("bx","by"), default=None, help="lattice vector a2 (embedded)")
    ap.add_argument("--pair", action="append", default=None,
                    help="Repeatable. 'Rx Ry Rz : 9 J entries' (row-major: Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz)")
    ap.add_argument("--out", type=str, default="interall.def")
    ap.add_argument("--plot", type=str, default="lattice.png")

    ap.add_argument("--locspin", type=str, default=None, help="path to write locspin.def (requires --twosz)")
    ap.add_argument("--twosz", type=int, default=None, help="value for all sites in locspin.def")
    ap.add_argument("--greenone", type=str, default="greenone.def", help="write GreenOne definition to this path")
    ap.add_argument("--greentwo", type=str, default="greentwo.def", help="write TwoBodyG definition (greentwo.def) for SzSz/Nq (+S+S-)")
    ap.add_argument("--no-spinflip", dest="no_spinflip", action="store_true", help="omit spin-flip terms (only SzSz & N)")

    # --- CIPSI-friendly outputs ---
    ap.add_argument("--for-cipsi", action="store_true", help="emit CIPSI-friendly namelist.def / modpara.def / calcmod.def")
    ap.add_argument("--namelist", type=str, default="namelist.def")
    ap.add_argument("--modpara",  type=str, default="modpara.def")
    ap.add_argument("--calcmod",  type=str, default="calcmod.def")

    # CIPSI knobs
    ap.add_argument("--cipsi-grand", dest="cipsi_grand", action="store_true", help="CIPSIGrandCanonical true (default true)")
    ap.add_argument("--cipsi-no-grand", dest="cipsi_no_grand", action="store_true", help="CIPSIGrandCanonical false")
    ap.add_argument("--cipsi-seeds", type=int, default=32)
    ap.add_argument("--cipsi-cycles", type=int, default=20)
    ap.add_argument("--cipsi-add", type=int, default=100, help="CIPSIAddPerCycle")
    ap.add_argument("--cipsi-prune", type=int, default=None)
    ap.add_argument("--cipsi-eps", type=float, default=1e-6)
    ap.add_argument("--cipsi-sector-sz", type=float, default=None)
    ap.add_argument("--cipsi-seed", type=int, default=1337, help="CIPSIRandomSeed (optional)")
    ap.add_argument("--cipsi-add-auto", type=float, default=None,
                    help="Target per-site total additions A; compute CIPSIAddPerCycle = round(A*N/Cycles).")
    ap.add_argument("--cipsi-cycles-auto", type=float, default=None,
                    help="Target per-site total additions A; compute CIPSICycles = round(A*N/AddPerCycle).")
    ap.add_argument("--cipsi-seed-mode", choices=["random","diag"], default=None,
                    help="Write CIPSISeedMode to modpara.def (random|diag)")
    ap.add_argument("--cipsi-seed-pool", type=int, default=None,
                    help="Write CIPSISeedPool to modpara.def (candidate pool size)")
    return ap
