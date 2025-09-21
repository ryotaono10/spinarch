# edcipsi_gen/cli.py
from __future__ import annotations
import os, sys, math, logging
from .argparsing import build_parser
from .parse import parse_spec, parse_cli_pairs          
from .lattice import build_interall, plot_lattice_and_vectors 
from .writers import write_greenone, write_greentwo, write_locspin, write_namelist, write_modpara_cipsi
from edcipsi_gen.cipsi import cipsi_big_defaults

log = logging.getLogger("edcipsi_gen")
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def main():
    log.info(f"[import] cli loaded: __name__={__name__} __file__={__file__}")
    log.info("[run] cli.main() start")

    parser = build_parser()
    args = parser.parse_args()

    if args.spec:
        Lx, Ly, a1, a2, items, outfile, plotfile = parse_spec(args.spec)
        if args.out != "interall.def":
            outfile = args.out
        if args.plot is not None:
            plotfile = args.plot
    else:
        if args.Lx is None or args.Ly is None or not args.pair:
            print("Either --spec OR (--Lx --Ly and at least one --pair) is required.", file=sys.stderr)
            sys.exit(2)
        Lx, Ly = args.Lx, args.Ly
        a1 = tuple(args.a1) if args.a1 is not None else (1.0, 0.0)
        a2 = tuple(args.a2) if args.a2 is not None else (0.5, math.sqrt(3)/2.0)
        items = parse_cli_pairs(args.pair)     # ← ["0 1 0 : 9 numbers"] → [(Rx,Ry,Rz,J(3x3))]
        outfile = args.out
        plotfile = args.plot if args.plot else f"lattice_{os.path.basename(outfile)}.png"

    N = Lx * Ly

    # InterAll
    build_interall(Lx, Ly, items, outfile, a1=a1, a2=a2)
    log.info(f"[OK] wrote InterAll to {outfile}")

    # GreenOne
    if getattr(args, "greenone", None):
        write_greenone(Lx, Ly, args.greenone)
        log.info(f"[OK] wrote OneBodyG to {args.greenone}")

    # GreenTwo
    if getattr(args, "greentwo", None):
        include_spin = not getattr(args, "no_spinflip", False)
        write_greentwo(Lx, Ly, args.greentwo, include_spinflip=include_spin)
        log.info(f"[OK] wrote TwoBodyG to {args.greentwo} (spinflip={include_spin})")

    # NameList
    nl_loc = args.locspin if args.locspin is not None else None
    nl_g1  = args.greenone if args.greenone is not None else None
    nl_g2  = args.greentwo if args.greentwo is not None else None
    write_namelist(args.namelist,modpara=args.modpara,interall=outfile,locspin=nl_loc,greenone=nl_g1,greentwo=nl_g2)
    log.info(f"[OK] wrote Namelist to {args.namelist}")

    # ModPara
    # Grand canonical default = True. CLI can flip either way.
    gc = True
    if args.cipsi_no_grand:
        gc = False
    if args.cipsi_grand:
        gc = True
    knobs = cipsi_big_defaults(N, grand=gc)
    write_modpara_cipsi(
        Lx, Ly, path=args.modpara, grand=gc,
        seeds=knobs["seeds"],
        cycles=knobs["cycles"],
        add_per=knobs["add_per_cycle"],
        seed_pool=knobs["seed_pool"],
    )
    log.info(f"[OK] wrote ModPara to {args.modpara}")

    # Lattice Plot
    plot_lattice_and_vectors(Lx, Ly, items, plotfile, a1=a1, a2=a2)
    log.info(f"[OK] wrote plot to {plotfile}")

    log.info("[run] cli.main() end")

if __name__ == "__main__":
    main()

