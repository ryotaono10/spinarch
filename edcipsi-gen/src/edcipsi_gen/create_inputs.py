#!/usr/bin/env python3
# === CIPSI seed defaults & formatting helpers ===
import math as _math


import sys, os, math
from dataclasses import dataclass
from typing import List, Tuple, Iterable
import numpy as np

# プロットは任意依存
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

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

def main():
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
        items = parse_cli_pairs(args.pair)
        outfile = args.out
        plotfile = args.plot if args.plot else f"lattice_{os.path.basename(outfile)}.png"

    # 出力
    build_interall(Lx, Ly, items, outfile, a1=a1, a2=a2)
    if args.plot is not None and HAVE_MPL:
        plot_lattice_and_vectors(Lx, Ly, items, plotfile, a1=a1, a2=a2, annotate_sites=True)

    if args.locspin is not None:
        if args.twosz is None:
            print("[error] --locspin を指定したら --twosz も必要です。", file=sys.stderr)
            sys.exit(2)
        write_locspin(Lx, Ly, int(args.twosz), args.locspin)

    if args.for_cipsi:
        # --- Auto-scaling (optional) ---
        N = Lx * Ly
        cycles = int(args.cipsi_cycles)
        add_per = int(args.cipsi_add)
        # If user supplied --cipsi-add-auto=A, compute AddPerCycle from A and current cycles
        if args.cipsi_add_auto is not None:
            A = float(args.cipsi_add_auto)
            add_per = max(1, int(round(A * N / max(cycles,1))))
            print(f"[auto] Using A={A} -> AddPerCycle={add_per} (N={N}, Cycles={cycles})")
        # If user supplied --cipsi-cycles-auto=A, compute Cycles from A and (possibly updated) add_per
        if args.cipsi_cycles_auto is not None:
            A2 = float(args.cipsi_cycles_auto)
            cycles = max(1, int(round(A2 * N / max(add_per,1))))
            print(f"[auto] Using A={A2} -> Cycles={cycles} (N={N}, AddPerCycle={add_per})")
    
        # Grand canonical default = True. CLI can flip either way.
        gc = True
        if args.cipsi_no_grand:
            gc = False
        if args.cipsi_grand:
            gc = True

        seed_mode = locals().get('seed_mode', None)
        seed_pool = locals().get('seed_pool', None)
        write_modpara_cipsi(Lx, Ly, args.modpara,
                            grand=gc,
                            seeds=args.cipsi_seeds,
                            cycles=cycles,
                            add_per=add_per,
                            prune=args.cipsi_prune,
                            eps=args.cipsi_eps,
                            sector_sz=args.cipsi_sector_sz,
                            rng=args.cipsi_seed)
        write_calcmod_cipsi(args.calcmod)

        nl_loc = args.locspin if args.locspin is not None else None
        nl_g1  = args.greenone if args.greenone is not None else None  
        nl_g2  = args.greentwo if args.greentwo is not None else None  
        write_namelist(args.namelist, args.modpara, nl_loc, outfile, nl_g1, nl_g2, args.calcmod)

    print(f"a1={a1}, a2={a2}")
    print(f"Wrote: {outfile}")
    if args.plot is not None and HAVE_MPL: print(f"Wrote: {plotfile}")
    if args.locspin:  print(f"Wrote: {args.locspin}")
    if args.for_cipsi:
        # --- Auto-scaling (optional) ---
        N = Lx * Ly
        cycles = int(args.cipsi_cycles)
        add_per = int(args.cipsi_add)
        # If user supplied --cipsi-add-auto=A, compute AddPerCycle from A and current cycles
        if args.cipsi_add_auto is not None:
            A = float(args.cipsi_add_auto)
            add_per = max(1, int(round(A * N / max(cycles,1))))
            print(f"[auto] Using A={A} -> AddPerCycle={add_per} (N={N}, Cycles={cycles})")
        # If user supplied --cipsi-cycles-auto=A, compute Cycles from A and (possibly updated) add_per
        if args.cipsi_cycles_auto is not None:
            A2 = float(args.cipsi_cycles_auto)
            cycles = max(1, int(round(A2 * N / max(add_per,1))))
            print(f"[auto] Using A={A2} -> Cycles={cycles} (N={N}, AddPerCycle={add_per})")
    
        print(f"Wrote: {args.modpara}")
        print(f"Wrote: {args.calcmod}")
        print(f"Wrote: {args.namelist}")

if __name__ == "__main__":
    main()

