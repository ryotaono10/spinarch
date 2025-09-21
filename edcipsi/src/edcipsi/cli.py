from __future__ import annotations
import os, sys, math, time, atexit, logging, random
import numpy as np
from .argparsing import build_parser  
from .utils import TeeWithTimestamp, _log_read
from .config import read_namelist, read_modpara
from .io import read_interall, read_greenone_def, read_greentwo_def
from .cipsi import run_cipsi_once, compute_PT2
from .observables import expect_greenone, expect_greentwo

log = logging.getLogger("edcipsi")
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def _setup_threads(threads: int | None) -> None:
    if threads is None: return
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(threads)
    print(f"[Threads] Set OMP/MKL threads = {threads}")

def _setup_outdir_and_tee():
    outdir = os.path.join(os.getcwd(), "output")
    os.makedirs(outdir, exist_ok=True)
    std_path = os.path.join(outdir, "std.out")
    f = open(std_path, "w", encoding="utf-8")
    old_out, old_err = sys.__stdout__, sys.__stderr__
    tee_out = TeeWithTimestamp(old_out, f)
    tee_err = TeeWithTimestamp(old_err, f)
    sys.stdout, sys.stderr = tee_out, tee_err
    def _cleanup():
        try: sys.stdout, sys.stderr = old_out, old_err
        except: pass
        try: tee_out.flush(); tee_err.flush()
        except: pass
        try: f.flush(); f.close()
        except: pass
    atexit.register(_cleanup)
    print("=== Start ED-CIPSI ===")
    print("Command:", " ".join(sys.argv))
    return outdir

def main():
    log.info(f"[import] cli loaded: __name__={__name__} __file__={__file__}")
    log.info("[run] cli.main() start")
    parser = build_parser()
    args = parser.parse_args()

    _setup_threads(args.threads)
    outdir = _setup_outdir_and_tee()

    # 入力
    _log_read(args.namelist)
    nl = read_namelist(args.namelist)

    modpara_path = nl["ModPara"]; _log_read(modpara_path)
    mp = read_modpara(modpara_path)
    N = mp["Nsite"]
    print(f"[OK] ModPara: Nsite={N}, Grand={mp['CIPSIGrandCanonical']}, Seeds={mp['CIPSISeeds']}, Cycles={mp['CIPSICycles']}")

    interall_path = nl["InterAll"]; _log_read(interall_path)
    t0 = time.perf_counter()
    diag_terms, bilinear_terms = read_interall(interall_path)
    print(f"[OK] InterAll: diag={len(diag_terms)} bilinear={len(bilinear_terms)} ({time.perf_counter()-t0:.3f}s)")

    greenone_path = nl.get("OneBodyG"); greentwo_path = nl.get("TwoBodyG")

    # 実行パラメータ解決（CLI優先）
    gc = bool(args.grand_canonical or mp["CIPSIGrandCanonical"])
    seeds = int(args.seeds if args.seeds is not None else mp["CIPSISeeds"])
    cycles = int(args.cycles if args.cycles is not None else mp["CIPSICycles"])
    add_per = int(args.add_per_cycle if args.add_per_cycle is not None else mp["CIPSIAddPerCycle"])
    prune_max = int(args.prune if args.prune is not None else mp["CIPSIPrune"])
    eps = float(args.eps if args.eps is not None else mp["CIPSIEps"])
    rngseed = int(args.seed if args.seed is not None else mp["CIPSIRandomSeed"])
    seed_mode = (args.seed_mode or mp.get("CIPSISeedMode", "random")).lower()
    seed_pool = int(args.seed_pool if args.seed_pool is not None else int(mp.get("CIPSISeedPool", 0)) or max(1024, 32*seeds))

    random.seed(rngseed); np.random.seed(rngseed & 0xFFFFFFFF)

    # HBプリセレクション設定
    hb_pre = bool(args.hb_preselect)
    max_abs_coeff = max((abs(t[-1]) for t in bilinear_terms), default=0.0)
    if hb_pre:
        hb_gamma = float(args.hb_gamma) if args.hb_gamma is not None else math.sqrt(max(eps, 0.0)) * max_abs_coeff
        bilinear_terms = sorted(bilinear_terms, key=lambda x: abs(x[-1]), reverse=True)
        print(f"[HB] Preselection ON: Gamma={hb_gamma:.3e}, max|alpha|={max_abs_coeff:.3e}, terms_sorted=True")
    else:
        hb_gamma = None
        print("[HB] Preselection OFF")

    # 実行
    E, vec, basis = run_cipsi_once(
        N, diag_terms, bilinear_terms,
        grand_canonical=gc, seeds=seeds, cycles=cycles, add_per_cycle=add_per, prune=prune_max, eps=eps,
        hb_gamma=hb_gamma, hb_sorted=hb_pre, max_abs_coeff=max_abs_coeff,
        threads=args.threads, accel_matvec=args.accel_matvec, nb_parallel=args.nb_parallel,
        build_blocked=args.build_blocked, block_size=args.block_size, build_procs=args.build_procs,
        seed_mode=seed_mode, seed_pool=seed_pool, sector_Sz=mp.get("CIPSISectorSz"), rng=random
    )

    print(f"[Final] Basis={len(basis)}, E0={E:.12f}  E0/site={(E.real)/N:.12f}")
    if args.pt2:
        M_final = {}  # 簡潔に：必要なら cipsi.connected_amplitudes を呼んで PT2 を再計算
        from .cipsi import connected_amplitudes
        M_final = connected_amplitudes(basis, vec, bilinear_terms, hb_gamma=hb_gamma, max_abs_coeff=max_abs_coeff, terms_sorted=hb_pre)
        Ept2_final, npt2_final = compute_PT2(E, M_final, diag_terms, level_shift=args.level_shift)
        print(f"[Final PT2] terms={npt2_final}  E_PT2={Ept2_final:+.6e}  E_var+PT2={E.real+Ept2_final:.12f}  per-site={(E.real+Ept2_final)/N:.12f}")

    # 出力
    energy_path = os.path.join(outdir, "energy.out")
    green1_path = os.path.join(outdir, "greenone.out")
    green2_path = os.path.join(outdir, "greentwo.out")

    with open(energy_path, "w", encoding="utf-8") as fE:
        fE.write(f"# N={N}\n# BasisSize={len(basis)}\n")
        fE.write(f"E0 {E.real:.16e} {E.imag:.3e}\n")

    if greenone_path is not None:
        ops1 = read_greenone_def(greenone_path)
        vals1 = expect_greenone(basis, vec, ops1)
        with open(green1_path, "w", encoding="utf-8") as f1:
            for ((i,si,j,sj), v) in zip(ops1, vals1):
                f1.write(f"{i:5d}{si:5d}{j:5d}{sj:5d} {v.real: .10f} {v.imag: .10f}\n")
    else:
        open(green1_path, "w", encoding="utf-8").close()

    if greentwo_path is not None:
        ops = read_greentwo_def(greentwo_path)
        vals = expect_greentwo(basis, vec, N, ops)
        with open(green2_path, "w", encoding="utf-8") as fG:
            for ((i,si,j,sj,k,sk,l,sl), v) in zip(ops, vals):
                fG.write(f"{i:5d}{si:5d}{j:5d}{sj:5d}{k:5d}{sk:5d}{l:5d}{sl:5d} {v.real: .10f} {v.imag: .10f}\n")
    else:
        open(green2_path, "w", encoding="utf-8").close()

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(f"# N={N}\n# BasisSize={len(basis)}\n")
            f.write(f"E0 {E.real:.16e} {E.imag:.3e}\n")

    done_msg = f"[DONE] Wrote energy to {energy_path}"
    if greentwo_path is not None: done_msg += f" and greentwo to {green2_path}"
    if greenone_path is not None: done_msg += f" and greenone to {green1_path}"
    print(done_msg)
    log.info("[run] cli.main() end")

if __name__ == "__main__":
    main()
