# edcipsi_gen/cipsi.py
import math

def cipsi_big_defaults(nsite: int, *, grand: bool = True,
                       A: int = 256, cycles: int | None = None,
                       add_per: int | None = None) -> dict:
    """
    できるだけ安全側（大きめ）に寄せた初期値を返す。
      - A: 1サイトあたりの総追加目標（既定 256 と大きめ）
      - cycles 未指定なら 64 固定（十分長め）
      - add_per は ceil で上振れ、8の倍数に切り上げ
      - Seeds は 6*sqrt(N) を基準に 32..256（4の倍数）
      - SeedPool は max(4*Seeds, 2N or N) を Seeds の倍数へ切上げ
    """
    if nsite <= 0:
        raise ValueError("nsite must be positive")

    # 1) Cycles / AddPerCycle
    if cycles is None and add_per is None:
        cycles = 64  # 大きめ固定
        add_per = math.ceil(A * nsite / cycles)
    elif cycles is None:
        cycles = max(64, math.ceil(A * nsite / max(1, add_per)))
    elif add_per is None:
        add_per = math.ceil(A * nsite / max(1, cycles))
    # 粒度を8に丸め上げ
    add_per = int(math.ceil(add_per / 8.0) * 8)
    cycles = int(cycles)

    Full = int(round(150*math.exp(2*math.sqrt(nsite))))
    Fullstep = int(Full/cycles)
    add_per = Fullstep

    # 2) Seeds（大きめ）
    seeds = int(round(Fullstep/nsite))

    # 3) SeedPool（母集団）も強め
    seed_pool = 4*seeds

    return {
        "seeds": seeds,
        "cycles": cycles,
        "add_per_cycle": add_per,
        "seed_pool": seed_pool,
        "A_target": A,
    }

