from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# Pauli
sigma_x = np.array([[0+0j, 1+0j],[1+0j, 0+0j]], dtype=complex)
sigma_y = np.array([[0+0j, 0-1j],[0+1j, 0+0j]], dtype=complex)
sigma_z = np.array([[1+0j, 0+0j],[0+0j, -1+0j]], dtype=complex)
PAULI = [sigma_x, sigma_y, sigma_z]
EPS = 1e-14


@dataclass(frozen=True)
class TriRhombus:
    Lx: int
    Ly: int
    a1: Tuple[float,float] = (1.0, 0.0)
    a2: Tuple[float,float] = (0.5, math.sqrt(3)/2.0)
    def nsite(self): return self.Lx * self.Ly
    def idx(self, x:int, y:int): return (x % self.Lx) + self.Lx * (y % self.Ly)
    def all_sites(self) -> Iterable[Tuple[int,int,int]]:
        for y in range(self.Ly):
            for x in range(self.Lx):
                yield (self.idx(x,y), x, y)
    def pos(self, x:int, y:int):
        return (x*self.a1[0] + y*self.a2[0], x*self.a1[1] + y*self.a2[1])

def coeff_from_J(alpha:int, beta:int, gamma:int, delta:int, J: np.ndarray) -> complex:
    val = 0+0j
    for a in range(3):
        for b in range(3):
            val += 0.25 * J[a,b] * PAULI[a][alpha,beta] * PAULI[b][gamma,delta]
    return val

def build_interall(Lx:int, Ly:int, items, outfile:str, a1:tuple, a2:tuple):
    tri = TriRhombus(Lx, Ly, a1=a1, a2=a2)
    all_entries = []
    for (Rx, Ry, Rz, J) in items:
        selfinv = is_self_inverse(Rx, Ry, Lx, Ly)
        for (_, x, y) in tri.all_sites():
            i = tri.idx(x, y)
            j = tri.idx(x + Rx, y + Ry)  # 2DなのでRzは無視
            if selfinv:
                if i < j:
                    all_entries.extend(entries_for_oriented_bond(i, j, J, add_conj=True))
            else:
                all_entries.extend(entries_for_oriented_bond(i, j, J, add_conj=True))

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("======================\n")
        f.write(f"NInterAll {len(all_entries)}\n")
        f.write("======================\n")
        f.write("========zInterAll=====\n")
        f.write("======================\n")
        for e in all_entries:
            f.write("{:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {: .15g} {: .15g}\n".format(*e))

def is_self_inverse(Rx: int, Ry: int, Lx: int, Ly: int) -> bool:
    return ((2*Rx) % Lx == 0) and ((2*Ry) % Ly == 0)

def entries_for_oriented_bond(i:int, j:int, J: np.ndarray, add_conj: bool):
    entries = []
    for a in (0,1):
        for b in (0,1):
            for g in (0,1):
                for d in (0,1):
                    c = coeff_from_J(a,b,g,d,J)
                    if abs(c) < EPS:
                        continue
                    entries.append((i,a, i,b, j,g, j,d, float(c.real), float(c.imag)))
                    if add_conj:
                        cc = complex(c.real, -c.imag)
                        entries.append((j,d, j,g, i,b, i,a, float(cc.real), float(cc.imag)))
    return entries

def plot_lattice_and_vectors(Lx:int, Ly:int, items, png_path:str, a1:Tuple[float,float], a2:Tuple[float,float], annotate_sites=True):
    tri = TriRhombus(Lx, Ly, a1=a1, a2=a2)
    xs, ys, labels = [], [], []
    for (i,x,y) in tri.all_sites():
        X,Y = tri.pos(x,y)
        xs.append(X); ys.append(Y); labels.append(str(i))

    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.scatter(xs, ys, s=20)

    if annotate_sites and tri.nsite() <= 100:
        for i,(X,Y) in enumerate(zip(xs, ys)):
            ax.text(X, Y, labels[i], fontsize=8, ha='center', va='bottom')

    a1x,a1y = tri.a1; a2x,a2y = tri.a2
    ax.arrow(0,0, a1x,a1y, length_includes_head=True, head_width=0.08); ax.text(a1x, a1y, "a1", fontsize=10)
    ax.arrow(0,0, a2x,a2y, length_includes_head=True, head_width=0.08); ax.text(a2x, a2y, "a2", fontsize=10)

    for (Rx, Ry, Rz, J) in items:
        Rxp = Rx*tri.a1[0] + Ry*tri.a2[0]; Ryp = Rx*tri.a1[1] + Ry*tri.a2[1]
        ax.arrow(0,0, Rxp,Ryp, length_includes_head=True, head_width=0.08)
        ax.text(Rxp, Ryp, f"R=({Rx},{Ry},{Rz})", fontsize=9)

    margin_x = 0.05 * (max(xs) - min(xs) if xs else 1.0)
    margin_y = 0.05 * (max(ys) - min(ys) if ys else 1.0)
    ax.set_xlim(min(xs)-margin_x, max(xs)+margin_x if xs else 1.0)
    ax.set_ylim(min(ys)-margin_y, max(ys)+margin_y if ys else 1.0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x (embedded)"); ax.set_ylabel("y (embedded)")
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
