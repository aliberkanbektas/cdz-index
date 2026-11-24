"""
Minimisers of CDZ over chemical trees, and the evidence for Conjecture 3.16.

Reproduces
  * Table 1 (minimisers over chemical trees on 4..17 vertices: value,
    internal degrees, edge types present);
  * the minima for n = 6..13 quoted in Remark 3.7;
  * the verification of Conjecture 3.16 (no minimiser has an edge joining
    two vertices of equal degree) over all chemical trees with 4 <= n <= 17
    and over all trees, with no restriction on the maximum degree, with
    4 <= n <= 13.

A chemical tree is a tree of maximum degree at most 4.  All graphs here are
homogeneous, so CDZ reduces to its degree form (Proposition 2.5).

Run time: about two minutes, dominated by the 48 629 trees on 17 vertices.
"""

import os
import sys
from collections import Counter

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cdz_core import cdz  # noqa: E402

TOL = 1e-9

# Table 1 of the paper: n -> (value, internal degrees, edge types present)
TABLE_1 = {
    4:  (1.7321, [3],                [(1, 3)]),
    5:  (2.0000, [4],                [(1, 4)]),
    6:  (2.8081, [3, 2, 2],          [(1, 2), (1, 3), (2, 3)]),
    7:  (2.9434, [4, 3],             [(1, 3), (1, 4), (3, 4)]),
    8:  (3.4814, [4, 3, 2],          [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]),
    9:  (3.8868, [4, 3, 3],          [(1, 3), (1, 4), (3, 4)]),
    10: (4.1547, [4, 4, 3],          [(1, 3), (1, 4), (3, 4)]),
    11: (4.6927, [4, 4, 3, 2],       [(1, 2), (1, 4), (2, 3), (3, 4)]),
    12: (5.0981, [4, 4, 3, 3],       [(1, 3), (1, 4), (3, 4)]),
    13: (5.3660, [4, 4, 4, 3],       [(1, 4), (3, 4)]),
    14: (6.0415, [4, 4, 3, 3, 3],    [(1, 3), (1, 4), (3, 4)]),
    15: (6.3094, [4, 4, 4, 3, 3],    [(1, 3), (1, 4), (3, 4)]),
    16: (6.7599, [4, 4, 4, 3, 3, 2], [(1, 3), (1, 4), (2, 3), (3, 4)]),
    17: (7.2528, [4, 4, 4, 3, 3, 3], [(1, 3), (1, 4), (3, 4)]),
}
# Remark 3.7: true minimum over chemical trees, n = 6..13
REMARK_3_5 = [2.80806, 2.94338, 3.48138, 3.88675, 4.15470, 4.69271, 5.09808, 5.36603]


def internal_degrees(T):
    d = dict(T.degree())
    return sorted((d[v] for v in T if d[v] > 1), reverse=True)


def edge_types(T):
    d = dict(T.degree())
    return sorted(Counter(tuple(sorted((d[u], d[v]))) for u, v in T.edges()))


def has_equal_degree_edge(T):
    d = dict(T.degree())
    return any(d[u] == d[v] for u, v in T.edges())


def minimisers(trees):
    """Return (minimum value, list of all trees attaining it)."""
    vals = [(cdz(T), T) for T in trees]
    m = min(v for v, _ in vals)
    return m, [T for v, T in vals if abs(v - m) < TOL]


def main():
    ok = True
    print("Table 1 -- minimisers of CDZ over chemical trees\n")
    print(f"{'n':>3}{'#trees':>8}{'min CDZ':>10}{'#min':>5}  {'internal degrees':20s}edge types")
    print("-" * 96)
    chem_min = {}
    for n in range(4, 18):
        chem = [T for T in nx.nonisomorphic_trees(n) if max(dict(T.degree()).values()) <= 4]
        m, mins = minimisers(chem)
        chem_min[n] = m
        T = mins[0]
        ideg, et = internal_degrees(T), edge_types(T)
        exp_val, exp_ideg, exp_et = TABLE_1[n]
        row_ok = (round(m, 4) == exp_val and ideg == exp_ideg and et == exp_et
                  and all(internal_degrees(S) == ideg and edge_types(S) == et for S in mins))
        conj_ok = not any(has_equal_degree_edge(S) for S in mins)
        ok &= row_ok and conj_ok
        print(f"{n:3d}{len(chem):8d}{m:10.4f}{len(mins):5d}  {str(ideg):20s}{et}"
              f"   {'OK' if row_ok else 'MISMATCH'}"
              f"{'' if conj_ok else '  CONJECTURE FAILS'}")

    print("\nRemark 3.7 -- minimum over chemical trees, n = 6..13")
    got = [round(chem_min[n], 5) for n in range(6, 14)]
    r_ok = got == REMARK_3_5
    ok &= r_ok
    print(f"  {got}  {'OK' if r_ok else 'MISMATCH'}")

    print("\nConjecture 3.16 over all trees (no degree restriction), 4 <= n <= 13")
    for n in range(4, 14):
        trees = list(nx.nonisomorphic_trees(n))
        m, mins = minimisers(trees)
        conj_ok = not any(has_equal_degree_edge(S) for S in mins)
        ok &= conj_ok
        print(f"  n={n:2d}: {len(trees):5d} trees, min {m:.5f}, {len(mins)} minimiser(s), "
              f"{'no equal-degree edge -- OK' if conj_ok else 'CONJECTURE FAILS'}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
