"""
Remark 3.22: no single edge relocation improves the tree H_12, which is far
from optimal.

H_12 has two vertices of degree 3 joined by a path with two internal
vertices; each degree-3 vertex carries two further pendant paths of length 2.
The script checks

  * CDZ(H_12) = 1 + sqrt6 + 2 sqrt2 (one (2,2), six (2,3), four (1,2) edges);
  * no tree obtained from H_12 by deleting one edge and reconnecting the two
    components by a different edge, subject to maximum degree <= 4, has
    smaller CDZ: H_12 is a local minimum under single-edge relocation;
  * the minimum is not strict: up to isomorphism exactly one relocation
    gives a different tree with the same value, and from that tree a single
    relocation does decrease CDZ, so two moves suffice to leave H_12;
  * the minimum over all chemical trees on 12 vertices is 5.0981, so H_12
    is not a global minimiser.
"""

import os
import sys
from collections import Counter
from math import sqrt

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cdz_core import cdz  # noqa: E402


def build_h12():
    H = nx.Graph()
    H.add_edges_from([(0, 1), (1, 2), (2, 3)])          # 0 and 3 have degree 3
    nxt = 4
    for hub in (0, 3):
        for _ in range(2):
            H.add_edge(hub, nxt)
            H.add_edge(nxt, nxt + 1)
            nxt += 2
    return H


def edge_relocations(T, max_degree=4):
    """All trees obtained by deleting one edge and reconnecting differently."""
    for a, b in list(T.edges()):
        H = T.copy()
        H.remove_edge(a, b)
        A, B = (list(c) for c in nx.connected_components(H))
        for u in A:
            for v in B:
                if {u, v} == {a, b}:
                    continue
                if H.degree(u) + 1 > max_degree or H.degree(v) + 1 > max_degree:
                    continue
                K = H.copy()
                K.add_edge(u, v)
                yield K


def main():
    H = build_h12()
    d = dict(H.degree())
    et = Counter(tuple(sorted((d[u], d[v]))) for u, v in H.edges())
    val = cdz(H)
    exact = 1 + sqrt(6) + 2 * sqrt(2)
    c1 = (H.number_of_nodes() == 12 and dict(et) == {(2, 2): 1, (2, 3): 6, (1, 2): 4}
          and abs(val - exact) < 1e-12)
    print(f"H_12: {H.number_of_nodes()} vertices, edge types {dict(sorted(et.items()))}")
    print(f"  [{'OK' if c1 else 'FAIL'}] CDZ(H_12) = {val:.5f} = 1 + sqrt6 + 2 sqrt2")

    neighbours = [K for K in edge_relocations(H) if not nx.is_isomorphic(K, H)]
    worst = min(cdz(K) for K in neighbours)
    c2 = worst > val - 1e-12
    print(f"  [{'OK' if c2 else 'FAIL'}] none of the {len(neighbours)} non-isomorphic relocations has "
          f"smaller CDZ (smallest {worst:.5f})")
    plateau = []
    for K in neighbours:
        if abs(cdz(K) - val) < 1e-12 and not any(nx.is_isomorphic(K, P) for P in plateau):
            plateau.append(K)
    escape = min(cdz(L) for L in edge_relocations(plateau[0])) if plateau else None
    c2 = c2 and len(plateau) == 1 and escape is not None and escape < val - 1e-12
    print(f"  [{'OK' if c2 else 'FAIL'}] equal-valued neighbours up to isomorphism: {len(plateau)}; "
          f"from it one relocation reaches {escape:.5f} < {val:.5f}")

    chem12 = [T for T in nx.nonisomorphic_trees(12) if max(dict(T.degree()).values()) <= 4]
    m12 = min(cdz(T) for T in chem12)
    c3 = round(m12, 4) == 5.0981 and m12 < val
    print(f"  [{'OK' if c3 else 'FAIL'}] minimum over the {len(chem12)} chemical trees on 12 "
          f"vertices is {m12:.4f} < {val:.4f}")

    ok = c1 and c2 and c3
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
