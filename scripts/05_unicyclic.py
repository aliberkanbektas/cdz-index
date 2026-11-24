"""
Minimisers of CDZ over connected unicyclic graphs (Conjecture 3.25).

Every connected unicyclic graph on n vertices is a spanning tree plus one
edge, so adding each missing edge to each tree on n vertices enumerates all
of them (with repetitions, which do not affect the minimum).  For
4 <= n <= 12 the script reports the minimum and the minimisers, and checks
the claims of Section 3.6 (unicyclic graphs):

  * for 9 <= n <= 12 the minimiser is a triangle with pendant vertices,
    TP(n; p, q, r), with triangle degrees (2,3,7), (2,2,9), (2,5,7), (2,2,11)
    and values 3.36090, 3.80474, 3.98398, 4.14000;
  * for n in {5, 6, 8} the minimiser is not of that form (for n = 6 and
    n = 8 its cycle has length 4), whereas for n = 4 and n = 7 it is, with
    triangle degrees (2,2,3) and (2,3,5).

No degree restriction is imposed.
"""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cdz_core import cdz  # noqa: E402

TOL = 1e-9
PAPER = {4: ((2, 2, 3), 2.39385), 7: ((2, 3, 5), 2.90167),
         9: ((2, 3, 7), 3.36090), 10: ((2, 2, 9), 3.80474),
         11: ((2, 5, 7), 3.98398), 12: ((2, 2, 11), 4.14000)}


def unicyclic_graphs(n):
    for T in nx.nonisomorphic_trees(n):
        nodes = list(T)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if not T.has_edge(u, v):
                    G = T.copy()
                    G.add_edge(u, v)
                    yield G


def describe(G):
    """Cycle length, and triangle degrees if G is a triangle with pendants."""
    cycle = nx.cycle_basis(G)[0]
    d = dict(G.degree())
    on_cycle = set(cycle)
    is_tp = (len(cycle) == 3 and
             all(d[v] == 1 and next(iter(G[v])) in on_cycle for v in G if v not in on_cycle))
    tri = tuple(sorted(d[v] for v in cycle)) if is_tp else None
    return len(cycle), is_tp, tri


def main():
    ok = True
    print(f"{'n':>3}{'min CDZ':>10}  {'cycle':>5}  TP form  triangle degrees")
    for n in range(4, 13):
        best, mins = None, []
        for G in unicyclic_graphs(n):
            v = cdz(G)
            if best is None or v < best - TOL:
                best, mins = v, [G]
            elif abs(v - best) < TOL:
                mins.append(G)
        # deduplicate minimisers up to isomorphism
        uniq = []
        for G in mins:
            if not any(nx.is_isomorphic(G, K) for K in uniq):
                uniq.append(G)
        descr = [describe(G) for G in uniq]
        cyc = sorted({c for c, _, _ in descr})
        tp = [t for _, is_tp, t in descr if is_tp]
        line = f"{n:3d}{best:10.5f}  {str(cyc):>5}  {'yes' if tp else 'no ':3s}      {tp if tp else ''}"
        if n in PAPER:
            exp_tri, exp_val = PAPER[n]
            row_ok = tp == [exp_tri] and abs(best - exp_val) < 5e-6 and len(uniq) == 1
            line += "   OK" if row_ok else f"   MISMATCH (paper {exp_tri}, {exp_val})"
            ok &= row_ok
        else:
            row_ok = not tp
            if n in (6, 8):
                row_ok &= cyc == [4]
            line += "   OK" if row_ok else "   MISMATCH"
            ok &= row_ok
        print(line)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
